import 'dart:convert';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart' show rootBundle;

import 'package:flutter_app/core/native_har_bridge.dart';
import 'package:flutter_app/services/comm_logger.dart';
import 'package:flutter_app/services/fl_api_client.dart';
import 'package:flutter_app/services/model_manager.dart';
import 'package:flutter_app/services/sensor_service.dart';

enum _TrainingSource { demoBundle, liveCaptured }

class TrainScreen extends StatefulWidget {
  final NativeHarBridge bridge;
  final ModelManager modelManager;
  final SensorService sensorService;
  final FlApiClient apiClient;
  final CommLogger logger;

  const TrainScreen({
    super.key,
    required this.bridge,
    required this.modelManager,
    required this.sensorService,
    required this.apiClient,
    required this.logger,
  });

  @override
  State<TrainScreen> createState() => _TrainScreenState();
}

class _TrainScreenState extends State<TrainScreen> {
  _TrainingSource _source = _TrainingSource.demoBundle;
  int _epochs = 5;
  int _liveLabelIndex = 2; // Walking default
  bool _isTraining = false;

  double? _preAcc;
  double? _postAcc;
  double? _finalLoss;
  double? _trainTimeMs;
  String? _submitStatus;

  Future<({Float32List features, Int32List labels, String inputMode})>
      _prepareTrainingData() async {
    if (_source == _TrainingSource.liveCaptured) {
      final windows = widget.sensorService.capturedFeatureWindows;
      if (windows.isEmpty) {
        throw StateError('No live sensor windows buffered yet. Keep Monitor running.');
      }
      final n = windows.length;
      final flat = Float32List(n * HarConstants.inDim);
      final labels = Int32List(n);
      for (var i = 0; i < n; i++) {
        flat.setRange(i * HarConstants.inDim, (i + 1) * HarConstants.inDim, windows[i]);
        labels[i] = _liveLabelIndex;
      }
      return (features: flat, labels: labels, inputMode: 'sensor');
    }

    // Bundled MotionSense demo asset (already in g units, scaleToG = 1.0)
    final jsonStr = await rootBundle.loadString('assets/mobile_demo_bundle.json');
    final decoded = jsonDecode(jsonStr) as Map<String, dynamic>;
    final rawWindows = decoded['windows_xyz'] as List<dynamic>;
    final rawLabels = decoded['labels'] as List<dynamic>;

    final n = rawLabels.length;
    final flat = Float32List(n * HarConstants.inDim);
    final labels = Int32List(n);

    for (var wIdx = 0; wIdx < n; wIdx++) {
      final win = rawWindows[wIdx] as List<dynamic>;
      final interleaved = List<double>.filled(HarConstants.inDim * 3, 0.0);
      for (var t = 0; t < HarConstants.inDim; t++) {
        final xyz = win[t] as List<dynamic>;
        interleaved[t * 3 + 0] = (xyz[0] as num).toDouble();
        interleaved[t * 3 + 1] = (xyz[1] as num).toDouble();
        interleaved[t * 3 + 2] = (xyz[2] as num).toDouble();
      }
      final mags = widget.bridge.preprocessWindow(
        interleaved,
        windowLen: HarConstants.inDim,
        scaleToG: 1.0,
      );
      flat.setRange(wIdx * HarConstants.inDim, (wIdx + 1) * HarConstants.inDim, mags);
      labels[wIdx] = (rawLabels[wIdx] as num).toInt();
    }

    return (features: flat, labels: labels, inputMode: 'dataset');
  }

  Future<void> _trainAndSubmit() async {
    final baseWeights = widget.modelManager.weights;
    if (baseWeights == null || _isTraining) return;

    setState(() {
      _isTraining = true;
      _submitStatus = null;
    });

    try {
      final baseVersion = widget.modelManager.version;
      final prepared = await _prepareTrainingData();
      final sampleCount = prepared.labels.length;
      final epochs = _epochs;
      final weightsCopy = Float32List.fromList(baseWeights);
      final flatFeatures = prepared.features;
      final labels = prepared.labels;
      final seed = widget.modelManager.clientId.hashCode ^ baseVersion;

      widget.logger.log(
        'Starting on-device training: $sampleCount windows (${prepared.inputMode}), $epochs epochs...',
      );

      final result = await Isolate.run(() {
        final isolateBridge = NativeHarBridge();
        return isolateBridge.trainEpochs(
          initialWeights: weightsCopy,
          flatFeatures: flatFeatures,
          labels: labels,
          epochs: epochs,
          batchSize: 32,
          lr: 0.05,
          seed: seed,
        );
      });

      widget.logger.log(
        'Trained $epochs epochs in ${result.trainTimeMs.toStringAsFixed(1)}ms '
        '(pre_acc=${(result.preAcc * 100).toStringAsFixed(1)}%, '
        'post_acc=${(result.postAcc * 100).toStringAsFixed(1)}%, '
        'loss=${result.finalLoss.toStringAsFixed(4)})',
      );

      final status = await widget.apiClient.submitWeights(
        weights: result.updatedWeights,
        baseVersion: baseVersion,
        sampleCount: sampleCount,
        localLoss: result.finalLoss,
        localAccuracy: result.postAcc,
        trainTimeMs: result.trainTimeMs,
        inputMode: prepared.inputMode,
      );

      if (!mounted) return;
      setState(() {
        _preAcc = result.preAcc;
        _postAcc = result.postAcc;
        _finalLoss = result.finalLoss;
        _trainTimeMs = result.trainTimeMs;
        _submitStatus = status;
      });
    } catch (e) {
      widget.logger.log('Training error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Training failed: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isTraining = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ListenableBuilder(
      listenable: Listenable.merge([widget.modelManager, widget.sensorService]),
      builder: (context, _) {
        final capturedCount = widget.sensorService.capturedFeatureWindows.length;
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Card(
              elevation: 0,
              color: theme.colorScheme.surfaceContainerHighest.withAlpha(110),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'On-Device Training Dataset',
                      style: theme.textTheme.titleMedium,
                    ),
                    const SizedBox(height: 12),
                    SegmentedButton<_TrainingSource>(
                      segments: [
                        const ButtonSegment(
                          value: _TrainingSource.demoBundle,
                          label: Text('Demo Bundle (180)'),
                          icon: Icon(Icons.dataset),
                        ),
                        ButtonSegment(
                          value: _TrainingSource.liveCaptured,
                          label: Text('Live ($capturedCount)'),
                          icon: const Icon(Icons.sensors),
                        ),
                      ],
                      selected: {_source},
                      onSelectionChanged: _isTraining
                          ? null
                          : (selection) =>
                              setState(() => _source = selection.first),
                    ),
                    if (_source == _TrainingSource.liveCaptured) ...[
                      const SizedBox(height: 12),
                      DropdownButtonFormField<int>(
                        initialValue: _liveLabelIndex,
                        decoration: const InputDecoration(
                          labelText: 'Activity Label for Captured Windows',
                          border: OutlineInputBorder(),
                          isDense: true,
                        ),
                        items: List.generate(
                          HarConstants.outDim,
                          (i) => DropdownMenuItem(
                            value: i,
                            child: Text(HarConstants.classNames[i]),
                          ),
                        ),
                        onChanged: _isTraining
                            ? null
                            : (val) {
                                if (val != null) {
                                  setState(() => _liveLabelIndex = val);
                                }
                              },
                      ),
                    ],
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Text(
                          'Local Epochs: $_epochs',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        Expanded(
                          child: Slider(
                            value: _epochs.toDouble(),
                            min: 1,
                            max: 20,
                            divisions: 19,
                            label: '$_epochs',
                            onChanged: _isTraining
                                ? null
                                : (v) => setState(() => _epochs = v.round()),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _isTraining ? null : _trainAndSubmit,
                        icon: _isTraining
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              )
                            : const Icon(Icons.model_training),
                        label: Text(
                          _isTraining
                              ? 'Training in C99 Isolate...'
                              : 'Train & Submit Weights',
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            if (_finalLoss != null) ...[
              const SizedBox(height: 12),
              Card(
                elevation: 0,
                color: theme.colorScheme.secondaryContainer.withAlpha(110),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Text(
                            'Last Training Run',
                            style: theme.textTheme.titleMedium,
                          ),
                          const Spacer(),
                          if (_submitStatus != null)
                            Chip(
                              label: Text(_submitStatus!.toUpperCase()),
                              visualDensity: VisualDensity.compact,
                            ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          _MetricBox(
                            label: 'Pre-train Acc',
                            value: '${(_preAcc! * 100).toStringAsFixed(1)}%',
                          ),
                          _MetricBox(
                            label: 'Post-train Acc',
                            value: '${(_postAcc! * 100).toStringAsFixed(1)}%',
                          ),
                          _MetricBox(
                            label: 'Final Loss',
                            value: _finalLoss!.toStringAsFixed(4),
                          ),
                          _MetricBox(
                            label: 'Train Time',
                            value: '${_trainTimeMs!.toStringAsFixed(1)} ms',
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ],
        );
      },
    );
  }
}

class _MetricBox extends StatelessWidget {
  final String label;
  final String value;

  const _MetricBox({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Expanded(
      child: Column(
        children: [
          Text(
            label,
            style: theme.textTheme.labelSmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: theme.textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.bold,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}
