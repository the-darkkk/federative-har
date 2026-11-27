import 'dart:isolate';
import 'dart:typed_data';

import 'package:material_ui/material_ui.dart';

import 'package:flutter_app/core/native_har_bridge.dart';
import 'package:flutter_app/services/comm_logger.dart';
import 'package:flutter_app/services/dataset_manager.dart';
import 'package:flutter_app/services/fl_api_client.dart';
import 'package:flutter_app/services/model_manager.dart';
import 'package:flutter_app/services/sensor_service.dart';

enum _TrainingSource { datasetNpz, liveSensor }

class TrainScreen extends StatefulWidget {
  final ModelManager modelManager;
  final DatasetManager datasetManager;
  final SensorService sensorService;
  final FlApiClient apiClient;
  final CommLogger logger;

  const TrainScreen({
    super.key,
    required this.modelManager,
    required this.datasetManager,
    required this.sensorService,
    required this.apiClient,
    required this.logger,
  });

  @override
  State<TrainScreen> createState() => _TrainScreenState();
}

class _TrainScreenState extends State<TrainScreen> {
  _TrainingSource _source = _TrainingSource.datasetNpz;
  int _epochs = 5;
  int _liveLabelIndex = 2; // Walking default
  bool _isTraining = false;

  double? _preAcc;
  double? _postAcc;
  double? _finalLoss;
  double? _trainTimeMs;
  String? _submitStatus;

  Future<({Float32List flatX, Int32List labels, String inputMode})>
      _prepareTrainingData() async {
    if (_source == _TrainingSource.liveSensor) {
      final windows = widget.sensorService.capturedFeatureWindows;
      if (windows.isEmpty) {
        throw StateError('No live sensor windows captured yet.');
      }
      final n = windows.length;
      final flat = Float32List(n * HarConstants.inDim);
      final labels = Int32List(n);
      for (var i = 0; i < n; i++) {
        flat.setRange(
          i * HarConstants.inDim,
          (i + 1) * HarConstants.inDim,
          windows[i],
        );
        labels[i] = _liveLabelIndex;
      }
      return (flatX: flat, labels: labels, inputMode: 'sensor');
    }

    final npz = await widget.datasetManager.loadSelectedDataset();
    return (flatX: npz.flatX, labels: npz.labels, inputMode: 'dataset');
  }

  Future<void> _importNpz() async {
    final count = await widget.datasetManager.importNpzFiles();
    if (!mounted) return;
    if (count > 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Imported $count .npz dataset(s)')),
      );
    }
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
      final seed = widget.modelManager.clientId.hashCode ^ baseVersion;

      final params = TrainTaskParams(
        initialWeights: Float32List.fromList(baseWeights),
        flatX: prepared.flatX,
        labels: prepared.labels,
        numSamples: sampleCount,
        inDim: HarConstants.inDim,
        hidDim: HarConstants.hidDim,
        outDim: HarConstants.outDim,
        epochs: epochs,
        batchSize: 32,
        learningRate: 0.05,
        seed: seed,
      );

      widget.logger.log(
        'Starting on-device training: $sampleCount windows (${prepared.inputMode}), $epochs epochs...',
      );

      final result = await Isolate.run(() => runTrainingIsolate(params));

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
      listenable: Listenable.merge([
        widget.modelManager,
        widget.datasetManager,
        widget.sensorService,
      ]),
      builder: (context, _) {
        final datasets = widget.datasetManager.availableDatasets;
        final selected = widget.datasetManager.selectedDataset;
        final capturedCount =
            widget.sensorService.capturedFeatureWindows.length;

        final canTrain = !_isTraining &&
            widget.modelManager.weights != null &&
            ((_source == _TrainingSource.datasetNpz && selected != null) ||
                (_source == _TrainingSource.liveSensor && capturedCount > 0));

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
                      'On-Device Training Source',
                      style: theme.textTheme.titleMedium,
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: SegmentedButton<_TrainingSource>(
                        segments: const [
                          ButtonSegment(
                            value: _TrainingSource.datasetNpz,
                            label: Text('Dataset (.npz)'),
                            icon: Icon(Icons.folder_zip_outlined),
                          ),
                          ButtonSegment(
                            value: _TrainingSource.liveSensor,
                            label: Text('Live Sensor'),
                            icon: Icon(Icons.sensors),
                          ),
                        ],
                        selected: {_source},
                        onSelectionChanged: _isTraining
                            ? null
                            : (selection) =>
                                setState(() => _source = selection.first),
                      ),
                    ),
                    const SizedBox(height: 14),
                    if (_source == _TrainingSource.datasetNpz) ...[
                      if (datasets.isEmpty)
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.surface,
                            borderRadius: BorderRadius.circular(10),
                            border: Border.all(
                              color: theme.colorScheme.outlineVariant,
                            ),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'No .npz files found. Transfer client_*.npz files from data/processed/ to your phone and tap Import .npz.',
                                style: theme.textTheme.bodySmall,
                              ),
                              const SizedBox(height: 10),
                              OutlinedButton.icon(
                                onPressed: _isTraining ? null : _importNpz,
                                icon: const Icon(Icons.file_upload_outlined),
                                label: const Text('Import .npz'),
                              ),
                            ],
                          ),
                        )
                      else ...[
                        DropdownButtonFormField<String>(
                          initialValue: selected?.name,
                          isExpanded: true,
                          decoration: const InputDecoration(
                            labelText: 'Selected Dataset',
                            border: OutlineInputBorder(),
                            isDense: true,
                          ),
                          items: datasets
                              .map(
                                (d) => DropdownMenuItem(
                                  value: d.name,
                                  child: Text(
                                    d.displayLabel,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ),
                              )
                              .toList(),
                          onChanged: _isTraining
                              ? null
                              : (val) =>
                                  widget.datasetManager.selectDataset(val),
                        ),
                        const SizedBox(height: 10),
                        Row(
                          children: [
                            OutlinedButton.icon(
                              onPressed: _isTraining ? null : _importNpz,
                              icon: const Icon(
                                Icons.file_upload_outlined,
                                size: 18,
                              ),
                              label: const Text('Import .npz'),
                            ),
                            const Spacer(),
                            if (selected != null)
                              TextButton.icon(
                                onPressed: _isTraining
                                    ? null
                                    : () => widget.datasetManager
                                        .deleteDataset(selected.name),
                                icon: const Icon(
                                  Icons.delete_outline,
                                  size: 18,
                                ),
                                label: const Text('Remove'),
                              ),
                          ],
                        ),
                      ],
                    ] else ...[
                      Text(
                        'Captures live 50 Hz accelerometer windows from this phone and tags them with the selected activity below.',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Row(
                        children: [
                          Text(
                            'Captured windows: $capturedCount / 60',
                            style: theme.textTheme.bodyMedium?.copyWith(
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const Spacer(),
                          TextButton.icon(
                            onPressed: _isTraining || capturedCount == 0
                                ? null
                                : widget.sensorService.clearCapturedWindows,
                            icon: const Icon(Icons.clear, size: 18),
                            label: const Text('Clear'),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      DropdownButtonFormField<int>(
                        initialValue: _liveLabelIndex,
                        decoration: const InputDecoration(
                          labelText: 'Activity Label',
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
                          'Epochs: $_epochs',
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
                        onPressed: canTrain ? _trainAndSubmit : null,
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
                            label: 'Loss',
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
