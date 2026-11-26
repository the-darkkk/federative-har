import 'dart:async';
import 'dart:collection';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:sensors_plus/sensors_plus.dart';

import 'package:flutter_app/core/native_har_bridge.dart';
import 'package:flutter_app/services/comm_logger.dart';
import 'package:flutter_app/services/model_manager.dart';

class SensorService extends ChangeNotifier {
  static const int windowSize = HarConstants.inDim; // 128
  static const int stride = 32; // 0.64s @ 50Hz

  final NativeHarBridge bridge;
  final ModelManager modelManager;
  final CommLogger logger;

  final ListQueue<List<double>> _ringBuffer = ListQueue<List<double>>(windowSize + 4);
  final List<Float32List> _capturedFeatureWindows = [];

  StreamSubscription<AccelerometerEvent>? _sub;
  bool _isStreaming = false;
  String? _sensorError;

  double _x = 0.0;
  double _y = 0.0;
  double _z = 0.0;
  double _magnitudeG = 0.0;

  int _samplesSinceInfer = 0;
  DateTime _lastUiNotify = DateTime.fromMillisecondsSinceEpoch(0);

  int _predictedIndex = -1;
  List<double> _probabilities = List<double>.filled(HarConstants.outDim, 0.0);

  SensorService({
    required this.bridge,
    required this.modelManager,
    required this.logger,
  });

  bool get isStreaming => _isStreaming;
  String? get sensorError => _sensorError;
  int get bufferFill => _ringBuffer.length;
  double get x => _x;
  double get y => _y;
  double get z => _z;
  double get magnitudeG => _magnitudeG;
  int get predictedIndex => _predictedIndex;
  String get predictedLabel =>
      (_predictedIndex >= 0 && _predictedIndex < HarConstants.classNames.length)
          ? HarConstants.classNames[_predictedIndex]
          : 'Buffering...';
  double get confidence => (_predictedIndex >= 0) ? _probabilities[_predictedIndex] : 0.0;
  List<double> get probabilities => List<double>.unmodifiable(_probabilities);
  List<Float32List> get capturedFeatureWindows =>
      List<Float32List>.unmodifiable(_capturedFeatureWindows);

  void start() {
    if (_isStreaming) return;
    _sensorError = null;
    try {
      _sub = accelerometerEventStream(
        samplingPeriod: const Duration(milliseconds: 20),
      ).listen(
        _onSensorEvent,
        onError: (Object err) {
          _sensorError = err.toString();
          _isStreaming = false;
          logger.log('Accelerometer stream error: $err');
          notifyListeners();
        },
      );
      _isStreaming = true;
      notifyListeners();
    } catch (e) {
      _sensorError = e.toString();
      _isStreaming = false;
      logger.log('Failed to start accelerometer: $e');
      notifyListeners();
    }
  }

  void stop() {
    _sub?.cancel();
    _sub = null;
    _isStreaming = false;
    notifyListeners();
  }

  void _onSensorEvent(AccelerometerEvent event) {
    _x = event.x;
    _y = event.y;
    _z = event.z;
    _magnitudeG =
        sqrt(_x * _x + _y * _y + _z * _z) / HarConstants.gravityMs2;

    if (_ringBuffer.length >= windowSize) {
      _ringBuffer.removeFirst();
    }
    _ringBuffer.addLast([_x, _y, _z]);
    _samplesSinceInfer++;

    var ranInference = false;
    if (_ringBuffer.length == windowSize && _samplesSinceInfer >= stride) {
      _samplesSinceInfer = 0;
      _runInference();
      ranInference = true;
    }

    final now = DateTime.now();
    if (ranInference || now.difference(_lastUiNotify).inMilliseconds >= 80) {
      _lastUiNotify = now;
      notifyListeners();
    }
  }

  void _runInference() {
    final weights = modelManager.weights;
    if (weights == null) return;

    final rawInterleaved = List<double>.filled(windowSize * 3, 0.0);
    var idx = 0;
    for (final sample in _ringBuffer) {
      rawInterleaved[idx++] = sample[0];
      rawInterleaved[idx++] = sample[1];
      rawInterleaved[idx++] = sample[2];
    }

    final features = bridge.preprocessWindow(
      rawInterleaved,
      windowLen: windowSize,
      scaleToG: 1.0 / HarConstants.gravityMs2,
    );

    if (_capturedFeatureWindows.length >= 60) {
      _capturedFeatureWindows.removeAt(0);
    }
    _capturedFeatureWindows.add(features);

    final probs = bridge.predict(weights, features);
    var bestIdx = 0;
    var bestVal = probs[0];
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > bestVal) {
        bestVal = probs[i];
        bestIdx = i;
      }
    }

    _probabilities = probs;
    _predictedIndex = bestIdx;
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }
}
