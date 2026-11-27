import 'dart:ffi';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

// C function signatures
typedef _FhWeightsLenC = Int32 Function(Int32 inDim, Int32 hidDim, Int32 outDim);
typedef _FhWeightsLenDart = int Function(int inDim, int hidDim, int outDim);

typedef _FhInitWeightsC = Void Function(
  Pointer<Float> weights,
  Int32 inDim,
  Int32 hidDim,
  Int32 outDim,
  Uint32 seed,
);
typedef _FhInitWeightsDart = void Function(
  Pointer<Float> weights,
  int inDim,
  int hidDim,
  int outDim,
  int seed,
);

typedef _FhPreprocessWindowC = Void Function(
  Pointer<Float> rawXyz,
  Pointer<Float> outFeatures,
  Int32 windowLen,
  Float scaleToG,
);
typedef _FhPreprocessWindowDart = void Function(
  Pointer<Float> rawXyz,
  Pointer<Float> outFeatures,
  int windowLen,
  double scaleToG,
);

typedef _FhPredictC = Void Function(
  Pointer<Float> weights,
  Pointer<Float> features,
  Pointer<Float> probsOut,
  Int32 inDim,
  Int32 hidDim,
  Int32 outDim,
);
typedef _FhPredictDart = void Function(
  Pointer<Float> weights,
  Pointer<Float> features,
  Pointer<Float> probsOut,
  int inDim,
  int hidDim,
  int outDim,
);

typedef _FhTrainBatchC = Float Function(
  Pointer<Float> weights,
  Pointer<Float> x,
  Pointer<Int32> y,
  Int32 batchSize,
  Int32 inDim,
  Int32 hidDim,
  Int32 outDim,
  Float lr,
);
typedef _FhTrainBatchDart = double Function(
  Pointer<Float> weights,
  Pointer<Float> x,
  Pointer<Int32> y,
  int batchSize,
  int inDim,
  int hidDim,
  int outDim,
  double lr,
);

class HarConstants {
  static const int inDim = 128;
  static const int hidDim = 64;
  static const int outDim = 6;
  static const int weightsCount = 8646;
  static const int weightsByteSize = weightsCount * 4;
  static const double gravityMs2 = 9.80665;
  static const List<String> classNames = [
    'Downstairs',
    'Upstairs',
    'Walking',
    'Jogging',
    'Sitting',
    'Standing',
  ];
}

class TrainTaskParams {
  final Float32List initialWeights;
  final Float32List flatX;
  final Int32List labels;
  final int numSamples;
  final int inDim;
  final int hidDim;
  final int outDim;
  final int epochs;
  final int batchSize;
  final double learningRate;
  final int seed;

  const TrainTaskParams({
    required this.initialWeights,
    required this.flatX,
    required this.labels,
    required this.numSamples,
    this.inDim = HarConstants.inDim,
    this.hidDim = HarConstants.hidDim,
    this.outDim = HarConstants.outDim,
    required this.epochs,
    this.batchSize = 32,
    this.learningRate = 0.05,
    this.seed = 42,
  });
}

class TrainTaskResult {
  final Float32List updatedWeights;
  final double preAcc;
  final double postAcc;
  final double finalLoss;
  final double trainTimeMs;

  const TrainTaskResult({
    required this.updatedWeights,
    required this.preAcc,
    required this.postAcc,
    required this.finalLoss,
    required this.trainTimeMs,
  });
}

/// Top-level isolate entry point. Receives only transferable typed-data/primitive fields
/// and opens its own DynamicLibrary handle inside the worker isolate.
TrainTaskResult runTrainingIsolate(TrainTaskParams params) {
  final bridge = NativeHarBridge();
  final wLen = params.initialWeights.length;
  final n = params.numSamples;
  final inDim = params.inDim;
  final hidDim = params.hidDim;
  final outDim = params.outDim;
  final batchSize = params.batchSize;

  final wPtr = calloc<Float>(wLen);
  final xBatchPtr = calloc<Float>(batchSize * inDim);
  final yBatchPtr = calloc<Int32>(batchSize);

  try {
    wPtr.asTypedList(wLen).setAll(0, params.initialWeights);

    final preAcc = bridge.evaluateAccuracy(
      wPtr,
      params.flatX,
      params.labels,
      inDim: inDim,
      hidDim: hidDim,
      outDim: outDim,
    );

    final rng = Random(params.seed);
    final indices = List<int>.generate(n, (i) => i);
    final xBatchView = xBatchPtr.asTypedList(batchSize * inDim);
    final yBatchView = yBatchPtr.asTypedList(batchSize);

    final sw = Stopwatch()..start();
    var finalEpochLoss = 0.0;

    for (var epoch = 0; epoch < params.epochs; epoch++) {
      indices.shuffle(rng);
      var epochLossSum = 0.0;
      var numBatches = 0;

      for (var start = 0; start < n; start += batchSize) {
        final end = (start + batchSize < n) ? start + batchSize : n;
        final bs = end - start;
        for (var b = 0; b < bs; b++) {
          final idx = indices[start + b];
          xBatchView.setRange(
            b * inDim,
            (b + 1) * inDim,
            params.flatX,
            idx * inDim,
          );
          yBatchView[b] = params.labels[idx];
        }
        final loss = bridge._trainBatch(
          wPtr,
          xBatchPtr,
          yBatchPtr,
          bs,
          inDim,
          hidDim,
          outDim,
          params.learningRate,
        );
        epochLossSum += loss;
        numBatches++;
      }
      finalEpochLoss = epochLossSum / max(1, numBatches);
    }

    sw.stop();
    final postAcc = bridge.evaluateAccuracy(
      wPtr,
      params.flatX,
      params.labels,
      inDim: inDim,
      hidDim: hidDim,
      outDim: outDim,
    );
    final updatedWeights = Float32List.fromList(wPtr.asTypedList(wLen));

    return TrainTaskResult(
      updatedWeights: updatedWeights,
      preAcc: preAcc,
      postAcc: postAcc,
      finalLoss: finalEpochLoss,
      trainTimeMs: sw.elapsedMicroseconds / 1000.0,
    );
  } finally {
    calloc.free(wPtr);
    calloc.free(xBatchPtr);
    calloc.free(yBatchPtr);
  }
}

class TrainResult {
  final double preAcc;
  final double postAcc;
  final double finalLoss;
  final double trainTimeMs;
  final Float32List updatedWeights;

  const TrainResult({
    required this.preAcc,
    required this.postAcc,
    required this.finalLoss,
    required this.trainTimeMs,
    required this.updatedWeights,
  });
}

class NativeHarBridge {
  late final DynamicLibrary _lib;
  late final _FhWeightsLenDart _weightsLen;
  late final _FhInitWeightsDart _initWeights;
  late final _FhPreprocessWindowDart _preprocessWindow;
  late final _FhPredictDart _predict;
  late final _FhTrainBatchDart _trainBatch;

  NativeHarBridge({String? libraryPath}) {
    _lib = _openLibrary(libraryPath);
    _weightsLen = _lib
        .lookup<NativeFunction<_FhWeightsLenC>>('fh_weights_len')
        .asFunction();
    _initWeights = _lib
        .lookup<NativeFunction<_FhInitWeightsC>>('fh_init_weights')
        .asFunction();
    _preprocessWindow = _lib
        .lookup<NativeFunction<_FhPreprocessWindowC>>('fh_preprocess_window')
        .asFunction();
    _predict = _lib
        .lookup<NativeFunction<_FhPredictC>>('fh_predict')
        .asFunction();
    _trainBatch = _lib
        .lookup<NativeFunction<_FhTrainBatchC>>('fh_train_batch')
        .asFunction();
  }

  static DynamicLibrary _openLibrary(String? libraryPath) {
    if (libraryPath != null) {
      return DynamicLibrary.open(libraryPath);
    }
    if (Platform.isAndroid || Platform.isLinux) {
      return DynamicLibrary.open('libfederative_har.so');
    }
    if (Platform.isWindows) {
      final candidates = [
        'federative_har.dll',
        '../../core/build/federative_har.dll',
      ];
      for (final candidate in candidates) {
        if (File(candidate).existsSync()) {
          return DynamicLibrary.open(candidate);
        }
      }
      return DynamicLibrary.open('federative_har.dll');
    }
    if (Platform.isMacOS || Platform.isIOS) {
      return DynamicLibrary.open('libfederative_har.dylib');
    }
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  int weightsLen({
    int inDim = HarConstants.inDim,
    int hidDim = HarConstants.hidDim,
    int outDim = HarConstants.outDim,
  }) {
    return _weightsLen(inDim, hidDim, outDim);
  }

  Float32List initWeights({
    int inDim = HarConstants.inDim,
    int hidDim = HarConstants.hidDim,
    int outDim = HarConstants.outDim,
    int seed = 42,
  }) {
    final len = weightsLen(inDim: inDim, hidDim: hidDim, outDim: outDim);
    final ptr = calloc<Float>(len);
    try {
      _initWeights(ptr, inDim, hidDim, outDim, seed);
      return Float32List.fromList(ptr.asTypedList(len));
    } finally {
      calloc.free(ptr);
    }
  }

  Float32List preprocessWindow(
    List<double> rawXyzInterleaved, {
    int windowLen = HarConstants.inDim,
    double scaleToG = 1.0 / HarConstants.gravityMs2,
  }) {
    if (rawXyzInterleaved.length != windowLen * 3) {
      throw ArgumentError(
        'Expected ${windowLen * 3} floats, got ${rawXyzInterleaved.length}',
      );
    }
    final inPtr = calloc<Float>(windowLen * 3);
    final outPtr = calloc<Float>(windowLen);
    try {
      final inView = inPtr.asTypedList(windowLen * 3);
      for (var i = 0; i < windowLen * 3; i++) {
        inView[i] = rawXyzInterleaved[i];
      }
      _preprocessWindow(inPtr, outPtr, windowLen, scaleToG);
      return Float32List.fromList(outPtr.asTypedList(windowLen));
    } finally {
      calloc.free(inPtr);
      calloc.free(outPtr);
    }
  }

  List<double> predict(
    Float32List weights,
    Float32List features, {
    int inDim = HarConstants.inDim,
    int hidDim = HarConstants.hidDim,
    int outDim = HarConstants.outDim,
  }) {
    final wPtr = calloc<Float>(weights.length);
    final fPtr = calloc<Float>(features.length);
    final pPtr = calloc<Float>(outDim);
    try {
      wPtr.asTypedList(weights.length).setAll(0, weights);
      fPtr.asTypedList(features.length).setAll(0, features);
      _predict(wPtr, fPtr, pPtr, inDim, hidDim, outDim);
      return List<double>.from(pPtr.asTypedList(outDim));
    } finally {
      calloc.free(wPtr);
      calloc.free(fPtr);
      calloc.free(pPtr);
    }
  }

  double trainBatch(
    Float32List weights,
    Float32List xBatch,
    Int32List yBatch,
    int batchSize, {
    int inDim = HarConstants.inDim,
    int hidDim = HarConstants.hidDim,
    int outDim = HarConstants.outDim,
    double lr = 0.05,
  }) {
    final wPtr = calloc<Float>(weights.length);
    final xPtr = calloc<Float>(xBatch.length);
    final yPtr = calloc<Int32>(yBatch.length);
    try {
      wPtr.asTypedList(weights.length).setAll(0, weights);
      xPtr.asTypedList(xBatch.length).setAll(0, xBatch);
      yPtr.asTypedList(yBatch.length).setAll(0, yBatch);
      final loss = _trainBatch(
        wPtr,
        xPtr,
        yPtr,
        batchSize,
        inDim,
        hidDim,
        outDim,
        lr,
      );
      weights.setAll(0, wPtr.asTypedList(weights.length));
      return loss;
    } finally {
      calloc.free(wPtr);
      calloc.free(xPtr);
      calloc.free(yPtr);
    }
  }

  double evaluateAccuracy(
    Pointer<Float> wPtr,
    Float32List flatFeatures,
    Int32List labels, {
    int inDim = HarConstants.inDim,
    int hidDim = HarConstants.hidDim,
    int outDim = HarConstants.outDim,
  }) {
    final n = labels.length;
    if (n == 0) return 0.0;
    final fPtr = calloc<Float>(inDim);
    final pPtr = calloc<Float>(outDim);
    try {
      final fView = fPtr.asTypedList(inDim);
      final pView = pPtr.asTypedList(outDim);
      var correct = 0;
      for (var i = 0; i < n; i++) {
        fView.setRange(0, inDim, flatFeatures, i * inDim);
        _predict(wPtr, fPtr, pPtr, inDim, hidDim, outDim);
        var bestIdx = 0;
        var bestVal = pView[0];
        for (var k = 1; k < outDim; k++) {
          if (pView[k] > bestVal) {
            bestVal = pView[k];
            bestIdx = k;
          }
        }
        if (bestIdx == labels[i]) {
          correct++;
        }
      }
      return correct / n;
    } finally {
      calloc.free(fPtr);
      calloc.free(pPtr);
    }
  }

  /// Runs shuffled mini-batch SGD for [epochs] on flat [features] ([N * inDim]) and [labels] ([N]).
  /// Safe to invoke inside `Isolate.run(...)`.
  TrainResult trainEpochs({
    required Float32List initialWeights,
    required Float32List flatFeatures,
    required Int32List labels,
    required int epochs,
    int batchSize = 32,
    double lr = 0.05,
    int seed = 42,
  }) {
    final n = labels.length;
    const inDim = HarConstants.inDim;
    const hidDim = HarConstants.hidDim;
    const outDim = HarConstants.outDim;
    final wLen = initialWeights.length;

    final wPtr = calloc<Float>(wLen);
    final xBatchPtr = calloc<Float>(batchSize * inDim);
    final yBatchPtr = calloc<Int32>(batchSize);

    try {
      wPtr.asTypedList(wLen).setAll(0, initialWeights);

      final preAcc = evaluateAccuracy(wPtr, flatFeatures, labels);
      final rng = Random(seed);
      final indices = List<int>.generate(n, (i) => i);

      final xBatchView = xBatchPtr.asTypedList(batchSize * inDim);
      final yBatchView = yBatchPtr.asTypedList(batchSize);

      final sw = Stopwatch()..start();
      var finalEpochLoss = 0.0;

      for (var epoch = 0; epoch < epochs; epoch++) {
        indices.shuffle(rng);
        var epochLossSum = 0.0;
        var numBatches = 0;

        for (var start = 0; start < n; start += batchSize) {
          final end = (start + batchSize < n) ? start + batchSize : n;
          final bs = end - start;
          for (var b = 0; b < bs; b++) {
            final idx = indices[start + b];
            xBatchView.setRange(
              b * inDim,
              (b + 1) * inDim,
              flatFeatures,
              idx * inDim,
            );
            yBatchView[b] = labels[idx];
          }
          final loss = _trainBatch(
            wPtr,
            xBatchPtr,
            yBatchPtr,
            bs,
            inDim,
            hidDim,
            outDim,
            lr,
          );
          epochLossSum += loss;
          numBatches++;
        }
        finalEpochLoss = epochLossSum / max(1, numBatches);
      }

      sw.stop();
      final postAcc = evaluateAccuracy(wPtr, flatFeatures, labels);
      final updatedWeights = Float32List.fromList(wPtr.asTypedList(wLen));

      return TrainResult(
        preAcc: preAcc,
        postAcc: postAcc,
        finalLoss: finalEpochLoss,
        trainTimeMs: sw.elapsedMicroseconds / 1000.0,
        updatedWeights: updatedWeights,
      );
    } finally {
      calloc.free(wPtr);
      calloc.free(xBatchPtr);
      calloc.free(yBatchPtr);
    }
  }
}
