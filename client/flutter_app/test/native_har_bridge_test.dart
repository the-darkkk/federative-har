import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_app/core/native_har_bridge.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('NativeHarBridge binds C99 symbols and runs preprocess, predict, and train', () {
    final dllFile = File('../../core/build/federative_har.dll');
    if (!dllFile.existsSync()) {
      markTestSkipped('Desktop shared library not built at ${dllFile.path}');
      return;
    }

    final bridge = NativeHarBridge(libraryPath: dllFile.path);
    expect(bridge.weightsLen(), 8646);

    final weights = bridge.initWeights(seed: 42);
    expect(weights.length, 8646);

    // 128 samples of [0, 0, 9.80665] -> 1.0 g
    final rawXyz = List<double>.generate(128 * 3, (i) => (i % 3 == 2) ? 9.80665 : 0.0);
    final features = bridge.preprocessWindow(rawXyz);
    expect(features.length, 128);
    expect(features[0], closeTo(1.0, 1e-5));

    final probs = bridge.predict(weights, features);
    expect(probs.length, 6);
    final sum = probs.reduce((a, b) => a + b);
    expect(sum, closeTo(1.0, 1e-5));

    final flatFeatures = Float32List(128 * 2);
    flatFeatures.setRange(0, 128, features);
    flatFeatures.setRange(128, 256, features);
    final labels = Int32List.fromList([0, 1]);

    final result = bridge.trainEpochs(
      initialWeights: weights,
      flatFeatures: flatFeatures,
      labels: labels,
      epochs: 3,
      batchSize: 2,
      lr: 0.05,
    );
    expect(result.finalLoss.isFinite, isTrue);
    expect(result.updatedWeights.length, 8646);
  });
}
