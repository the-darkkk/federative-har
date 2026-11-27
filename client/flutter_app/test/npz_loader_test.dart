import 'dart:io';

import 'package:flutter_app/services/npz_loader.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('NpzLoader parses real processed client_0.npz if present', () async {
    final file = File('../../data/processed/non_iid/client_0.npz');
    if (!file.existsSync()) {
      markTestSkipped('Processed dataset not found at ${file.path}');
      return;
    }

    final dataset = await NpzLoader.loadFile(file);
    expect(dataset.numSamples, greaterThan(0));
    expect(dataset.flatX.length, dataset.numSamples * 128);
    expect(dataset.labels.length, dataset.numSamples);
  });
}
