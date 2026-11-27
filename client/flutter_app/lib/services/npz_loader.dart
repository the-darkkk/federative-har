import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';

class NpzDataset {
  final Float32List flatX;
  final Int32List labels;
  final int numSamples;

  const NpzDataset({
    required this.flatX,
    required this.labels,
    required this.numSamples,
  });
}

class NpzLoader {
  static Future<NpzDataset> loadFile(File file) async {
    final bytes = await file.readAsBytes();
    return decodeBytes(bytes);
  }

  static NpzDataset decodeBytes(Uint8List bytes) {
    final archive = ZipDecoder().decodeBytes(bytes);

    ArchiveFile? xEntry;
    ArchiveFile? yEntry;
    for (final entry in archive) {
      if (entry.name == 'X.npy') {
        xEntry = entry;
      } else if (entry.name == 'y.npy') {
        yEntry = entry;
      }
    }

    if (xEntry == null || yEntry == null) {
      throw const FormatException('Archive must contain X.npy and y.npy');
    }

    final xParsed = _parseNpy(Uint8List.fromList(xEntry.content as List<int>));
    final yParsed = _parseNpy(Uint8List.fromList(yEntry.content as List<int>));

    if (xParsed.shape.length != 2 || xParsed.shape[1] != 128) {
      throw FormatException(
        'Expected X shape [N, 128], got ${xParsed.shape}',
      );
    }
    final numSamples = xParsed.shape[0];
    if (yParsed.shape.length != 1 || yParsed.shape[0] != numSamples) {
      throw FormatException(
        'Expected y shape [$numSamples], got ${yParsed.shape}',
      );
    }

    final flatX = _decodeFloat32Payload(xParsed.payload, xParsed.descr, numSamples * 128);
    final labels = _decodeInt32Payload(yParsed.payload, yParsed.descr, numSamples);

    return NpzDataset(
      flatX: flatX,
      labels: labels,
      numSamples: numSamples,
    );
  }

  static ({String descr, List<int> shape, Uint8List payload}) _parseNpy(
    Uint8List data,
  ) {
    // Check \x93NUMPY magic
    if (data.length < 10 ||
        data[0] != 0x93 ||
        data[1] != 0x4E ||
        data[2] != 0x55 ||
        data[3] != 0x4D ||
        data[4] != 0x50 ||
        data[5] != 0x59) {
      throw const FormatException('Invalid NPY magic header');
    }

    final major = data[6];
    final byteData = ByteData.sublistView(data);
    int headerLen;
    int headerStart;

    if (major == 1) {
      headerLen = byteData.getUint16(8, Endian.little);
      headerStart = 10;
    } else if (major == 2 || major == 3) {
      if (data.length < 12) {
        throw const FormatException('Truncated NPY v2/v3 header');
      }
      headerLen = byteData.getUint32(8, Endian.little);
      headerStart = 12;
    } else {
      throw FormatException('Unsupported NPY version: $major');
    }

    final payloadStart = headerStart + headerLen;
    if (data.length < payloadStart) {
      throw const FormatException('Truncated NPY payload');
    }

    final headerStr = ascii.decode(
      data.sublist(headerStart, payloadStart),
      allowInvalid: true,
    );

    final descrMatch = RegExp(r"'descr'\s*:\s*'([^']+)'").firstMatch(headerStr);
    if (descrMatch == null) {
      throw const FormatException('Missing descr in NPY header');
    }
    final descr = descrMatch.group(1)!;

    final fortranMatch =
        RegExp(r"'fortran_order'\s*:\s*(True|False)").firstMatch(headerStr);
    if (fortranMatch != null && fortranMatch.group(1) == 'True') {
      throw const FormatException('Fortran-ordered NPY arrays are not supported');
    }

    final shapeMatch =
        RegExp(r"'shape'\s*:\s*\(([^)]*)\)").firstMatch(headerStr);
    if (shapeMatch == null) {
      throw const FormatException('Missing shape in NPY header');
    }
    final shapeParts = shapeMatch
        .group(1)!
        .split(',')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .map(int.parse)
        .toList();

    final payload = Uint8List.sublistView(data, payloadStart);
    return (descr: descr, shape: shapeParts, payload: payload);
  }

  static Float32List _decodeFloat32Payload(
    Uint8List payload,
    String descr,
    int count,
  ) {
    if (descr != '<f4' && descr != 'float32' && descr != '|f4') {
      throw FormatException('Expected float32 (<f4) for X, got $descr');
    }
    final expectedBytes = count * 4;
    if (payload.lengthInBytes < expectedBytes) {
      throw FormatException(
        'X payload truncated: ${payload.lengthInBytes} < $expectedBytes',
      );
    }
    final aligned = Uint8List.fromList(payload.sublist(0, expectedBytes));
    return aligned.buffer.asFloat32List();
  }

  static Int32List _decodeInt32Payload(
    Uint8List payload,
    String descr,
    int count,
  ) {
    if (descr == '<i4' || descr == 'int32' || descr == '|i4') {
      final expectedBytes = count * 4;
      if (payload.lengthInBytes < expectedBytes) {
        throw FormatException(
          'y payload truncated: ${payload.lengthInBytes} < $expectedBytes',
        );
      }
      final aligned = Uint8List.fromList(payload.sublist(0, expectedBytes));
      return aligned.buffer.asInt32List();
    } else if (descr == '<i8' || descr == 'int64' || descr == '|i8') {
      final expectedBytes = count * 8;
      if (payload.lengthInBytes < expectedBytes) {
        throw FormatException(
          'y payload truncated: ${payload.lengthInBytes} < $expectedBytes',
        );
      }
      final bd = ByteData.sublistView(payload);
      final out = Int32List(count);
      for (var i = 0; i < count; i++) {
        out[i] = bd.getInt64(i * 8, Endian.little);
      }
      return out;
    } else {
      throw FormatException('Expected int32 (<i4) or int64 (<i8) for y, got $descr');
    }
  }
}
