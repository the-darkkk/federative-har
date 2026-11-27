import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import 'package:flutter_app/services/comm_logger.dart';
import 'package:flutter_app/services/npz_loader.dart';

class DatasetInfo {
  final String name;
  final String path;
  final int sampleCount;

  const DatasetInfo({
    required this.name,
    required this.path,
    required this.sampleCount,
  });

  String get displayLabel => '$name ($sampleCount windows)';
}

class DatasetManager extends ChangeNotifier {
  final CommLogger logger;

  final List<DatasetInfo> _availableDatasets = [];
  String? _selectedDatasetName;
  Directory? _datasetsDir;

  DatasetManager({required this.logger});

  List<DatasetInfo> get availableDatasets =>
      List<DatasetInfo>.unmodifiable(_availableDatasets);

  String? get selectedDatasetName => _selectedDatasetName;

  DatasetInfo? get selectedDataset {
    if (_selectedDatasetName == null) return null;
    for (final d in _availableDatasets) {
      if (d.name == _selectedDatasetName) return d;
    }
    return _availableDatasets.isNotEmpty ? _availableDatasets.first : null;
  }

  Future<void> init() async {
    final docsDir = await getApplicationDocumentsDirectory();
    _datasetsDir = Directory('${docsDir.path}/datasets');
    if (!await _datasetsDir!.exists()) {
      await _datasetsDir!.create(recursive: true);
    }
    await refresh();
  }

  Future<void> refresh() async {
    final dir = _datasetsDir;
    if (dir == null || !await dir.exists()) return;

    final entries = await dir.list().toList();
    final found = <DatasetInfo>[];

    for (final entity in entries) {
      if (entity is File && entity.path.toLowerCase().endsWith('.npz')) {
        try {
          final parsed = await NpzLoader.loadFile(entity);
          final name = entity.uri.pathSegments.last;
          found.add(
            DatasetInfo(
              name: name,
              path: entity.path,
              sampleCount: parsed.numSamples,
            ),
          );
        } catch (e) {
          logger.log('Skipped invalid .npz (${entity.path}): $e');
        }
      }
    }

    found.sort((a, b) => a.name.compareTo(b.name));
    _availableDatasets
      ..clear()
      ..addAll(found);

    if (_availableDatasets.isEmpty) {
      _selectedDatasetName = null;
    } else if (_selectedDatasetName == null ||
        !_availableDatasets.any((d) => d.name == _selectedDatasetName)) {
      _selectedDatasetName = _availableDatasets.first.name;
    }
    notifyListeners();
  }

  void selectDataset(String? name) {
    if (name == null) return;
    _selectedDatasetName = name;
    notifyListeners();
  }

  Future<int> importNpzFiles() async {
    final dir = _datasetsDir;
    if (dir == null) await init();

    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      allowMultiple: true,
    );
    if (result == null || result.files.isEmpty) return 0;

    var importedCount = 0;
    String? lastImportedName;

    for (final picked in result.files) {
      if (!picked.name.toLowerCase().endsWith('.npz')) {
        logger.log('Ignored non-.npz file: ${picked.name}');
        continue;
      }

      try {
        final Uint8List bytes;
        if (picked.bytes != null) {
          bytes = picked.bytes!;
        } else if (picked.path != null) {
          bytes = await File(picked.path!).readAsBytes();
        } else {
          continue;
        }

        // Validate archive before storing permanently
        final parsed = NpzLoader.decodeBytes(bytes);
        final destFile = File('${_datasetsDir!.path}/${picked.name}');
        await destFile.writeAsBytes(bytes, flush: true);

        lastImportedName = picked.name;
        importedCount++;
        logger.log(
          'Imported ${picked.name} (${parsed.numSamples} windows)',
        );
      } catch (e) {
        logger.log('Failed to import ${picked.name}: $e');
      }
    }

    if (importedCount > 0) {
      await refresh();
      if (lastImportedName != null) {
        _selectedDatasetName = lastImportedName;
        notifyListeners();
      }
    }

    return importedCount;
  }

  Future<void> deleteDataset(String name) async {
    final dir = _datasetsDir;
    if (dir == null) return;
    final file = File('${dir.path}/$name');
    if (await file.exists()) {
      await file.delete();
      logger.log('Deleted dataset $name');
    }
    await refresh();
  }

  Future<NpzDataset> loadSelectedDataset() async {
    final info = selectedDataset;
    if (info == null) {
      throw StateError('No .npz dataset selected');
    }
    return NpzLoader.loadFile(File(info.path));
  }
}
