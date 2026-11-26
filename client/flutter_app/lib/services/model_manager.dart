import 'dart:io';
import 'dart:math';

import 'package:crypto/crypto.dart' as crypto;
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:flutter_app/core/native_har_bridge.dart';
import 'package:flutter_app/services/comm_logger.dart';

class ModelManager extends ChangeNotifier {
  static const String _keyVersion = 'har_model_version';
  static const String _keySource = 'har_model_source';
  static const String _keySha256 = 'har_model_sha256';
  static const String _keyServerUrl = 'har_server_url';
  static const String _keyClientId = 'har_client_id';
  static const String defaultServerUrl = 'http://192.168.1.100:8000';

  final NativeHarBridge bridge;
  final CommLogger logger;

  Float32List? _weights;
  int _version = 0;
  String _source = 'local_init';
  String _sha256 = '';
  String _serverUrl = defaultServerUrl;
  String _clientId = '';
  bool _isReady = false;
  Directory? _docsDir;

  ModelManager({
    required this.bridge,
    required this.logger,
  });

  Float32List? get weights => _weights;
  int get version => _version;
  String get source => _source;
  String get sha256 => _sha256;
  String get serverUrl => _serverUrl;
  String get clientId => _clientId;
  bool get isReady => _isReady;

  String get displayLabel =>
      _source == 'server' ? 'Model v$_version' : 'Local Init v$_version';

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _docsDir = await getApplicationDocumentsDirectory();

    var storedClientId = prefs.getString(_keyClientId);
    if (storedClientId == null || storedClientId.isEmpty) {
      storedClientId = _generateClientId();
      await prefs.setString(_keyClientId, storedClientId);
    }
    _clientId = storedClientId;

    _serverUrl = prefs.getString(_keyServerUrl) ?? defaultServerUrl;
    _version = prefs.getInt(_keyVersion) ?? 0;
    _source = prefs.getString(_keySource) ?? 'local_init';
    _sha256 = prefs.getString(_keySha256) ?? '';

    final modelFile = File('${_docsDir!.path}/local_model.bin');
    if (await modelFile.exists()) {
      final bytes = await modelFile.readAsBytes();
      if (bytes.lengthInBytes == HarConstants.weightsByteSize) {
        _weights = Uint8List.fromList(bytes).buffer.asFloat32List();
        if (_sha256.isEmpty) {
          _sha256 = crypto.sha256.convert(bytes).toString();
        }
        _isReady = true;
        logger.log(
          'Loaded persisted $displayLabel (${bytes.lengthInBytes} bytes)',
        );
        notifyListeners();
        return;
      }
    }

    await resetToLocalInit();
  }

  Future<void> resetToLocalInit() async {
    final initWeights = bridge.initWeights(seed: 42);
    final rawBytes = initWeights.buffer.asUint8List(
      initWeights.offsetInBytes,
      initWeights.lengthInBytes,
    );
    final digest = crypto.sha256.convert(rawBytes).toString();

    await _writeModelFile(rawBytes);

    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_keyVersion, 0);
    await prefs.setString(_keySource, 'local_init');
    await prefs.setString(_keySha256, digest);

    _weights = initWeights;
    _version = 0;
    _source = 'local_init';
    _sha256 = digest;
    _isReady = true;

    logger.log(
      'Initialized baseline model Local Init v0 (seed=42, ${rawBytes.lengthInBytes} bytes)',
    );
    notifyListeners();
  }

  Future<void> saveServerModel({
    required Uint8List rawBytes,
    required int version,
    required String sha256Hex,
  }) async {
    if (rawBytes.lengthInBytes != HarConstants.weightsByteSize) {
      throw ArgumentError(
        'Invalid weights length: ${rawBytes.lengthInBytes}, expected ${HarConstants.weightsByteSize}',
      );
    }

    final copied = Uint8List.fromList(rawBytes);
    final floatWeights = copied.buffer.asFloat32List();

    await _writeModelFile(copied);

    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_keyVersion, version);
    await prefs.setString(_keySource, 'server');
    await prefs.setString(_keySha256, sha256Hex);

    _weights = floatWeights;
    _version = version;
    _source = 'server';
    _sha256 = sha256Hex;
    _isReady = true;
    notifyListeners();
  }

  Future<void> setServerUrl(String url) async {
    final normalized = url.trim().replaceAll(RegExp(r'/+$'), '');
    if (normalized.isEmpty) return;
    _serverUrl = normalized;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyServerUrl, normalized);
    notifyListeners();
  }

  Future<void> _writeModelFile(Uint8List bytes) async {
    final dir = _docsDir ?? await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/local_model.bin');
    await file.writeAsBytes(bytes, flush: true);
  }

  static String _generateClientId() {
    final rng = Random.secure();
    final buf = StringBuffer('android_');
    for (var i = 0; i < 8; i++) {
      buf.write(rng.nextInt(16).toRadixString(16));
    }
    return buf.toString();
  }
}
