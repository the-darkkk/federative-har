import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;

import 'package:flutter_app/core/native_har_bridge.dart';
import 'package:flutter_app/services/comm_logger.dart';
import 'package:flutter_app/services/model_manager.dart';

class FlApiClient {
  final ModelManager modelManager;
  final CommLogger logger;
  final http.Client _httpClient;

  FlApiClient({
    required this.modelManager,
    required this.logger,
    http.Client? httpClient,
  }) : _httpClient = httpClient ?? http.Client();

  Future<bool> checkHealthAndSync([String? overrideServerUrl]) async {
    if (overrideServerUrl != null && overrideServerUrl.trim().isNotEmpty) {
      await modelManager.setServerUrl(overrideServerUrl);
    }

    final baseUrl = modelManager.serverUrl;
    logger.log('Connecting to $baseUrl');

    try {
      final healthResp = await _httpClient
          .get(Uri.parse('$baseUrl/api/v1/health'))
          .timeout(const Duration(seconds: 8));

      if (healthResp.statusCode != 200) {
        logger.log('Health check failed: HTTP ${healthResp.statusCode}');
        return false;
      }

      final healthJson = jsonDecode(healthResp.body) as Map<String, dynamic>;
      logger.log(
        'Connected to $baseUrl (db=${healthJson['database_backend']}, server_v=${healthJson['current_model_version']})',
      );

      final latestResp = await _httpClient
          .get(Uri.parse('$baseUrl/api/v1/model/latest'))
          .timeout(const Duration(seconds: 8));

      if (latestResp.statusCode != 200) {
        logger.log('Failed to fetch model metadata: HTTP ${latestResp.statusCode}');
        return false;
      }

      final meta = jsonDecode(latestResp.body) as Map<String, dynamic>;
      final serverVersion = meta['version'] as int;
      final serverSha = meta['sha256'] as String;
      final expectedBytes = (meta['weights_count'] as int) * 4;
      final localDesc = modelManager.source == 'server'
          ? 'v${modelManager.version}'
          : 'none (local_init v0)';

      logger.log('Local model: $localDesc | Server latest: v$serverVersion');

      final needsDownload = modelManager.source != 'server' ||
          modelManager.version != serverVersion ||
          modelManager.sha256 != serverSha;

      if (!needsDownload) {
        logger.log('Model v$serverVersion is already up to date (SHA-256 matched)');
        return true;
      }

      final weightsResp = await _httpClient
          .get(Uri.parse('$baseUrl/api/v1/model/latest/weights'))
          .timeout(const Duration(seconds: 15));

      if (weightsResp.statusCode != 200) {
        logger.log('Failed to download weights: HTTP ${weightsResp.statusCode}');
        return false;
      }

      final rawBytes = weightsResp.bodyBytes;
      if (rawBytes.lengthInBytes != expectedBytes ||
          rawBytes.lengthInBytes != HarConstants.weightsByteSize) {
        logger.log(
          'Weight size mismatch: got ${rawBytes.lengthInBytes}, expected $expectedBytes',
        );
        return false;
      }

      final actualSha = sha256.convert(rawBytes).toString();
      if (actualSha != serverSha) {
        logger.log('SHA-256 verification failed for v$serverVersion');
        return false;
      }

      await modelManager.saveServerModel(
        rawBytes: rawBytes,
        version: serverVersion,
        sha256Hex: actualSha,
      );

      logger.log(
        'Downloaded model v$serverVersion (${rawBytes.lengthInBytes} bytes, SHA-256 OK)',
      );
      return true;
    } catch (e) {
      logger.log('Sync error ($baseUrl): $e');
      return false;
    }
  }

  Future<String> submitWeights({
    required Float32List weights,
    required int baseVersion,
    required int sampleCount,
    required double localLoss,
    required double localAccuracy,
    required double trainTimeMs,
    required String inputMode,
  }) async {
    final baseUrl = modelManager.serverUrl;
    final uri = Uri.parse('$baseUrl/api/v1/model/submit');

    try {
      final request = http.MultipartRequest('POST', uri)
        ..fields['client_id'] = modelManager.clientId
        ..fields['base_version'] = baseVersion.toString()
        ..fields['sample_count'] = sampleCount.toString()
        ..fields['local_loss'] = localLoss.toStringAsFixed(6)
        ..fields['local_accuracy'] = localAccuracy.toStringAsFixed(6)
        ..fields['platform'] = 'android'
        ..fields['input_mode'] = inputMode
        ..fields['train_time_ms'] = trainTimeMs.toStringAsFixed(2)
        ..files.add(
          http.MultipartFile.fromBytes(
            'weights',
            weights.buffer.asUint8List(
              weights.offsetInBytes,
              weights.lengthInBytes,
            ),
            filename: 'weights.bin',
          ),
        );

      final streamed = await _httpClient
          .send(request)
          .timeout(const Duration(seconds: 15));
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200) {
        final body = jsonDecode(response.body) as Map<String, dynamic>;
        final status = body['status']?.toString() ?? 'accepted';
        logger.log(
          'Submitted base_v=$baseVersion ($sampleCount samples, loss=${localLoss.toStringAsFixed(4)}) -> 200 OK ($status)',
        );
        return status;
      }

      if (response.statusCode == 409) {
        logger.log(
          'Submitted base_v=$baseVersion -> 409 Conflict (stale model). Syncing latest...',
        );
        await checkHealthAndSync();
        return 'stale';
      }

      logger.log(
        'Submission rejected: HTTP ${response.statusCode} - ${response.body}',
      );
      return 'error';
    } catch (e) {
      logger.log('Submission failed ($baseUrl): $e');
      return 'error';
    }
  }
}
