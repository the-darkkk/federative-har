import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';

import 'package:flutter_app/services/comm_logger.dart';
import 'package:flutter_app/services/fl_api_client.dart';
import 'package:flutter_app/services/model_manager.dart';

class LogsServerScreen extends StatefulWidget {
  final ModelManager modelManager;
  final FlApiClient apiClient;
  final CommLogger logger;

  const LogsServerScreen({
    super.key,
    required this.modelManager,
    required this.apiClient,
    required this.logger,
  });

  @override
  State<LogsServerScreen> createState() => _LogsServerScreenState();
}

class _LogsServerScreenState extends State<LogsServerScreen> {
  late final TextEditingController _urlController;
  bool _isConnecting = false;

  @override
  void initState() {
    super.initState();
    _urlController = TextEditingController(text: widget.modelManager.serverUrl);
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _connectAndSync() async {
    if (_isConnecting) return;
    setState(() => _isConnecting = true);
    await widget.apiClient.checkHealthAndSync(_urlController.text);
    if (mounted) {
      setState(() => _isConnecting = false);
    }
  }

  Future<void> _copyLogs() async {
    final text = widget.logger.copyAll();
    await Clipboard.setData(ClipboardData(text: text));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Copied logs to clipboard'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Card(
            elevation: 0,
            color: theme.colorScheme.surfaceContainerHighest.withAlpha(110),
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  TextField(
                    controller: _urlController,
                    keyboardType: TextInputType.url,
                    decoration: const InputDecoration(
                      labelText: 'Server URL',
                      hintText: 'http://192.168.1.100:8000',
                      prefixIcon: Icon(Icons.dns),
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                  const SizedBox(height: 10),
                  ListenableBuilder(
                    listenable: widget.modelManager,
                    builder: (context, _) {
                      return Row(
                        children: [
                          Icon(
                            Icons.fingerprint,
                            size: 16,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                          const SizedBox(width: 6),
                          Expanded(
                            child: Text(
                              'Client ID: ${widget.modelManager.clientId}',
                              style: theme.textTheme.bodySmall?.copyWith(
                                fontFamily: 'monospace',
                              ),
                            ),
                          ),
                          Text(
                            widget.modelManager.displayLabel,
                            style: theme.textTheme.labelMedium?.copyWith(
                              color: theme.colorScheme.primary,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      );
                    },
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed: _isConnecting ? null : _connectAndSync,
                      icon: _isConnecting
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.cloud_sync),
                      label: Text(
                        _isConnecting ? 'Syncing...' : 'Connect & Sync',
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Text(
                'Communication Log Console',
                style: theme.textTheme.titleSmall,
              ),
              const Spacer(),
              TextButton.icon(
                onPressed: _copyLogs,
                icon: const Icon(Icons.copy, size: 16),
                label: const Text('Copy'),
              ),
              TextButton.icon(
                onPressed: widget.logger.clear,
                icon: const Icon(Icons.delete_outline, size: 16),
                label: const Text('Clear'),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Expanded(
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFF121417),
                borderRadius: BorderRadius.circular(12),
              ),
              child: ListenableBuilder(
                listenable: widget.logger,
                builder: (context, _) {
                  final entries = widget.logger.entries;
                  if (entries.isEmpty) {
                    return const Text(
                      'No log entries yet.',
                      style: TextStyle(
                        color: Colors.white54,
                        fontFamily: 'monospace',
                        fontSize: 12,
                      ),
                    );
                  }
                  return ListView.builder(
                    reverse: true,
                    itemCount: entries.length,
                    itemBuilder: (context, index) {
                      final entry = entries[entries.length - 1 - index];
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 2),
                        child: SelectableText(
                          entry,
                          style: const TextStyle(
                            color: Color(0xFFD6E3DF),
                            fontFamily: 'monospace',
                            fontSize: 12,
                            height: 1.35,
                          ),
                        ),
                      );
                    },
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}
