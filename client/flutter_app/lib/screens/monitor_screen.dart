import 'package:material_ui/material_ui.dart';

import 'package:flutter_app/core/native_har_bridge.dart';
import 'package:flutter_app/services/fl_api_client.dart';
import 'package:flutter_app/services/model_manager.dart';
import 'package:flutter_app/services/sensor_service.dart';

class MonitorScreen extends StatefulWidget {
  final ModelManager modelManager;
  final SensorService sensorService;
  final FlApiClient apiClient;

  const MonitorScreen({
    super.key,
    required this.modelManager,
    required this.sensorService,
    required this.apiClient,
  });

  @override
  State<MonitorScreen> createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen> {
  bool _isSyncing = false;

  Future<void> _handleQuickSync() async {
    if (_isSyncing) return;
    setState(() => _isSyncing = true);
    final ok = await widget.apiClient.checkHealthAndSync();
    if (!mounted) return;
    setState(() => _isSyncing = false);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          ok
              ? 'Synced: ${widget.modelManager.displayLabel}'
              : 'Sync failed — check Logs & Server tab',
        ),
        duration: const Duration(seconds: 2),
      ),
    );
  }

  static IconData _iconForClass(int idx) {
    switch (idx) {
      case 0:
        return Icons.south_east; // Downstairs
      case 1:
        return Icons.north_east; // Upstairs
      case 2:
        return Icons.directions_walk; // Walking
      case 3:
        return Icons.directions_run; // Jogging
      case 4:
        return Icons.airline_seat_recline_normal; // Sitting
      case 5:
        return Icons.accessibility_new; // Standing
      default:
        return Icons.sensors;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ListenableBuilder(
      listenable: Listenable.merge([widget.modelManager, widget.sensorService]),
      builder: (context, _) {
        final sensor = widget.sensorService;
        final model = widget.modelManager;
        final predIdx = sensor.predictedIndex;
        final confPct = (sensor.confidence * 100).toStringAsFixed(1);

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Row(
              children: [
                ActionChip(
                  avatar: Icon(
                    model.source == 'server'
                        ? Icons.cloud_done
                        : Icons.memory,
                    size: 18,
                  ),
                  label: Text(model.displayLabel),
                  onPressed: _handleQuickSync,
                ),
                const Spacer(),
                IconButton.filledTonal(
                  onPressed: _isSyncing ? null : _handleQuickSync,
                  tooltip: 'Sync with server',
                  icon: _isSyncing
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.sync),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Card(
              elevation: 0,
              color: theme.colorScheme.primaryContainer.withAlpha(90),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        CircleAvatar(
                          radius: 26,
                          backgroundColor: theme.colorScheme.primary,
                          foregroundColor: theme.colorScheme.onPrimary,
                          child: Icon(_iconForClass(predIdx), size: 28),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Predicted Activity',
                                style: theme.textTheme.labelMedium?.copyWith(
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                              ),
                              Text(
                                sensor.predictedLabel,
                                style: theme.textTheme.headlineSmall?.copyWith(
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                        ),
                        Text(
                          predIdx >= 0 ? '$confPct%' : '--',
                          style: theme.textTheme.titleLarge?.copyWith(
                            fontWeight: FontWeight.w700,
                            color: theme.colorScheme.primary,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: LinearProgressIndicator(
                        value: predIdx >= 0 ? sensor.confidence : 0.0,
                        minHeight: 8,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Class Probabilities',
                      style: theme.textTheme.labelLarge,
                    ),
                    const SizedBox(height: 8),
                    ...List.generate(HarConstants.outDim, (i) {
                      final p = sensor.probabilities[i];
                      final isTop = i == predIdx;
                      return Padding(
                        padding: const EdgeInsets.symmetric(vertical: 3),
                        child: Row(
                          children: [
                            SizedBox(
                              width: 88,
                              child: Text(
                                HarConstants.classNames[i],
                                style: theme.textTheme.bodySmall?.copyWith(
                                  fontWeight: isTop
                                      ? FontWeight.bold
                                      : FontWeight.normal,
                                ),
                              ),
                            ),
                            Expanded(
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(4),
                                child: LinearProgressIndicator(
                                  value: p.clamp(0.0, 1.0),
                                  minHeight: 6,
                                  color: isTop
                                      ? theme.colorScheme.primary
                                      : theme.colorScheme.secondary,
                                ),
                              ),
                            ),
                            const SizedBox(width: 8),
                            SizedBox(
                              width: 44,
                              child: Text(
                                '${(p * 100).toStringAsFixed(1)}%',
                                textAlign: TextAlign.end,
                                style: theme.textTheme.bodySmall?.copyWith(
                                  fontFeatures: const [
                                    FontFeature.tabularFigures(),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    }),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            Card(
              elevation: 0,
              color: theme.colorScheme.surfaceContainerHighest.withAlpha(120),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.speed, size: 20),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Live Accelerometer Telemetry',
                            style: theme.textTheme.titleMedium,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.secondaryContainer,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            'Buffer ${sensor.bufferFill}/${SensorService.windowSize}',
                            style: theme.textTheme.labelSmall,
                          ),
                        ),
                      ],
                    ),
                    if (sensor.sensorError != null) ...[
                      const SizedBox(height: 8),
                      Text(
                        'Sensor error: ${sensor.sensorError}',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.error,
                        ),
                      ),
                    ],
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        _TelemetryTile(
                          label: 'X (m/s²)',
                          value: sensor.x.toStringAsFixed(2),
                        ),
                        _TelemetryTile(
                          label: 'Y (m/s²)',
                          value: sensor.y.toStringAsFixed(2),
                        ),
                        _TelemetryTile(
                          label: 'Z (m/s²)',
                          value: sensor.z.toStringAsFixed(2),
                        ),
                        _TelemetryTile(
                          label: '|a| (g)',
                          value: sensor.magnitudeG.toStringAsFixed(3),
                          highlight: true,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _TelemetryTile extends StatelessWidget {
  final String label;
  final String value;
  final bool highlight;

  const _TelemetryTile({
    required this.label,
    required this.value,
    this.highlight = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(
            label,
            style: theme.textTheme.labelSmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: highlight ? theme.colorScheme.primary : null,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}
