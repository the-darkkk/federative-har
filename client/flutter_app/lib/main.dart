import 'package:material_ui/material_ui.dart';

import 'package:flutter_app/core/native_har_bridge.dart';
import 'package:flutter_app/screens/logs_server_screen.dart';
import 'package:flutter_app/screens/monitor_screen.dart';
import 'package:flutter_app/screens/train_screen.dart';
import 'package:flutter_app/services/comm_logger.dart';
import 'package:flutter_app/services/fl_api_client.dart';
import 'package:flutter_app/services/model_manager.dart';
import 'package:flutter_app/services/sensor_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const FederativeHarApp());
}

class FederativeHarApp extends StatelessWidget {
  const FederativeHarApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Federative HAR',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
      ),
      home: const HomeShell(),
    );
  }
}

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _selectedIndex = 0;

  late final CommLogger _logger;
  late final NativeHarBridge _bridge;
  late final ModelManager _modelManager;
  late final FlApiClient _apiClient;
  late final SensorService _sensorService;

  bool _initialized = false;
  String? _initError;

  @override
  void initState() {
    super.initState();
    _logger = CommLogger();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    try {
      _bridge = NativeHarBridge();
      _modelManager = ModelManager(bridge: _bridge, logger: _logger);
      _apiClient = FlApiClient(modelManager: _modelManager, logger: _logger);
      _sensorService = SensorService(
        bridge: _bridge,
        modelManager: _modelManager,
        logger: _logger,
      );

      await _modelManager.init();
      _sensorService.start();

      if (mounted) {
        setState(() => _initialized = true);
      }
    } catch (e) {
      _logger.log('Initialization error: $e');
      if (mounted) {
        setState(() => _initError = e.toString());
      }
    }
  }

  @override
  void dispose() {
    if (_initialized) {
      _sensorService.dispose();
      _modelManager.dispose();
    }
    _logger.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_initError != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Federative HAR')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Text('Native initialization failed:\n$_initError'),
          ),
        ),
      );
    }

    if (!_initialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final screens = [
      MonitorScreen(
        modelManager: _modelManager,
        sensorService: _sensorService,
        apiClient: _apiClient,
      ),
      TrainScreen(
        bridge: _bridge,
        modelManager: _modelManager,
        sensorService: _sensorService,
        apiClient: _apiClient,
        logger: _logger,
      ),
      LogsServerScreen(
        modelManager: _modelManager,
        apiClient: _apiClient,
        logger: _logger,
      ),
    ];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Federative HAR'),
      ),
      body: IndexedStack(
        index: _selectedIndex,
        children: screens,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (idx) => setState(() => _selectedIndex = idx),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.sensors_outlined),
            selectedIcon: Icon(Icons.sensors),
            label: 'Monitor',
          ),
          NavigationDestination(
            icon: Icon(Icons.model_training_outlined),
            selectedIcon: Icon(Icons.model_training),
            label: 'Train MVP',
          ),
          NavigationDestination(
            icon: Icon(Icons.terminal_outlined),
            selectedIcon: Icon(Icons.terminal),
            label: 'Logs & Server',
          ),
        ],
      ),
    );
  }
}
