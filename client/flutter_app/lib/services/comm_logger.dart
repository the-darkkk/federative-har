import 'dart:collection';

import 'package:flutter/foundation.dart';

class CommLogger extends ChangeNotifier {
  static const int _maxEntries = 500;
  final List<String> _entries = [];

  UnmodifiableListView<String> get entries => UnmodifiableListView(_entries);

  void log(String message) {
    final now = DateTime.now();
    final hh = now.hour.toString().padLeft(2, '0');
    final mm = now.minute.toString().padLeft(2, '0');
    final ss = now.second.toString().padLeft(2, '0');
    final formatted = '[$hh:$mm:$ss] $message';

    _entries.add(formatted);
    if (_entries.length > _maxEntries) {
      _entries.removeAt(0);
    }
    notifyListeners();
  }

  void clear() {
    _entries.clear();
    notifyListeners();
  }

  String copyAll() {
    return _entries.join('\n');
  }
}
