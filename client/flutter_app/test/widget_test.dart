import 'package:flutter_app/services/comm_logger.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('CommLogger records, copies, and clears timestamped entries', () {
    final logger = CommLogger();
    expect(logger.entries, isEmpty);

    logger.log('Connected to server');
    logger.log('Downloaded model v1');

    expect(logger.entries.length, 2);
    expect(logger.entries[0], contains('Connected to server'));
    expect(logger.copyAll(), contains('Downloaded model v1'));

    logger.clear();
    expect(logger.entries, isEmpty);
  });
}
