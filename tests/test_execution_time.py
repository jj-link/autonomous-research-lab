from datetime import datetime, timezone
import unittest

from autonomous_research_lab.execution import timestamp


class PodmanTimeTests(unittest.TestCase):
    def test_nanosecond_runtime_timestamp_on_supported_python_versions(self):
        expected = datetime(2026, 7, 10, 12, 34, 56, 123456, timezone.utc).timestamp()
        self.assertEqual(timestamp("2026-07-10T12:34:56.123456789Z"), expected)

    def test_never_started_container_has_unknown_time_not_year_one_epoch(self):
        self.assertIsNone(timestamp("0001-01-01T00:00:00Z"))


if __name__ == "__main__":
    unittest.main()
