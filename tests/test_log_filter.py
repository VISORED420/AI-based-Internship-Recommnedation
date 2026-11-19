"""The dev access-log filter must drop port-probe noise and nothing else."""
import logging
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import app as flask_app  # noqa: E402


def _record(message):
    return logging.LogRecord(name='werkzeug', level=logging.INFO, pathname='', lineno=0,
                             msg=message, args=(), exc_info=None)


class ProbeLogFilterTests(unittest.TestCase):

    def setUp(self):
        logger = logging.getLogger('test_probe_logger')
        logger.handlers = []
        self.logger = logger
        self.filter = flask_app.quiet_probe_noise('test_probe_logger')

    def test_drops_the_steam_post_probe(self):
        line = ('127.0.0.1 - - [20/Nov/2026 02:30:00] "POST / HTTP/1.1" 405 -')
        self.assertFalse(self.filter.filter(_record(line)))

    def test_drops_http_1_0_variants(self):
        line = '127.0.0.1 - - [x] "POST / HTTP/1.0" 405 -'
        self.assertFalse(self.filter.filter(_record(line)))

    def test_keeps_real_requests(self):
        kept = [
            '127.0.0.1 - - [x] "GET /jobs HTTP/1.1" 200 -',
            '127.0.0.1 - - [x] "POST /recommend HTTP/1.1" 200 -',
            '127.0.0.1 - - [x] "POST /login HTTP/1.1" 302 -',
            '127.0.0.1 - - [x] "GET /api/jobs?q=python HTTP/1.1" 200 -',
            '127.0.0.1 - - [x] "POST / HTTP/1.1" 500 -',   # a real POST / error still shows
        ]
        for line in kept:
            self.assertTrue(self.filter.filter(_record(line)), line)

    def test_filter_is_attached_to_the_named_logger(self):
        self.assertIn(self.filter, self.logger.filters)


if __name__ == '__main__':
    unittest.main()
