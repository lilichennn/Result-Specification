"""Offline tests for isolated RC experiments."""
import unittest
from unittest.mock import patch


class OfflineTestCase(unittest.TestCase):
    """Fail immediately if a source/controller test reaches a real transport."""
    def setUp(self):
        super().setUp()
        for target in ('socket.socket.connect', 'socket.create_connection', 'psycopg.connect'):
            guard = patch(target, side_effect=AssertionError('Real network/database calls are forbidden in offline tests'))
            guard.start()
            self.addCleanup(guard.stop)
