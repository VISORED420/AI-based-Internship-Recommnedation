"""Shared test support: a logged-in Flask test client on a throwaway database.

Not collected by ``unittest discover`` (the pattern is ``test*.py``), it exists
only so the API test modules can authenticate once instead of duplicating the
registration dance. The auth database is a temporary file under the OS temp
directory, so tests never touch ``data/app.db``.
"""
import atexit
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import app as flask_app  # noqa: E402

TEST_NAME = 'Test Student'
TEST_EMAIL = 'tester@example.com'
TEST_PASSWORD = 'correct-horse-battery'

_TMPDIR = tempfile.mkdtemp(prefix='s2i-tests-')
os.environ.setdefault('S2I_DB_PATH', os.path.join(_TMPDIR, 'users.db'))
atexit.register(shutil.rmtree, _TMPDIR, True)


def logged_in_client():
    """Return a test client whose session belongs to a registered user."""
    client = flask_app.app.test_client()

    response = client.post('/login', data={'email': TEST_EMAIL, 'password': TEST_PASSWORD})
    if response.status_code == 302:
        return client

    response = client.post('/register', data={
        'name': TEST_NAME, 'email': TEST_EMAIL,
        'password': TEST_PASSWORD, 'confirm': TEST_PASSWORD})
    assert response.status_code == 302, response.data
    return client
