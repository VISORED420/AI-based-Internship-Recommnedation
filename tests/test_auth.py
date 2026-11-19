"""Tests for the login backend: registration, login, logout and route gating.

Run from the repository root:

    python -m unittest discover -s tests -v

Every test runs against a temporary SQLite file so the developer's
``data/app.db`` is never modified.
"""
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import app as flask_app  # noqa: E402
import auth  # noqa: E402

VALID = {'name': 'Asha Rao', 'email': 'asha@example.com',
         'password': 'strong-passphrase', 'confirm': 'strong-passphrase'}


class AuthTestCase(unittest.TestCase):
    """Base: a fresh client and an empty database per test."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='s2i-auth-')
        self.db_path = os.path.join(self.tmpdir, 'users.db')
        self._previous_db = os.environ.get('S2I_DB_PATH')
        os.environ['S2I_DB_PATH'] = self.db_path
        self.client = flask_app.app.test_client()

    def tearDown(self):
        if self._previous_db is None:
            os.environ.pop('S2I_DB_PATH', None)
        else:
            os.environ['S2I_DB_PATH'] = self._previous_db
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def register(self, **overrides):
        payload = dict(VALID, **overrides)
        return self.client.post('/register', data=payload)

    def login(self, **overrides):
        payload = {'email': VALID['email'], 'password': VALID['password'], **overrides}
        return self.client.post('/login', data=payload)

    def users_rows(self):
        with flask_app.app.app_context():
            return auth.get_db().execute('SELECT * FROM users').fetchall()


class RouteGatingTests(AuthTestCase):

    def test_recommend_requires_login_with_json_401(self):
        response = self.client.post('/recommend', json={'qualification': 'B.Tech', 'skills': 'Python'})
        self.assertEqual(response.status_code, 401)
        self.assertTrue(response.content_type.startswith('application/json'))
        self.assertIn('Authentication required', response.get_json()['error'])

    def test_home_redirects_anonymous_user_to_login(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers['Location'])

    def test_health_stays_public_for_app_runner(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['status'], 'ok')

    def test_api_me_reports_unauthenticated(self):
        response = self.client.get('/api/me')
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json(), {'authenticated': False, 'user': None})

    def test_protected_route_works_after_login(self):
        self.register()
        response = self.client.post('/recommend', json={'qualification': 'B.Tech', 'skills': 'Python, SQL'})
        self.assertEqual(response.status_code, 200, response.data)
        self.assertTrue(response.get_json())


class RegistrationTests(AuthTestCase):

    def test_register_creates_user_and_logs_in(self):
        response = self.register()
        self.assertEqual(response.status_code, 302)
        rows = self.users_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['email'], 'asha@example.com')
        self.assertEqual(self.client.get('/api/me').get_json()['user']['name'], 'Asha Rao')

    def test_password_is_hashed_not_stored_plaintext(self):
        self.register()
        row = self.users_rows()[0]
        self.assertNotIn('strong-passphrase', row['password_hash'])
        self.assertTrue(row['password_hash'].startswith(('pbkdf2:', 'scrypt:')))

    def test_email_is_normalised_to_lowercase(self):
        self.register(email='Asha@Example.COM')
        self.assertEqual(self.users_rows()[0]['email'], 'asha@example.com')

    def test_duplicate_email_is_rejected(self):
        self.register()
        response = flask_app.app.test_client().post('/register', data=dict(VALID))
        self.assertEqual(response.status_code, 409)
        self.assertIn('already exists', response.get_data(as_text=True))

    def test_short_password_is_rejected(self):
        response = self.register(password='short', confirm='short')
        self.assertEqual(response.status_code, 400)
        self.assertIn('at least 8 characters', response.get_data(as_text=True))
        self.assertEqual(len(self.users_rows()), 0)

    def test_mismatched_confirmation_is_rejected(self):
        response = self.register(confirm='different-passphrase')
        self.assertEqual(response.status_code, 400)
        self.assertIn('do not match', response.get_data(as_text=True))

    def test_invalid_email_is_rejected(self):
        for bad in ('not-an-email', 'a@b', '@example.com', 'a b@example.com', ''):
            response = self.register(email=bad)
            self.assertEqual(response.status_code, 400, bad)
        self.assertEqual(len(self.users_rows()), 0)

    def test_blank_name_is_rejected(self):
        response = self.register(name='   ')
        self.assertEqual(response.status_code, 400)
        self.assertIn('enter your name', response.get_data(as_text=True))

    def test_register_page_is_public(self):
        response = self.client.get('/register')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Create Account', response.get_data(as_text=True))


class LoginTests(AuthTestCase):

    def test_login_succeeds_with_correct_password(self):
        self.register()
        self.client.post('/logout')
        response = self.login()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/')

    def test_wrong_password_is_rejected(self):
        self.register()
        self.client.post('/logout')
        response = self.login(password='wrong-passphrase')
        self.assertEqual(response.status_code, 401)
        self.assertIn('Incorrect email or password', response.get_data(as_text=True))
        self.assertEqual(self.client.get('/api/me').status_code, 401)

    def test_unknown_email_gives_same_message_as_wrong_password(self):
        response = self.login(email='nobody@example.com')
        self.assertEqual(response.status_code, 401)
        self.assertIn('Incorrect email or password', response.get_data(as_text=True))

    def test_login_is_case_insensitive_on_email(self):
        self.register()
        self.client.post('/logout')
        self.assertEqual(self.login(email='ASHA@EXAMPLE.COM').status_code, 302)

    def test_next_parameter_redirects_to_relative_target(self):
        self.register()
        self.client.post('/logout')
        response = self.client.post('/login?next=/register',
                                    data={'email': VALID['email'], 'password': VALID['password']})
        self.assertEqual(response.headers['Location'], '/register')

    def test_next_parameter_rejects_external_url(self):
        self.register()
        self.client.post('/logout')
        response = self.client.post('/login?next=https://evil.example.com/phish',
                                    data={'email': VALID['email'], 'password': VALID['password']})
        self.assertEqual(response.headers['Location'], '/')
        response = self.client.post('/login?next=//evil.example.com',
                                    data={'email': VALID['email'], 'password': VALID['password']})
        self.assertEqual(response.headers['Location'], '/')


class LogoutTests(AuthTestCase):

    def test_logout_clears_session(self):
        self.register()
        self.assertIsNotNone(self.client.get('/api/me').get_json()['user'])
        response = self.client.post('/logout')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.get('/api/me').status_code, 401)
        self.assertEqual(self.client.get('/').status_code, 302)

    def test_logout_rejects_get(self):
        self.register()
        self.assertEqual(self.client.get('/logout').status_code, 405)

    def test_login_page_renders_for_anonymous_visitor(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Log In', response.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()
