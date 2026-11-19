"""Session-based authentication for Skill2Intern.

Self-registration with email + password. Passwords are hashed with
``werkzeug.security`` (scrypt by default), users live in a small SQLite file,
and the login state is a signed Flask session cookie, so no server-side
session store or extra dependency is required.

Environment variables
---------------------
S2I_DB_PATH    Path to the SQLite database file (default: ``data/app.db``).
SECRET_KEY     Flask session signing key. When unset, a key is generated once
               and stored in ``data/secret_key`` so restarts and multiple
               gunicorn workers keep the same key.
COOKIE_SECURE  ``1`` marks the session cookie ``Secure``; enable it behind
               HTTPS (AWS App Runner terminates TLS in front of the app).
"""
import os
import re
import secrets
import sqlite3
from datetime import timedelta
from functools import wraps

from flask import (Blueprint, flash, g, jsonify, redirect,
                   render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(REPO_ROOT, 'data', 'app.db')

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 200
MAX_NAME_LENGTH = 60
MAX_EMAIL_LENGTH = 254

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$')

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


# --- database ---------------------------------------------------------------

def _db_path():
    """Resolve the database path on every call so tests can redirect it."""
    return os.environ.get('S2I_DB_PATH') or DEFAULT_DB_PATH


def get_db():
    """Return the request-scoped connection, creating the schema on first use."""
    if 'db' not in g:
        path = _db_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        conn.executescript(SCHEMA)
        g.db = conn
    return g.db


def close_db(exception=None):
    conn = g.pop('db', None)
    if conn is not None:
        conn.close()


# --- session helpers --------------------------------------------------------

def _session_secret():
    """Read SECRET_KEY, else generate one once and persist it in data/."""
    key = os.environ.get('SECRET_KEY')
    if key:
        return key
    key_file = os.path.join(REPO_ROOT, 'data', 'secret_key')
    try:
        with open(key_file, 'r', encoding='utf-8') as handle:
            stored = handle.read().strip()
        if stored:
            return stored
    except FileNotFoundError:
        pass
    generated = secrets.token_hex(32)
    os.makedirs(os.path.dirname(key_file), exist_ok=True)
    with open(key_file, 'w', encoding='utf-8') as handle:
        handle.write(generated)
    return generated


def _safe_next(target):
    """Only allow same-site relative redirects after login."""
    if target and target.startswith('/') and not target.startswith('//'):
        return target
    return url_for('home')


def load_logged_in_user():
    """Populate ``g.user`` from the session on every request."""
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            'SELECT id, name, email FROM users WHERE id = ?', (user_id,)
        ).fetchone()
        if g.user is None:
            session.clear()


def login_required(view):
    """Gate a view behind login.

    API callers get a 401 JSON body (the documented error contract), page
    requests are redirected to the login form with a ``next`` parameter.
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            if request.path.startswith('/api') or request.is_json:
                return jsonify({'error': 'Authentication required. Please log in.'}), 401
            return redirect(url_for('auth.login', next=request.path))
        return view(*args, **kwargs)
    return wrapped


# --- validation -------------------------------------------------------------

def _clean(value):
    return str(value or '').strip()


def _validate_registration(name, email, password, confirm):
    """Return (name, email, password) or raise ValueError with a user-facing message."""
    name = _clean(name)
    email = _clean(email).lower()
    password = str(password or '')
    confirm = str(confirm or '')

    if not name:
        raise ValueError('Please enter your name.')
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f'Name must be {MAX_NAME_LENGTH} characters or fewer.')
    if not email:
        raise ValueError('Please enter your email address.')
    if len(email) > MAX_EMAIL_LENGTH or not _EMAIL_RE.match(email):
        raise ValueError('Please enter a valid email address.')
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f'Password must be at least {MIN_PASSWORD_LENGTH} characters.')
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f'Password must be {MAX_PASSWORD_LENGTH} characters or fewer.')
    if password != confirm:
        raise ValueError('Passwords do not match.')
    return name, email, password


# --- routes -----------------------------------------------------------------

bp = Blueprint('auth', __name__)


@bp.route('/register', methods=('GET', 'POST'))
def register():
    if g.user is not None:
        return redirect(url_for('home'))

    form = {'name': '', 'email': ''}
    if request.method == 'POST':
        form = {'name': _clean(request.form.get('name')),
                'email': _clean(request.form.get('email')).lower()}
        try:
            name, email, password = _validate_registration(
                form['name'], form['email'],
                request.form.get('password'), request.form.get('confirm'))
        except ValueError as exc:
            return render_template('register.html', error=str(exc), form=form), 400

        db = get_db()
        try:
            cursor = db.execute(
                'INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
                (name, email, generate_password_hash(password)))
            db.commit()
        except sqlite3.IntegrityError:
            # UNIQUE(email) — race-safe duplicate detection, no pre-check needed.
            return render_template(
                'register.html',
                error='An account with that email already exists. Try logging in.',
                form=form), 409

        session.clear()
        session['user_id'] = cursor.lastrowid
        session.permanent = True
        flash(f'Welcome, {name}. Your account is ready.')
        return redirect(url_for('home'))

    return render_template('register.html', error=None, form=form)


@bp.route('/login', methods=('GET', 'POST'))
def login():
    if g.user is not None:
        return redirect(url_for('home'))

    form = {'email': ''}
    if request.method == 'POST':
        form['email'] = _clean(request.form.get('email')).lower()
        password = str(request.form.get('password') or '')

        user = get_db().execute(
            'SELECT * FROM users WHERE email = ?', (form['email'],)
        ).fetchone()

        # One generic message for both cases: never reveal which emails exist.
        if user is None or not check_password_hash(user['password_hash'], password):
            return render_template(
                'login.html',
                error='Incorrect email or password.',
                form=form), 401

        session.clear()
        session['user_id'] = user['id']
        session.permanent = True
        return redirect(_safe_next(request.args.get('next')))

    return render_template('login.html', error=None, form=form)


@bp.route('/logout', methods=('POST',))
def logout():
    session.clear()
    g.user = None
    return redirect(url_for('auth.login'))


@bp.route('/api/me')
def me():
    if g.user is None:
        return jsonify({'authenticated': False, 'user': None}), 401
    return jsonify({'authenticated': True,
                    'user': {'id': g.user['id'], 'name': g.user['name'],
                             'email': g.user['email']}})


# --- wiring -----------------------------------------------------------------

def init_app(app):
    """Register the blueprint, session config and request hooks."""
    app.secret_key = _session_secret()
    app.config.setdefault('PERMANENT_SESSION_LIFETIME', timedelta(days=7))
    app.config.setdefault('SESSION_COOKIE_HTTPONLY', True)
    app.config.setdefault('SESSION_COOKIE_SAMESITE', 'Lax')
    app.config.setdefault(
        'SESSION_COOKIE_SECURE',
        os.environ.get('COOKIE_SECURE', '').lower() in ('1', 'true', 'yes', 'on'))

    app.register_blueprint(bp)
    app.teardown_appcontext(close_db)
    app.before_request(load_logged_in_user)

    @app.context_processor
    def _inject_user():
        return {'current_user': g.get('user')}
