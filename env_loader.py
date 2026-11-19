"""Minimal .env loader — stdlib only, no python-dotenv dependency.

Reads ``KEY=VALUE`` lines from a .env file next to the app and puts them in
``os.environ`` **without overriding variables that are already set**, so the
real deployment environment always wins over a local file.

Supported syntax: ``KEY=value``, ``KEY="value"``, ``KEY='value'``, ``export KEY=value``,
``#`` comments and blank lines. Values are not interpolated.
"""
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')


def parse_env_file(path):
    """Return an ordered dict of key -> value parsed from ``path``."""
    values = {}
    try:
        with open(path, 'r', encoding='utf-8-sig') as handle:
            lines = handle.readlines()
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return values

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.lower().startswith('export '):
            line = line[7:].strip()
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env(path=DEFAULT_PATH):
    """Load a .env file into os.environ. Returns the keys actually applied."""
    applied = []
    for key, value in parse_env_file(path).items():
        if key in os.environ:
            continue
        os.environ[key] = value
        applied.append(key)
    return applied
