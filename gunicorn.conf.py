"""Gunicorn configuration for Linux deployments (AWS App Runner, containers).

Start with:

    gunicorn -c gunicorn.conf.py app:app

Every value can be overridden by an environment variable so the same image works
locally, in CI and on App Runner without editing files.
"""
import os

# App Runner injects PORT (defaults to 8080). Bind all interfaces inside the container.
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# The TF-IDF model is fitted per worker at import (~7 ms, ~25 KiB), so a couple of
# workers is plenty; preload is deliberately not used to keep worker memory isolated.
workers = int(os.environ.get('WEB_CONCURRENCY', '2'))
threads = int(os.environ.get('GUNICORN_THREADS', '2'))
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '60'))
graceful_timeout = int(os.environ.get('GUNICORN_GRACEFUL_TIMEOUT', '30'))
keepalive = int(os.environ.get('GUNICORN_KEEPALIVE', '5'))

# Log to stdout/stderr so App Runner captures them.
accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')

forwarded_allow_ips = '*'          # behind the App Runner load balancer
secure_scheme_headers = {'X-FORWARDED-PROTO': 'https'}
