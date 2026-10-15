"""
wsgi.py - Production WSGI entry point for MachinaSense web app
Run with: gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 1
Or dev:   python web/app.py
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.app import app, train_model, train_ottawa_model

# Train on first import (for gunicorn / WSGI servers)
_t1 = threading.Thread(target=train_model,        daemon=True)
_t2 = threading.Thread(target=train_ottawa_model, daemon=True)
_t1.start()
_t2.start()
