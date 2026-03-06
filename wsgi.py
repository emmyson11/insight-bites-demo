import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter_app"))

try:
    from app.server import app
except ModuleNotFoundError:
    from starter_app.app.server import app
