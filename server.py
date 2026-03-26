"""
ASGI entrypoint for the FastAPI app (avoids name clash between root `api.py` and package `api/`).

Run:
  uvicorn server:app --host 0.0.0.0 --port 8000

Web UI: http://localhost:8000/app/
"""

import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "ai_detection_fastapi_app", _root / "api.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
app = _mod.app
