"""Lambda entry point — wraps the FastAPI app with Mangum."""

import os
import sys

# Ensure LAMBDA_TASK_ROOT (/var/task) is on sys.path so that the `src`
# package is importable. This is a no-op in local environments where the
# project root is already on the path.
task_root = os.environ.get("LAMBDA_TASK_ROOT", os.path.dirname(__file__))
if task_root not in sys.path:
    sys.path.insert(0, task_root)

from mangum import Mangum          # noqa: E402
from src.main import app           # noqa: E402

# Mangum adapts API Gateway / Lambda Function URL events to ASGI (FastAPI)
handler = Mangum(app, lifespan="off")
