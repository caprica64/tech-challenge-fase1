"""Lambda entry point — wraps the FastAPI app with Mangum."""

from mangum import Mangum
from src.main import app

# Mangum adapts API Gateway / Function URL events to ASGI (FastAPI)
handler = Mangum(app, lifespan="off")
