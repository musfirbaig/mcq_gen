"""
Vercel serverless entry point for the MCQ Generator FastAPI app.

Vercel's Python runtime calls the `handler` (Mangum) for each request.
`app` is also exported so tools like `vercel dev` can find it directly.
"""

import sys
import os

# Add the project root to the path so all local modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401
from mangum import Mangum

# Vercel invokes `handler` for each serverless request
handler = Mangum(app, lifespan="off")
