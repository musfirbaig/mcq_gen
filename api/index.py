"""
Vercel serverless entry point for the MCQ Generator FastAPI app.

Vercel's Python runtime looks for an `app` variable (ASGI/WSGI) in this file.
All requests are routed here via vercel.json.
"""

import sys
import os

# Add the project root to the path so all local modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401  – Vercel picks up `app` automatically
