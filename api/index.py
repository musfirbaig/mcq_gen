"""
Vercel serverless entry point for the MCQ Generator FastAPI app.
"""

import sys
import os

# Add the project root to the path so all local modules resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # Vercel picks up `app` as an ASGI application
