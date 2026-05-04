"""
Quick Start Script for MCQ Generator API

This script helps you get started with the API server.
"""

import os
import sys
from pathlib import Path


def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("[X] .env file not found!")
        print("\nPlease create .env file:")
        print("  1. Copy .env.example to .env:")
        print("     cp .env.example .env")
        print("\n  2. Edit .env and add your credentials:")
        print("     - MONGODB_URI (MongoDB Atlas connection string)")
        print("     - GOOGLE_API_KEY (or ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        return False
    
    print("[OK] .env file found")
    
    # Check for required variables
    from dotenv import load_dotenv
    load_dotenv()
    
    mongodb_uri = os.getenv("MONGODB_URI")
    google_key = os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not mongodb_uri or "your_" in mongodb_uri or "username:password" in mongodb_uri:
        print("[X] MONGODB_URI not configured in .env")
        print("   Please add your MongoDB Atlas connection string")
        return False
    
    print("[OK] MONGODB_URI configured")
    
    if not any([google_key, anthropic_key, openai_key]):
        print("[X] No LLM API key configured in .env")
        print("   Please add at least one of:")
        print("   - GOOGLE_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - OPENAI_API_KEY")
        return False
    
    if google_key and "your_" not in google_key:
        print("[OK] GOOGLE_API_KEY configured")
    if anthropic_key and "your_" not in anthropic_key:
        print("[OK] ANTHROPIC_API_KEY configured")
    if openai_key and "your_" not in openai_key:
        print("[OK] OPENAI_API_KEY configured")
    
    return True


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import motor
        import pymongo
        print("[OK] All dependencies installed")
        return True
    except ImportError as e:
        print(f"[X] Missing dependencies: {e.name}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False


def start_server():
    """Start the FastAPI server"""
    print("\n" + "="*60)
    print("Starting MCQ Generator API Server...")
    print("="*60)
    
    import uvicorn
    
    print("\nServer will be available at:")
    print("  - API: http://localhost:8000")
    print("  - Docs: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


def main():
    """Main function"""
    print("\n" + "="*60)
    print("MCQ GENERATOR API - STARTUP")
    print("="*60 + "\n")
    
    # Check environment file
    if not check_env_file():
        print("\n" + "="*60)
        print("Setup incomplete. Please configure .env file first.")
        print("="*60)
        sys.exit(1)
    
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\n" + "="*60)
        print("Setup incomplete. Please install dependencies first.")
        print("="*60)
        sys.exit(1)
    
    print("\n[OK] All checks passed! Ready to start server.\n")
    
    # Start server
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Server stopped by user")
        print("="*60)
    except Exception as e:
        print(f"\n\n[X] Server error: {e}")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
