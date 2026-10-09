"""FastAPI Uvicorn Local Development Launcher Script (Step 17).

Starts the local Uvicorn ASGI server binding to localhost by default:
http://127.0.0.1:8000
Interactive OpenAPI Documentation: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Fake News Detection FastAPI Backend")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "127.0.0.1"),
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8000)),
        help="Port number to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for local development",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Fake News Detection Using LLMs - FastAPI Server (Step 17)")
    print("=" * 80)
    print(f"Target Interface : http://{args.host}:{args.port}")
    print(f"Swagger API Docs : http://{args.host}:{args.port}/docs")
    print(f"ReDoc Docs       : http://{args.host}:{args.port}/redoc")
    print(f"OpenAPI Schema   : http://{args.host}:{args.port}/openapi.json")
    print("=" * 80)

    uvicorn.run(
        "backend.app.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
