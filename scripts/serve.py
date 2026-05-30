"""Launch the FastAPI application with uvicorn."""

import sys
from pathlib import Path

# Ensure src/ is on the path when running from the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).parent.parent / "src")],
    )
