"""Check environment and start API server."""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Checking dependencies...")

try:
    import fastapi
    print(f"✅ FastAPI {fastapi.__version__}")
except ImportError as e:
    print(f"❌ FastAPI not installed: {e}")
    sys.exit(1)

try:
    import uvicorn
    print(f"✅ Uvicorn installed")
except ImportError as e:
    print(f"❌ Uvicorn not installed: {e}")
    sys.exit(1)

try:
    import langgraph
    print(f"✅ LangGraph installed")
except ImportError as e:
    print(f"❌ LangGraph not installed: {e}")
    sys.exit(1)

try:
    import aiosqlite
    print(f"✅ Aiosqlite installed")
except ImportError as e:
    print(f"⚠️  Aiosqlite not installed (SQLite checkpointer will fallback to MemorySaver): {e}")

print("\n✅ All critical dependencies ready!")
print("\nTo start the API server, run:")
print("  cd c:/Users/DJ/Desktop/git/ai english/ai-english-tutor")
print("  uvicorn api.main:app --reload --port 8000")
print("\nThen visit: http://localhost:8000/docs")
