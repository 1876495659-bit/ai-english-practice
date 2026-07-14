"""Start API server and test it."""
import subprocess, sys, os, time, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("AI English Tutor - Starting Project")
print("=" * 60)

# Check dependencies
print("\n[1/4] Checking dependencies...")
try:
    import fastapi
    print(f"  ✅ FastAPI {fastapi.__version__}")
except ImportError:
    print("  ❌ FastAPI not installed")
    sys.exit(1)

try:
    import uvicorn
    print("  ✅ Uvicorn installed")
except ImportError:
    print("  ❌ Uvicorn not installed")
    sys.exit(1)

try:
    import langgraph
    print("  ✅ LangGraph installed")
except ImportError:
    print("  ❌ LangGraph not installed")
    sys.exit(1)

print("\n[2/4] Starting API server...")
server_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Wait for server to start
time.sleep(3)

if server_process.poll() is None:
    print("  ✅ Server started successfully!")
else:
    print("  ❌ Server failed to start")
    stdout, _ = server_process.communicate()
    print(f"  Error: {stdout[-500:]}")
    sys.exit(1)

print("\n[3/4] Testing API endpoints...")
import httpx

try:
    with httpx.Client(base_url="http://localhost:8000", timeout=10) as client:
        # Test health check
        print("\n  → GET / (Health Check)")
        resp = client.get("/")
        data = resp.json()
        print(f"    Status: {resp.status_code}")
        print(f"    Service: {data['service']}")
        print(f"    Version: {data['version']}")
        print(f"    Architecture: {data['architecture']}")
        print(f"    Nodes: {', '.join(data['nodes'])}")

        # Test session start
        print("\n  → POST /api/session/start")
        resp = client.post("/api/session/start", json={
            "scenario": "daily",
            "difficulty": "medium",
            "level": "intermediate"
        })
        data = resp.json()
        print(f"    Status: {resp.status_code}")
        print(f"    Session ID: {data['session_id']}")
        print(f"    Scenario: {data['scenario_name']}")
        print(f"    Opening: {data['opening_line'][:80]}...")

        # Test chat
        print("\n  → POST /api/chat")
        resp = client.post("/api/chat", json={"message": "I would like to order a hamburger"})
        data = resp.json()
        print(f"    Status: {resp.status_code}")
        print(f"    AI Reply: {data['ai_reply'][:100]}...")
        print(f"    User Input: {data['user_input']}")
        if data.get('correction'):
            print(f"    Has Errors: {data['correction'].get('has_errors')}")
        if data.get('score'):
            print(f"    Total Score: {data['score'].get('total')}/10")

        print("\n✅ All tests passed!")

finally:
    # Stop server
    print("\n[4/4] Stopping server...")
    server_process.terminate()
    server_process.wait(timeout=5)
    print("  ✅ Server stopped")

print("\n" + "=" * 60)
print("Project Demo Complete!")
print("=" * 60)
print("\nTo run the API server yourself:")
print("  cd c:/Users/DJ/Desktop/git/ai english/ai-english-tutor")
print("  uvicorn api.main:app --reload --port 8000")
print("\nThen visit:")
print("  API Docs: http://localhost:8000/docs")
print("  Web UI: streamlit run ui/main.py --server.port 8501")
