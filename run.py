import os
import sys
import socket
import uvicorn

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def main():
    port = int(os.environ.get("PORT", 8000))
    if is_port_in_use(port):
        print(f"⚠️ Port {port} is already in use by another process.")

    print("==================================================")
    print("🚀 GitHub Code RAG (OLMo + LangGraph)")
    print(f"🌐 Web UI & API live at: http://localhost:{port}")
    print("==================================================")

    sys.path.insert(0, os.path.abspath("src"))
    should_reload = os.environ.get("RELOAD", "false").lower() in ("true", "1")
    uvicorn.run(
        "src.code_rag.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=should_reload
    )

if __name__ == "__main__":
    main()
