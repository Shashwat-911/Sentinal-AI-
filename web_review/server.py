"""Local JSON report viewer; no camera/model loading or source execution."""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from domain import preview

class Handler(BaseHTTPRequestHandler):
    def reply(self, value, status=200):
        body = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8"),
                  "/sample.json": ("sample.json", "application/json")}
        if self.path not in assets:
            self.reply({"error": "Not found"}, 404)
            return
        name, content_type = assets[self.path]
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write((HERE / name).read_bytes())

    def do_POST(self):
        if self.path != "/api/preview":
            self.reply({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1048576:
                raise ValueError("Report must be between 1 byte and 1 MB.")
            payload = json.loads(self.rfile.read(length))
            self.reply(preview(payload))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.reply({"error": str(exc)}, 400)

    def log_message(self, *_args):
        pass

if __name__ == "__main__":
    port = int(os.getenv("PLAYWRIGHT_PORT", "9102"))
    print(f"Local report review: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
