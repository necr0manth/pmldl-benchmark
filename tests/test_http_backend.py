import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vlm_benchmark.backend import OpenAICompatibleBackend


def test_openai_compatible_backend_sends_exact_image_as_data_uri():
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            captured["path"] = self.path
            captured["body"] = json.loads(self.rfile.read(length))
            response = {"choices": [{"message": {"content": '{"tool":"idle","args":{},"confidence":0,"abstain":true}'}}]}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleBackend(f"http://127.0.0.1:{server.server_port}/v1", "fake", timeout_s=2)
        backend.generate("prompt", png, "image/png")
    finally:
        server.shutdown()
        thread.join()
    assert captured["path"] == "/v1/chat/completions"
    content = captured["body"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "prompt"}
    uri = content[1]["image_url"]["url"]
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == png
