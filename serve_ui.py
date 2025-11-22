#!/usr/bin/env python3
"""Simple HTTP server for test UI"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

if __name__ == '__main__':
    port = 3000
    print(f"🚀 Starting test UI server on http://localhost:{port}")
    print(f"📂 Open http://localhost:{port}/test_ui.html")

    server = HTTPServer(('localhost', port), CORSRequestHandler)
    server.serve_forever()