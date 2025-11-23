#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

if __name__ == '__main__':
    port = 3000
    print(f"🚀 Starting test UI server on http://0.0.0.0:{port}")
    server = HTTPServer(('0.0.0.0', port), CORSRequestHandler)  # Changed from 'localhost'
    server.serve_forever()