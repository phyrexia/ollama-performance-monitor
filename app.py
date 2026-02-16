import os
import sys

# macOS eventlet compatibility fix: force 'poll' hub
if sys.platform == 'darwin':
    os.environ['EVENTLET_HUB'] = 'poll'

import eventlet
eventlet.monkey_patch()
import sqlite3
import json
import threading
import requests
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ollama_monitor import OllamaProxy

app = Flask(__name__)
CORS(app)
# Increase timeouts for stability during heavy LLM loads
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=60, ping_interval=25)

# Initialize Proxy with a subtle modification to stream status via socketio
class WebOllamaProxy(OllamaProxy):
    def log_status(self, message, status_type="info"):
        print(f"[{status_type.upper()}] {message}")
        # Use socketio.emit directly to ensure it works from background tasks
        socketio.emit('status_update', {'message': message, 'type': status_type})

proxy = WebOllamaProxy()

def background_hw_monitor():
    """Periodically emits hardware stats to the UI."""
    while True:
        try:
            # Re-use global proxy instead of re-initializing
            hw = proxy._get_hardware_info()
            socketio.emit('hw_update', hw)
        except Exception as e:
            print(f"HW Monitor Error: {e}")
        socketio.sleep(2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    try:
        models = proxy.list_local_models()
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history')
def get_history():
    limit = request.args.get('limit', 20, type=int)
    model = request.args.get('model', None)
    db_path = proxy.db_path
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM ollama_logs"
            params = []
            if model and model != 'all':
                query += " WHERE model_name = ?"
                params.append(model)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            logs = [dict(row) for row in cursor.fetchall()]
        return jsonify({"history": logs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/log/<int:log_id>')
def get_log_detail(log_id):
    db_path = proxy.db_path
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM ollama_logs WHERE id = ?", (log_id,))
            row = cursor.fetchone()
            if not row:
                return jsonify({"error": "Log not found"}), 404
            
            # Parse JSON strings if they are valid
            log_data = dict(row)
            for key in ['request_json', 'response_json']:
                if log_data.get(key):
                    try:
                        log_data[key] = json.loads(log_data[key])
                    except:
                        pass
            
        return jsonify(log_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on('start_benchmark')
def handle_benchmark(data):
    # Use background task to avoid blocking the event loop
    socketio.start_background_task(run_benchmark_task, data)

def run_benchmark_task(data):
    model = data.get('model')
    test_type = data.get('test_type', 'standard')
    
    if not model:
        socketio.emit('error', {'message': 'Model name is required'})
        return

    try:
        if test_type == 'standard':
            proxy.log_status(f"Starting standard benchmark for {model}...", 'progress')
            prompt = "Generate a short science fiction story of approximately 100 tokens."
            results = proxy.run_benchmark(prompt, [model])
            socketio.emit('benchmark_complete', {'results': results})
            
        elif test_type == 'stress':
            steps = data.get('steps', 5)
            increment = data.get('increment', 1024)
            num_ctx = data.get('num_ctx', 4096)
            proxy.log_status(f"Starting context stress test for {model}...", 'progress')
            results = proxy.run_context_stress_test(model, steps=steps, increment_tokens=increment, num_ctx=num_ctx)
            socketio.emit('benchmark_complete', {'results': results, 'test_type': 'stress'})
            
        elif test_type == 'concurrency':
            users = data.get('users', 5)
            prompt = data.get('prompt', "Explain the importance of local LLMs.")
            proxy.log_status(f"Starting concurrency test for {model} with {users} users...", 'progress')
            results = proxy.run_concurrency_test(model, users=users, prompt=prompt)
            socketio.emit('benchmark_complete', {'results': results, 'test_type': 'concurrency'})
            
    except Exception as e:
        proxy.log_status(f"Benchmark Task Error: {e}", "error")
        socketio.emit('error', {'message': str(e)})

@app.route('/proxy/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def ollama_proxy(path):
    """Intercepts and logs requests to Ollama."""
    ollama_url = f"http://localhost:11434/{path}"
    method = request.method
    data = request.get_data()
    headers = {k: v for k, v in request.headers if k.lower() != 'host'}

    start_time = time.perf_counter()
    try:
        # Forward request to local Ollama
        resp = requests.request(method, ollama_url, data=data, headers=headers, stream=True)
        
        # If it's a generation request, we try to log the metrics
        is_generation = 'api/generate' in path or 'api/chat' in path
        
        if not is_generation:
            return (resp.content, resp.status_code, resp.headers.items())

        # For generation, we might want to capture the full response if it's not too large
        # or just log that a request happened. For now, let's capture non-streaming ones.
        is_stream = json.loads(data).get('stream', True) if data else True
        
        if is_stream:
            # Handle streaming: we can't easily capture the full response without blocking
            # but we can log that it started.
            proxy.log_status(f"Proxy: Streaming {path} request intercepted", "info")
            return (resp.content, resp.status_code, resp.headers.items())
        
        # Non-streaming: capture everything
        latency = time.perf_counter() - start_time
        resp_json = resp.json()
        
        # Calculate tokens if available
        in_tokens = resp_json.get('prompt_eval_count', 0)
        out_tokens = resp_json.get('eval_count', 0)
        tps = out_tokens / latency if latency > 0 else 0
        
        # Log to DB
        hw = proxy._get_hardware_info()
        model = resp_json.get('model', 'unknown')
        prompt = "Proxied Request" # Could extract from JSON if needed
        
        proxy._log_to_db(
            model, prompt, resp_json.get('response', resp_json.get('message', {}).get('content', '')),
            latency, in_tokens, out_tokens, tps,
            data.decode('utf-8'), json.dumps(resp_json),
            test_type="proxy", hw_info=hw
        )
        
        proxy.log_status(f"Proxy: Logged {model} request ({tps:.2f} tps)", "success")
        return jsonify(resp_json)

    except Exception as e:
        proxy.log_status(f"Proxy Error: {e}", "error")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure templates folder exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    socketio.start_background_task(background_hw_monitor)
    socketio.run(app, host='0.0.0.0', port=5050, debug=True)
