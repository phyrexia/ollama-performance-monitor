import eventlet
eventlet.monkey_patch()

import os
import sqlite3
import json
import threading
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
        print(f"Benchmark Task Error: {e}")
        socketio.emit('error', {'message': str(e)})

if __name__ == '__main__':
    # Ensure templates folder exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    socketio.start_background_task(background_hw_monitor)
    socketio.run(app, host='0.0.0.0', port=5050, debug=True)
