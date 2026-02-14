import os
import sys
import time
import sqlite3
import json
from datetime import datetime
import ollama
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for server/terminal environments
import matplotlib
matplotlib.use('Agg')
import psutil
from jinja2 import Template
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def parse_int_input(val: str) -> int:
    """Parses string inputs with K/M suffixes into integers."""
    if not val:
        return 0
    val = val.upper().strip()
    multiplier = 1
    if val.endswith('K'):
        multiplier = 1024
        val = val[:-1]
    elif val.endswith('M'):
        multiplier = 1024 * 1024
        val = val[:-1]
    
    try:
        return int(float(val) * multiplier)
    except ValueError:
        return 0

class OllamaProxy:
    def __init__(self, db_path: str = "ollama_performance.db"):
        self.client = ollama.Client()
        
        # If path is relative, place it in the same directory as the script
        if not os.path.isabs(db_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, db_path)
        else:
            self.db_path = db_path
            
        self._init_db()

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Captures current CPU, RAM and GPU metrics."""
        # Get percentages (interval=None for instant reading)
        info = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu_load": 0.0,
            "gpu_temp": 0.0,
            "vram_used": 0.0
        }
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0] # Monitor primary GPU
                    info["gpu_load"] = gpu.load * 100
                    info["gpu_temp"] = gpu.temperature
                    info["vram_used"] = gpu.memoryUsed
            except Exception:
                pass
        return info

    def _init_db(self):
        """Creates the logs table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ollama_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    model_name TEXT,
                    prompt TEXT,
                    full_response TEXT,
                    latency_seconds REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    tokens_per_sec REAL,
                    request_json TEXT,
                    response_json TEXT,
                    test_type TEXT DEFAULT 'standard',
                    accuracy_score REAL DEFAULT 1.0,
                    cpu_percent REAL,
                    ram_percent REAL,
                    gpu_load REAL,
                    gpu_temp REAL,
                    vram_used_mb REAL
                )
            ''')
            # Migration check
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(ollama_logs)")
            columns = [column[1] for column in cursor.fetchall()]
            
            new_columns = {
                "test_type": "TEXT DEFAULT 'standard'",
                "accuracy_score": "REAL DEFAULT 1.0",
                "cpu_percent": "REAL",
                "ram_percent": "REAL",
                "gpu_load": "REAL",
                "gpu_temp": "REAL",
                "vram_used_mb": "REAL"
            }
            
            for col, col_def in new_columns.items():
                if col not in columns:
                    conn.execute(f"ALTER TABLE ollama_logs ADD COLUMN {col} {col_def}")

    def query(self, model_id: str, prompt: str, log_to_db: bool = True, test_type: str = "standard"):
        # Latency: Total Round Trip Time
        start_time = time.perf_counter()
        
        request_payload = {
            "model": model_id,
            "prompt": prompt,
            "options": {"temperature": 0.7}
        }

        # Call execution with the Ollama client
        response = self.client.generate(
            model=model_id,
            prompt=prompt,
            options={"temperature": 0.7}
        )
        
        end_time = time.perf_counter()
        
        # Metrics
        latency = end_time - start_time
        in_tokens = response.get('prompt_eval_count', 0)
        out_tokens = response.get('eval_count', 0)
        
        # TPS: Generated Output tokens / Total latency
        tps = out_tokens / latency if latency > 0 else 0

        # Serialize payloads
        req_json = json.dumps(request_payload)
        try:
            # Handle Pydantic models in newer ollama versions
            resp_json = json.dumps(response.model_dump() if hasattr(response, 'model_dump') else dict(response))
        except Exception:
            resp_json = str(response)

        # Log to DB only if requested
        if log_to_db:
            hw = self._get_hardware_info()
            self._log_to_db(
                model_id, prompt, response['response'] if isinstance(response, dict) else response.response, 
                latency, in_tokens, out_tokens, tps, req_json, resp_json, test_type=test_type, hw_info=hw
            )

        resp_text = response['response'] if isinstance(response, dict) else response.response
        return {
            "response": resp_text,
            "metrics": {
                "tps": f"{tps:.2f} tokens/s",
                "latency": f"{latency:.2f}s",
                "tokens": f"{in_tokens}/{out_tokens}"
            }
        }

    def list_local_models(self):
        """Lists models currently installed in the local Ollama instance."""
        try:
            resp = self.client.list()
            # In ollama-python >= 0.3.0, models are objects with a .model attribute
            return [m.model for m in resp.models]
        except Exception as e:
            print(f"❌ Error listing Ollama models: {e}")
            return []

    def _check_recent_test(self, model_id: str, prompt: str):
        """Checks if the same model and prompt were tested in the last 10 days."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = """
                SELECT timestamp, latency_seconds, tokens_per_sec 
                FROM ollama_logs 
                WHERE model_name = ? AND prompt = ? 
                AND datetime(timestamp) >= datetime('now', '-10 days')
                ORDER BY timestamp DESC LIMIT 1
            """
            cursor.execute(query, (model_id, prompt))
            return cursor.fetchone()

    def run_benchmark(self, prompt: str, models: list):
        """Executes a benchmark over a list of models with warm-up and consistency check."""
        print(f"\n📊 OLLAMA PERFORMANCE MONITOR (Standard) - Prompt: '{prompt}'")
        results = {}
        for model_id in models:
            recent = self._check_recent_test(model_id, prompt)
            if recent:
                timestamp, lat, tps = recent
                print(f"\n⚠️ Model {model_id} was already tested on {timestamp}")
                print(f"   (Previous results: {tps:.2f} tokens/s, Latency: {lat:.2f}s)")
                option = input(f"   Do you want to repeat the test for {model_id}? (y/n): ").lower()
                if option != 'y':
                    print(f"⏭️  Skipping {model_id}...")
                    results[model_id] = {
                        "tps": f"{tps:.2f} tokens/s",
                        "latency": f"{lat:.2f}s",
                        "info": "Historical (last 10 days)"
                    }
                    continue

            print(f"\n⌛ Model {model_id}:")
            try:
                # 1. Warm-up (discard results, just load to memory)
                print(f"   - Phase 1: Warming up (Cold run / Loading)...")
                self.query(model_id, prompt, log_to_db=False)
                
                # 2. Measured Run 1
                print(f"   - Phase 2: Measured run 1/2...")
                res1 = self.query(model_id, prompt, log_to_db=True, test_type="standard")
                tps1 = float(res1["metrics"]["tps"].split()[0])
                
                # 3. Measured Run 2
                print(f"   - Phase 3: Measured run 2/2...")
                res2 = self.query(model_id, prompt, log_to_db=True, test_type="standard")
                tps2 = float(res2["metrics"]["tps"].split()[0])
                
                # Consistency Check
                dispersion = abs(tps1 - tps2) / max(tps1, tps2) if max(tps1, tps2) > 0 else 0
                avg_tps = (tps1 + tps2) / 2
                avg_latency = (float(res1["metrics"]["latency"].rstrip('s')) + 
                              float(res2["metrics"]["latency"].rstrip('s'))) / 2
                
                if dispersion > 0.10:
                    print(f"   ⚠️ Warning: High dispersion detected ({dispersion:.1%}). Results might be unstable.")
                else:
                    print(f"   ✅ Consistency check passed (Dispersion: {dispersion:.1%}).")

                results[model_id] = {
                    "tps": f"{avg_tps:.2f} tokens/s",
                    "latency": f"{avg_latency:.2f}s",
                    "tokens": res2["metrics"]["tokens"],
                    "dispersion": f"{dispersion:.1%}"
                }
            except Exception as e:
                print(f"   ❌ Error benchmarking {model_id}: {e}")
                results[model_id] = {"error": str(e)}
        
        return results

    def run_concurrency_test(self, model_id: str, users: int = 5, prompt: str = "Explain the importance of local LLMs."):
        """Simulates parallel users making requests to measure throughput degradation."""
        print(f"\n🚀 CONCURRENCY LOAD TEST - Model: {model_id} | Users: {users}")
        
        from concurrent.futures import ThreadPoolExecutor
        
        start_time = time.perf_counter()
        results = []
        
        with ThreadPoolExecutor(max_workers=users) as executor:
            # Launch all requests in parallel
            futures = [executor.submit(self.query, model_id, prompt, log_to_db=True, test_type=f"concurrency_{users}") for _ in range(users)]
            
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"❌ Worker Error: {e}")
        
        total_duration = time.perf_counter() - start_time
        
        # Aggregate metrics
        total_tokens = 0
        latencies = []
        for r in results:
            # Parse tokens from string "in/out"
            try:
                _, out = r['metrics']['tokens'].split('/')
                total_tokens += int(out)
                latencies.append(float(r['metrics']['latency'].replace('s', '')))
            except:
                pass
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        overall_tps = total_tokens / total_duration if total_duration > 0 else 0
        
        print("\n--- CONCURRENCY RESULTS ---")
        print(f"Total Requests: {len(results)}")
        print(f"Total Time: {total_duration:.2f}s")
        print(f"Average Latency: {avg_latency:.2f}s")
        print(f"Aggregate Throughput: {overall_tps:.2f} tokens/s")
        
        return {
            "users": users,
            "duration": total_duration,
            "avg_latency": avg_latency,
            "overall_tps": overall_tps
        }

    def run_context_stress_test(self, model_id: str, steps: int = 5, increment_tokens: int = 512, num_ctx: int = 4096):
        """Needle in a Haystack: Tests memory retention as context grows."""
        print(f"\n🧠 CONTEXT STRESS TEST - Model: {model_id} (num_ctx: {num_ctx})")
        secret_key = "DRAGON-AZUL-2026"
        needle_context = f"NOTE: The secret vault key is {secret_key}. Do not forget it."
        
        # Configure model context window
        options = {"temperature": 0.0, "num_ctx": num_ctx}
        
        messages = [{"role": "system", "content": "You are a helpful assistant. Remember the secret key provided."}]
        messages.append({"role": "user", "content": f"Hi. {needle_context} Please confirm the key."})
        
        print(f"⌛ Phase 0: Initializing conversation with secret key...")
        resp = self.client.chat(model=model_id, messages=messages, options=options)
        messages.append({"role": "assistant", "content": resp.message.content})
        
        results = []
        
        # Optimized filler block (~1024 tokens of text)
        filler_block = ("The quick brown fox jumps over the lazy dog. " * 20 + 
                       "Artificial Intelligence is transforming the world. " * 10 +
                       "Benchmarking local models is essential for performance. " * 10 +
                       "Ollama makes it easy to run LLMs locally on your machine. ")

        for i in range(1, steps + 1):
            print(f"⌛ Step {i}/{steps}: Increasing context (+~{increment_tokens} tokens)...")
            
            # Efficiently build large volume of filler
            num_blocks = max(1, increment_tokens // 1000)
            filler = (filler_block + "\n") * num_blocks
            
            # Step execution: Query the model with incremental history
            start_time = time.perf_counter()
            try:
                # Add the new filler segment to the conversation PERMANENTLY
                messages.append({"role": "user", "content": filler})
                
                # Query recall
                recall_msg = "RECALL CHALLENGE: What was the secret vault key mentioned at the very beginning? Be precise."
                response = self.client.chat(model=model_id, 
                                          messages=messages + [{"role": "user", "content": recall_msg}],
                                          options=options)
                
                # Add a placeholder assistant response to keep the turn order
                messages.append({"role": "assistant", "content": "Acknowledged. Segment processed."})
            except Exception as e:
                print(f"   ❌ API Error at Step {i} (Likely OOM): {e}")
                break

            latency = time.perf_counter() - start_time
            
            content = response.message.content
            # Improved accuracy check (case insensitive)
            accuracy = 1.0 if secret_key.lower() in content.lower() else 0.0
            
            in_tokens = response.get('prompt_eval_count', 0)
            out_tokens = response.get('eval_count', 0)
            tps = out_tokens / latency if latency > 0 else 0
            
            print(f"   - Current Context: ~{in_tokens} tokens")
            print(f"   - Results: {tps:.2f} tokens/s, Accuracy: {'✅ OK' if accuracy == 1.0 else '❌ FAILED'}")
            
            # Log this specific step to DB
            hw = self._get_hardware_info()
            self._log_to_db(model_id, filler, content, latency, in_tokens, out_tokens, tps, 
                            json.dumps(messages), str(response), test_type="stress", accuracy_score=accuracy, hw_info=hw)
            
            results.append({
                "step": i,
                "context_tokens": in_tokens,
                "tps": tps,
                "accuracy": accuracy
            })
            
            if accuracy == 0:
                print(f"🛑 RECALL LOST at {in_tokens} tokens. Final response: {content[:100]}...")
                break
                
        return results

    def export_performance_charts(self, results: Dict[str, Any], filename: str = "performance_report.png"):
        """Generates a PNG report with TPS and Latency comparisons."""
        models = [m for m in results if "tps" in results[m]]
        if not models:
            print("⚠️ No data available to generate charts.")
            return

        tps_values = [float(results[m]["tps"].split()[0]) for m in models]
        latency_values = [float(results[m]["latency"].rstrip('s')) for m in models]
        
        display_names = [m.replace(":latest", "").replace("llama", "L").replace("gemma", "G") for m in models]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.4)

        # 1. TPS Chart
        colors = plt.cm.plasma(np.linspace(0, 1, len(models)))
        bars1 = ax1.bar(display_names, tps_values, color=colors)
        ax1.set_title("Ollama Speed Comparison (Tokens per Second)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("TPS")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.bar_label(bars1, fmt='%.2f')

        # 2. Latency Chart
        bars2 = ax2.bar(display_names, latency_values, color=colors, alpha=0.8)
        ax2.set_title("Ollama Latency Comparison (Seconds)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Seconds")
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.bar_label(bars2, fmt='%.2fs')

        plt.figtext(0.5, 0.02, f"Ollama Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    ha="center", fontsize=10, bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})

        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visual report saved to: {save_path}")
        plt.close()

    def export_interactive_dashboard(self, filename: str = "performance_dashboard.html"):
        """Generates a premium interactive HTML dashboard with Plotly charts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM ollama_logs ORDER BY timestamp DESC LIMIT 100")
            logs = [dict(row) for row in cursor.fetchall()]

        if not logs:
            print("⚠️ No logs found to generate dashboard.")
            return

        # Prepare summary stats
        total_models = len(set(log['model_name'] for log in logs))
        avg_tps = sum(log['tokens_per_sec'] for log in logs) / len(logs)
        best_tps = max(log['tokens_per_sec'] for log in logs)
        
        # Prepare Chart 1: Avg TPS per Model
        model_tps = {}
        for log in logs:
            m = log['model_name']
            if m not in model_tps: model_tps[m] = []
            model_tps[m].append(log['tokens_per_sec'])
        
        tps_chart_data = {
            "labels": list(model_tps.keys()),
            "values": [sum(v)/len(v) for v in model_tps.values()]
        }

        # Prepare Chart 2: Hardware Telemetry (Timeline)
        # Sort logs by timestamp for timeline
        timed_logs = sorted(logs, key=lambda x: x['timestamp'])[-20:]
        hw_chart_data = {
            "labels": [l['timestamp'][11:19] for l in timed_logs],
            "cpu": [l['cpu_percent'] if l['cpu_percent'] else 0 for l in timed_logs],
            "vram": [l['vram_used_mb']/100 if l['vram_used_mb'] else 0 for l in timed_logs] # Scaled for visibility
        }

        # Render Template
        base_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(base_dir, "dashboard_template.html")
        
        try:
            with open(template_path, "r") as f:
                template = Template(f.read())
        except FileNotFoundError:
            print("❌ Dashboard template not found.")
            return

        html_out = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_models=total_models,
            avg_tps=f"{avg_tps:.2f}",
            best_tps=f"{best_tps:.2f}",
            total_logs=len(logs),
            logs=logs,
            tps_chart_data=tps_chart_data,
            hw_chart_data=hw_chart_data
        )

        save_path = os.path.join(base_dir, filename)
        with open(save_path, "w") as f:
            f.write(html_out)
        
        print(f"\n✨ Interactive dashboard generated: {save_path}")
        return save_path

    def _log_to_db(self, model, prompt, resp, latency, in_t, out_t, tps, req_json=None, resp_json=None, test_type="standard", accuracy_score=1.0, hw_info=None):
        if hw_info is None:
            hw_info = {"cpu_percent": None, "ram_percent": None, "gpu_load": None, "gpu_temp": None, "vram_used": None}
            
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO ollama_logs 
                (timestamp, model_name, prompt, full_response, latency_seconds, input_tokens, output_tokens, tokens_per_sec, 
                 request_json, response_json, test_type, accuracy_score, cpu_percent, ram_percent, gpu_load, gpu_temp, vram_used_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), model, prompt, resp, latency, in_t, out_t, tps, 
                  req_json, resp_json, test_type, accuracy_score, 
                  hw_info["cpu_percent"], hw_info["ram_percent"], hw_info["gpu_load"], 
                  hw_info["gpu_temp"], hw_info["vram_used"]))

if __name__ == "__main__":
    PROXY = OllamaProxy()
    
    # 1. Mode Selection First
    print("\n--- OLLAMA PERFORMANCE MONITOR ---")
    print("1. Standard Benchmark (Speed / Latency)")
    print("2. Context Stress Test (Needle in a Haystack)")
    print("3. Concurrency Load Test (Parallel Users)")
    mode = input("Select mode (1/2/3, default: 1): ") or "1"

    # 2. Show local models
    print("\nAvailable Local Models (Ollama):")
    available = PROXY.list_local_models()
    if not available:
        print("❌ No models found in Ollama. Make sure Ollama is running.")
        sys.exit(1)
        
    for m in available:
        print(f" - {m}")

    if mode == "1":
        # Standard logic
        fixed_prompt = "Generate a short science fiction story of approximately 100 tokens."
        print("\n--- BENCHMARK CONFIGURATION ---")
        filter_kw = input("Enter a keyword to choose models (or press Enter for defaults): ").lower()
        
        if filter_kw:
            models_to_test = [m for m in available if filter_kw in m.lower()]
            if not models_to_test:
                print(f"⚠️ No models found matching '{filter_kw}'.")
                models_to_test = []
        else:
            models_to_test = available[:2]
        
        if models_to_test:
            results = PROXY.run_benchmark(fixed_prompt, models_to_test)
            print("\n--- FINAL RESULTS ---")
            for mod, metrics in results.items():
                print(f"Model: {mod} -> {metrics}")
            
            ans = input("\n📊 Would you like to generate a visual performance report? (PNG: p, Interactive HTML: h, Both: b, None: n): ").lower()
            if ans in ['p', 'b']:
                PROXY.export_performance_charts(results)
            if ans in ['h', 'b']:
                PROXY.export_interactive_dashboard()
    elif mode == "2":
        # Stress Test logic
        print("\n--- STRESS TEST CONFIGURATION ---")
        model_name = input(f"Enter model name to test (available: {available}): ")
        if model_name not in available:
            # Try fuzzy match
            match = [m for m in available if model_name in m]
            if match:
                model_name = match[0]
                print(f"Using fuzzy match: {model_name}")
            else:
                print("❌ Invalid model.")
                sys.exit(1)
        
        steps_input = input("Number of steps (incremental context)? (default: 5): ")
        steps = int(steps_input) if steps_input else 5
        
        inc_input = input("Tokens to add per step? (default: 1024, try 50K for high context): ")
        inc = parse_int_input(inc_input) if inc_input else 1024
        
        ctx_input = input("Config MAX Context (num_ctx)? (default: 8192, try 128K): ")
        num_ctx = parse_int_input(ctx_input) if ctx_input else 8192
        
        stress_results = PROXY.run_context_stress_test(model_name, steps=steps, increment_tokens=inc, num_ctx=num_ctx)
        print("\n--- STRESS TEST FINAL SUMMARY ---")
        for sr in stress_results:
            status = "✅ OK" if sr["accuracy"] == 1.0 else "❌ FAIL"
            print(f"Step {sr['step']}: Context {sr['context_tokens']} tokens -> {sr['tps']:.2f} tps | {status}")
    elif mode == "3":
        # Concurrency Load Test logic
        print("\n--- CONCURRENCY LOAD TEST ---")
        model_name = input(f"Enter model name to test (available: {available}): ")
        # Fuzzy match logic
        if model_name not in available:
            match = [m for m in available if model_name in m]
            if match:
                model_name = match[0]
                print(f"Using fuzzy match: {model_name}")
            else:
                print("❌ Invalid model.")
                sys.exit(1)
        
        users_input = input("Number of parallel users? (default: 5): ")
        users = int(users_input) if users_input else 5
        
        custom_prompt = input("Custom prompt? (press Enter for default): ") or "Explain the importance of local LLMs."
        
        PROXY.run_concurrency_test(model_name, users=users, prompt=custom_prompt)
