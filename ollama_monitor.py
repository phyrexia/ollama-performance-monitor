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
                    response_json TEXT
                )
            ''')

    def query(self, model_id: str, prompt: str):
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

        # Log to DB
        self._log_to_db(
            model_id, prompt, response['response'] if isinstance(response, dict) else response.response, 
            latency, in_tokens, out_tokens, tps, req_json, resp_json
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
        """Executes a benchmark over a list of models."""
        print(f"\n📊 OLLAMA PERFORMANCE MONITOR - Prompt: '{prompt}'")
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

            print(f"⌛ Measuring speed on {model_id}...")
            try:
                res = self.query(model_id, prompt)
                results[model_id] = res["metrics"]
            except Exception as e:
                results[model_id] = {"error": str(e)}
        
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

    def _log_to_db(self, model, prompt, resp, latency, in_t, out_t, tps, req_json=None, resp_json=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO ollama_logs 
                (timestamp, model_name, prompt, full_response, latency_seconds, input_tokens, output_tokens, tokens_per_sec, request_json, response_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), model, prompt, resp, latency, in_t, out_t, tps, req_json, resp_json))

if __name__ == "__main__":
    PROXY = OllamaProxy()
    
    # 1. Show local models
    print("Available Local Models (Ollama):")
    available = PROXY.list_local_models()
    if not available:
        print("❌ No models found in Ollama. Make sure Ollama is running.")
        sys.exit(1)
        
    for m in available:
        print(f" - {m}")

    # 2. Benchmark Configuration
    fixed_prompt = "Generate a short science fiction story of approximately 100 tokens."
    print("\n--- BENCHMARK CONFIGURATION ---")
    filter_kw = input("Enter a keyword to choose models (or press Enter for defaults): ").lower()
    
    if filter_kw:
        models_to_test = [m for m in available if filter_kw in m.lower()]
        if not models_to_test:
            print(f"⚠️ No models found matching '{filter_kw}'.")
            models_to_test = []
    else:
        # Defaults if no filter
        models_to_test = available[:2] # Default to first 2 models
    
    if models_to_test:
        print(f"Selected models: {models_to_test}")
        results = PROXY.run_benchmark(fixed_prompt, models_to_test)
        
        print("\n--- FINAL RESULTS ---")
        for mod, metrics in results.items():
            print(f"Model: {mod} -> {metrics}")
            
        ans = input("\n📊 Would you like to generate a visual performance report? (y/n): ").lower()
        if ans == 'y':
            PROXY.export_performance_charts(results)
    else:
        print("\n❌ No models selected for benchmark.")
