# Ollama Performance Monitor 🦙

A standalone benchmarking tool for local LLMs running on Ollama. Track speed, latency, and resource usage for your local models with ease.

## Features ✨

- **Local Model Discovery**: Automatically lists models currently available in your Ollama instance.
- **Speed & Latency Tracking**: Measures Tokens Per Second (TPS) and total round trip time.
- **SQLite Database**: Persists all benchmark results in `ollama_performance.db` for long-term tracking.
- **Full JSON Logging**: Stores the complete request and response JSON for every inference.
- **Visual Reports**: Generates professional PNG charts comparing local model performance.
- **Smart Caching**: Identifies duplicate tests and asks before re-running to save resources.

## Installation 🛠️

1. **Prerequisites**:
   - Install [Ollama](https://ollama.com/) and have it running.
   - Pull some models (e.g., `ollama pull gemma3`).

2. **Clone and Setup**:
   ```bash
   git clone https://github.com/phyrexia/ollama-performance-monitor.git
   cd ollama-performance-monitor
   ```

3. **Install Python dependencies**:
   ```bash
   pip install ollama matplotlib numpy
   ```

## Usage 🚀

Start the monitor:
```bash
python ollama_monitor.py
```

### Options:
- **Filtering**: Enter keywords to select specific models to benchmark.
- **Visualization**: Generate a `performance_report.png` after the benchmark run.

## Project Structure 📁

- `ollama_monitor.py`: Core benchmarking and visualization logic.
- `ollama_performance.db`: Local SQLite database for results (git-ignored).
- `.gitignore`: Keeps your local environment clean and secure.

## License 📄
MIT License
