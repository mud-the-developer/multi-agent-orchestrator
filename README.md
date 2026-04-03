# Multi-Agent Orchestration System & Real-Time WebUI

[🇰🇷 한국어 버전 (Korean)](README_ko.md)

This project features a multi-agent AI orchestration pipeline (using models like Gemma) paired with a highly specialized **Real-Time Analytics & Hardware Monitoring Web UI** built in Rust.

## 🌟 Key Features

- **Multi-Agent Flow**: Coordinations between Planner, Coder, and Vision agents handling specific tasks independently.
- **Live Hardware Telemetry**: Native Rust hooks (`sysinfo`, `nvml-wrapper`) replacing the need for external CLI tools like `btop` and `nvtop`. CPU, System RAM, and Nvidia GPU VRAM metrics are streamed dynamically.
- **ApexCharts Dashboard**: Interactive, animated data visualizations for LLM Velocity (Tokens/ms) and Hardware Utilization.
- **Mermaid Protocol View**: A secondary tab constructs a running sequence diagram (`Actor -> Receiver`) as agents talk, tracking protocol state changes in real-time.
- **1-Click CSV Export**: Frontend extraction of research logging telemetry. Download event traces exactly as they fired without needing backend storage interactions.

---

## 🚀 Easy Setup & Installation (using `mise`)

This project provides unified setup for both Rust and Python using `mise` (a modern runtime executor).

1. **Install mise** (if you don't have it):
   ```bash
   curl https://mise.run | sh
   ```
2. **Install tools & dependencies**:
   ```bash
   # This will automatically install python, rust, create a .venv, 
   # install pip requirements, and compile the rust UI!
   mise run install
   ```

## 🚀 How to Run

### 1. Launch the Rust WebUI Backend
The UI backend runs separately from the Python AI ecosystem. Because it actively monitors hardware, you will start it using `cargo` (or our mise wrapper).

Open a fresh terminal:
```bash
mise run run-ui
```
*The server will mount on `http://0.0.0.0:3123`.*
*(Note: Requires a Linux system with active NVIDIA drivers for `nvml` GPU telemetry).*

### 2. Launch the Orchestrator Process 
In a separate terminal, trigger your AI payload using the provided runner. 

```bash
# Example syntax using the strict vocab configuration via mise:
mise run run-orchestrator
# Or manually: python multi_agent_hf_gemma4_args_with_gui.py --config gemma4_24gb_strict_vocab.json "Write a tiny websocket chat server" --webhook-url http://0.0.0.0:3123/api/hook
```

### 3. Open the Dashboard
With both services operating, open your web browser to:
[http://127.0.0.1:3123](http://127.0.0.1:3123)

- **Dashboard Tab**: Check here first to make sure your Rust node is successfully plotting live CPU and GPU memory loads without python running.
- **Live Feed Tab**: Watch individual agent messages and LLM blocks stream smoothly without text truncation.
- **Protocol Tab**: Watch a live sequence diagram map out the multi-agent communications logic.
