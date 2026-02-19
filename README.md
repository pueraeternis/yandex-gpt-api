# YandexGPT API Starter Kit

A production-ready reference implementation for interacting with YandexGPT (Yandex Cloud) using Python. This project demonstrates best practices, including **Clean Architecture**, **Async I/O**, **Semantic Search**, and **Function Calling**.

## 🚀 Features

- **Dual Client Implementation**:
  - `Native Client`: Direct REST API usage via `requests`.
  - `SDK Wrapper`: Modern usage via the `openai` Python SDK (fully compatible with Yandex).
- **Advanced AI Capabilities**:
  - **Semantic Search**: vector-based document retrieval (RAG foundation).
  - **Function Calling**: capability to connect LLM with external tools (APIs, calculators, databases).
- **Async Batch Processing**: High-performance script for processing datasets.
- **Production Logging**: Lazy formatting and proper log levels.

## 🛠 Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Recommended) or `pip`.
- Yandex Cloud Account with an active Folder and API Key.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pueraeternis/yandex-gpt-api.git
   cd yandex-gpt-starter
   ```

2. **Install dependencies:**
   Using `uv`:
   ```bash
   uv sync
   ```

3. **Configure Environment:**
   Create a `.env` file in the project root:
   ```ini
   YC_FOLDER_ID=b1g...
   YC_API_KEY=AQV...
   ```

## ▶️ Usage Examples

### 1. Basic Text Generation (Comparison)
Runs a simple prompt through both Native API and OpenAI SDK.
```bash
uv run main.py
```

### 2. Async Batch Processing (High Performance)
Process multiple prompts concurrently using `asyncio` and `Semaphores`. Ideal for generating datasets.
```bash
uv run examples/async_batch.py
```

### 3. Semantic Search (Embeddings)
Demonstrates vector search. Finds the most relevant document based on meaning, not just keywords.
```bash
uv run examples/semantic_search.py
```

### 4. Function Calling (Tools)
Shows how the model can "call" Python functions (e.g., getting weather data) to answer user questions.
```bash
uv run examples/tools_demo.py
```

## 📂 Project Structure

```text
.
├── main.py                 # Entry point for basic demo
├── pyproject.toml          # Dependencies (requests, openai, numpy)
├── src/
│   ├── config.py           # Centralized configuration (Singleton)
│   └── clients/            # API Client implementations
└── examples/
    ├── basic_usage.py      # "Hello World" example
    ├── async_batch.py      # Async processing example
    ├── semantic_search.py  # Embeddings & Cosine Similarity
    └── tools_demo.py       # Function Calling (Tools) example
```
```
