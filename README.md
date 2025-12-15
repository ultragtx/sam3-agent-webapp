# SAM3 Agent Web Application

A modern web application for interacting with SAM3 (Segment Anything Model 3) through an intelligent agent powered by Multimodal Large Language Models (MLLMs).

## Features

- 🤖 **Intelligent Agent**: Iterative reasoning with MLLM to segment complex queries
- 🎨 **Interactive Visualization**: Real-time display of all intermediate steps, masks, and model outputs
- ⚡ **High Performance**: SAM3 model kept in GPU memory for fast inference
- 🔧 **Configurable**: Flexible MLLM parameters configurable via UI and environment variables
- 🖼️ **Rich UI**: Modern React-based interface with drag-and-drop image upload

## Architecture

```
sam3-agent-webapp/
├── frontend/          # React + Vite + TailwindCSS
├── backend/           # Flask + SAM3 + OpenAI SDK
├── ref-code/          # Reference implementation
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (for SAM3)
- Access to Qwen3-VL or compatible MLLM API

## Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
cd /nfshome/xinrong/pubNAS3/projects/sam3-agent-webapp

# Copy environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SAM3 model checkpoint (if needed)
# Update SAM3_MODEL_PATH in .env
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Development Mode

```bash
# Terminal 1: Start backend
cd backend
source venv/bin/activate
python app.py

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Or use the root package.json:

```bash
npm install
npm start
```

### Access the Application

- Frontend: http://localhost:5173
- Backend API: http://localhost:5000

## Configuration

### MLLM Settings

Configure in `.env` or update via the web UI:

- **API Base URL**: Your Qwen3-VL endpoint
- **API Key**: Authentication key
- **Model Name**: Model identifier
- **Max Tokens**: Maximum generation length

### SAM3 Settings

- **Model Path**: Path to SAM3 checkpoint
- **Device**: cuda/cpu
- **Model Size**: large/base/small

## Usage

1. **Upload Image**: Drag and drop or click to upload an image
2. **Enter Query**: Type your segmentation query (e.g., "person holding a phone")
3. **Configure Parameters**: Adjust MLLM settings if needed
4. **Run Agent**: Click "Start Agent" to begin the iterative segmentation process
5. **View Results**: See all intermediate steps, MLLM reasoning, and final masks

## API Documentation

### Endpoints

#### `POST /api/upload`
Upload an image file

#### `POST /api/agent/run`
Run the SAM3 agent on an image with a text query

```json
{
  "image_path": "path/to/image.jpg",
  "text_prompt": "person holding a phone",
  "mllm_config": {
    "api_base": "http://localhost:8000/v1",
    "api_key": "key",
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "max_tokens": 4096
  }
}
```

#### `POST /api/sam3/segment`
Direct SAM3 segmentation with text prompt

#### `GET /api/outputs/<filename>`
Retrieve output images and results

## Project Structure

```
backend/
├── app.py                 # Flask application
├── agent/
│   ├── core.py           # Agent core logic
│   ├── client_llm.py     # MLLM client
│   └── client_sam3.py    # SAM3 client
├── utils/
│   ├── visualization.py  # Visualization utilities
│   └── helpers.py        # Helper functions
├── models/
│   └── sam3_loader.py    # SAM3 model loader
└── requirements.txt

frontend/
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── ImageUpload.tsx
│   │   ├── ConfigPanel.tsx
│   │   ├── AgentViewer.tsx
│   │   └── MaskVisualization.tsx
│   ├── hooks/
│   │   └── useAgent.ts
│   ├── services/
│   │   └── api.ts
│   └── types/
│       └── index.ts
├── package.json
└── vite.config.ts
```

## Development

### Backend Development

```bash
cd backend
source venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/
```

### Frontend Development

```bash
cd frontend
npm run dev
npm run lint
npm run type-check
```

## Troubleshooting

### SAM3 Model Loading Issues

- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check model path is correct
- Verify sufficient GPU memory

### MLLM Connection Issues

- Test API endpoint: `curl $MLLM_API_BASE/models`
- Verify API key is valid
- Check network connectivity

## License

Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

## Citation

If you use SAM3 Agent in your research, please cite:

```bibtex
@article{ravi2024sam3,
  title={SAM 3: Segment Anything in High Quality},
  author={Ravi, Nikhila and others},
  journal={arXiv preprint},
  year={2024}
}
```
