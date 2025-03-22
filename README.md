# LocalShapE - Text-to-3D API using Shap-E

This project provides a Flask-based API for generating 3D models from text prompts using OpenAI's Shap-E model. It's designed to run locally on your own hardware, specifically optimized for NVIDIA GPUs.

## Features

- **Text-to-3D generation**: Create 3D models from textual descriptions
- **REST API**: Simple HTTP API for easy integration with other applications
- **GPU Acceleration**: Utilizes your NVIDIA GPU for faster generation
- **Configurable parameters**: Control the quality and characteristics of the generated models
- **STL export**: Download models as STL files that can be used in any 3D software

## Requirements

- CUDA-compatible NVIDIA GPU (tested on NVIDIA 3050/3060)
- Python 3.8+
- PyTorch with CUDA support
- Flask

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/LocalShapE.git
cd LocalShapE
```

### 2. Clone the Shap-E repository inside this project

```bash
git clone https://github.com/openai/shap-e.git
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e ./shap-e
pip install flask numpy trimesh matplotlib
```

### 4. Create cache and output directories

```bash
mkdir -p shap_e_model_cache outputs
```

## Usage

### Running the API server

```bash
python api.py --port 9200
```

The API will be available at `http://localhost:9200/`.

### API Endpoints

- `GET /`: Simple HTML interface with API documentation
- `GET /health`: Health check endpoint
- `POST /generate`: Generate a 3D model from a text prompt

### Generating a 3D model

Send a POST request to the `/generate` endpoint with a JSON payload:

```bash
curl -X POST http://localhost:9200/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a detailed sculpture of a dragon", "guidance_scale": 15.0, "num_steps": 24}' \
  --output dragon.stl
```

#### Parameters

- `prompt` (required): Text description of what to generate
- `guidance_scale` (optional): How closely to follow the prompt (default: 15.0)
- `num_steps` (optional): Number of diffusion steps (default: 24, higher = better quality but slower)
- `seed` (optional): Random seed for reproducibility

### Using the standalone script

If you prefer to use the command-line tool without the API:

```bash
python generate_3d.py --prompt "a detailed sculpture of a dragon" --output outputs/dragon.stl
```

## Project Structure

- `api.py`: Flask API server
- `generate_3d.py`: Standalone command-line tool
- `shap-e/`: The Shap-E repository (to be cloned separately)
- `shap_e_model_cache/`: Cache directory for downloaded models
- `outputs/`: Directory for output STL files

## Hosting on GitHub

Since this project depends on the Shap-E repository which is already on GitHub, the structure works as follows:

1. This repository contains only the API code and doesn't include Shap-E itself
2. Users need to clone Shap-E separately into this project
3. The `.gitignore` file is set up to exclude the Shap-E directory to avoid duplicating it

## License

This project is released under the MIT License. See the LICENSE file for details.

The Shap-E model is developed by OpenAI and has its own license, which can be found in the [Shap-E repository](https://github.com/openai/shap-e).

## Acknowledgements

- [OpenAI Shap-E](https://github.com/openai/shap-e) - For the amazing Text-to-3D model
- [PyTorch](https://pytorch.org/) - For the deep learning framework
- [Flask](https://flask.palletsprojects.com/) - For the web framework 