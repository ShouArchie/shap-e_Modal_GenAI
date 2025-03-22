"""
LocalShapE - Text-to-3D API using Shap-E
----------------------------------------
A Flask-based API for generating 3D models from text prompts
using OpenAI's Shap-E model, optimized for GPU acceleration.

This is the entry point for the GenAIGenesis AI Compute Platform.
"""

import os
import sys
import torch
import argparse
import time
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, send_file
import json
import tempfile
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the local shap-e directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'shap-e'))
logger.info(f"Added {os.path.join(os.getcwd(), 'shap-e')} to Python path")

# Set environment variables for HuggingFace cache
os.environ['PYTORCH_HF_CACHE_HOME'] = os.path.join(os.getcwd(), 'shap_e_model_cache')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'shap_e_model_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'shap_e_model_cache')

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)
logger.info(f"Output directory: {os.path.abspath('outputs')}")

# Create cache directory if it doesn't exist
os.makedirs('shap_e_model_cache', exist_ok=True)
logger.info(f"Cache directory: {os.path.abspath('shap_e_model_cache')}")

app = Flask(__name__)

# Reference to loaded models (will be initialized on first request)
models = {
    'xm': None,
    'model': None,
    'diffusion': None,
    'device': None
}

def load_models_if_needed():
    """Load models if they haven't been loaded yet."""
    if models['model'] is not None:
        return
    
    logger.info("Loading Shap-E models...")
    
    # Import Shap-E modules
    from shap_e.models.download import load_model, load_config
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (GPU not available)")
    
    # Load models
    logger.info(f"Loading models to {device}...")
    models['xm'] = load_model('transmitter', device=device)
    models['model'] = load_model('text300M', device=device)
    models['diffusion'] = diffusion_from_config(load_config('diffusion'))
    models['device'] = device
    
    # Report memory usage after model loading
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        memory_reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.info(f"GPU memory after loading models - Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")
    
    logger.info("Models loaded successfully.")

def generate_3d_model(
    prompt,
    output_path,
    guidance_scale=15.0,
    num_steps=24,
    seed=None
):
    """Generate a 3D model from a text prompt using Shap-E."""
    try:
        start_time = time.time()
        
        # Ensure models are loaded
        load_models_if_needed()
        
        # Import required modules
        from shap_e.diffusion.sample import sample_latents
        from shap_e.util.collections import AttrDict
        from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
        import trimesh
        
        device = models['device']
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            logger.info(f"Random seed set to: {seed}")
        
        logger.info(f"Generating 3D model for prompt: '{prompt}'")
        logger.info(f"Parameters: guidance_scale={guidance_scale}, num_steps={num_steps}")
        
        # Generate latents
        latents = sample_latents(
            batch_size=1,
            model=models['model'],
            diffusion=models['diffusion'],
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=num_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        # Create mesh
        logger.info("Creating mesh...")
        
        # This is our own implementation of decode_latent_mesh function from notebooks.py
        # Create a simple camera setup (2x2 resolution is enough since we just need the mesh)
        @torch.no_grad()
        def create_pan_cameras(size, device):
            origins = []
            xs = []
            ys = []
            zs = []
            for theta in np.linspace(0, 2 * np.pi, num=20):
                z = np.array([np.sin(theta), np.cos(theta), -0.5])
                z /= np.sqrt(np.sum(z**2))
                origin = -z * 4
                x = np.array([np.cos(theta), -np.sin(theta), 0.0])
                y = np.cross(z, x)
                origins.append(origin)
                xs.append(x)
                ys.append(y)
                zs.append(z)
            return DifferentiableCameraBatch(
                shape=(1, len(xs)),
                flat_camera=DifferentiableProjectiveCamera(
                    origin=torch.from_numpy(np.stack(origins, axis=0)).float().to(device),
                    x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
                    y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
                    z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
                    width=size,
                    height=size,
                    x_fov=0.7,
                    y_fov=0.7,
                ),
            )
        
        # Use the renderer directly to get the mesh
        with torch.no_grad():
            cameras = create_pan_cameras(2, latents[0].device)  # lowest resolution possible
            
            # Get bottleneck params (needed to reconstruct the mesh)
            if hasattr(models['xm'], 'encoder'):
                params = models['xm'].encoder.bottleneck_to_params(latents[0][None])
            else:
                params = models['xm'].bottleneck_to_params(latents[0][None])
            
            # Render views with STF mode to get the mesh
            decoded = models['xm'].renderer.render_views(
                AttrDict(cameras=cameras),
                params=params,
                options=AttrDict(rendering_mode="stf", render_with_direction=False),
            )
            
            # Extract the mesh
            raw_mesh = decoded.raw_meshes[0]
            tri_mesh = raw_mesh.tri_mesh()
            
            # Convert to vertices and faces for trimesh
            # Get vertices and faces
            verts = tri_mesh.verts
            faces = tri_mesh.faces
            
            # Convert to numpy
            if isinstance(verts, torch.Tensor):
                verts = verts.cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()
                
            # Create a trimesh object
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Save as STL file
        logger.info(f"Saving mesh to {output_path}...")
        mesh.export(output_path)
        
        end_time = time.time()
        logger.info(f"Done! Total generation time: {end_time - start_time:.2f} seconds")
        logger.info(f"File saved to: {os.path.abspath(output_path)}")
        
        # Report final memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**2)
            memory_reserved = torch.cuda.memory_reserved() / (1024**2)
            logger.info(f"Final GPU memory - Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")
            
            # Clean up CUDA memory
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")
        
        return True, output_path
    
    except Exception as e:
        logger.error(f"Error generating 3D model: {e}")
        logger.error(traceback.format_exc())
        return False, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "pytorch_version": torch.__version__,
    })

@app.route('/generate', methods=['POST'])
def generate_model():
    """Generate a 3D model from a text prompt."""
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract parameters
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        guidance_scale = float(data.get('guidance_scale', 15.0))
        num_steps = int(data.get('num_steps', 24))
        seed = data.get('seed')
        if seed is not None:
            seed = int(seed)
        
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Generate the 3D model
        success, result = generate_3d_model(
            prompt=prompt,
            output_path=output_path,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed
        )
        
        if success:
            # Return the STL file
            return send_file(output_path, as_attachment=True, download_name=f"{prompt.replace(' ', '_')}.stl", mimetype='application/octet-stream')
        else:
            return jsonify({"error": f"Failed to generate model: {result}"}), 500
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Simple info page."""
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
    else:
        gpu_info = "CPU (No GPU detected)"
    
    html = f"""
    <html>
        <head>
            <title>Shap-E Text-to-3D API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                .gpu-info {{ background: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Shap-E Text-to-3D API</h1>
            <p>This API generates 3D models from text prompts using OpenAI's Shap-E model.</p>
            
            <div class="gpu-info">
                <strong>Hardware:</strong> {gpu_info}<br>
                <strong>PyTorch:</strong> {torch.__version__}<br>
            </div>
            
            <h2>API Usage</h2>
            <p>Send a POST request to <code>/generate</code> with a JSON body:</p>
            <pre>
{{
  "prompt": "a detailed sculpture of a dragon",
  "guidance_scale": 15.0,
  "num_steps": 24,
  "seed": null
}}
            </pre>
            
            <p>The API will return an STL file that you can open in any 3D software.</p>
            
            <h2>Parameters</h2>
            <ul>
                <li><strong>prompt</strong> (required): Text description of the 3D model</li>
                <li><strong>guidance_scale</strong> (optional): How closely to follow the prompt (default: 15.0)</li>
                <li><strong>num_steps</strong> (optional): Number of diffusion steps (default: 24, higher = better quality but slower)</li>
                <li><strong>seed</strong> (optional): Random seed for reproducibility</li>
            </ul>
            
            <h2>For GenAIGenesis Judges</h2>
            <p>This is a Text-to-3D generation API using OpenAI's Shap-E model, but running locally on the GenAIGenesis compute platform.</p>
            <p>It allows you to generate 3D models from text descriptions, which can be downloaded as STL files.</p>
            <p>You can test the API using the provided form below or by using the <code>test_client.py</code> script in the repository.</p>
            
            <h3>Test the API</h3>
            <form id="generation-form" style="background: #f4f4f4; padding: 20px; border-radius: 5px;">
                <div style="margin-bottom: 10px;">
                    <label for="prompt">Prompt:</label>
                    <input type="text" id="prompt" name="prompt" style="width: 100%; padding: 8px;" 
                           value="a detailed sculpture of a dragon" required>
                </div>
                <div style="margin-bottom: 10px;">
                    <label for="guidance_scale">Guidance Scale:</label>
                    <input type="number" id="guidance_scale" name="guidance_scale" min="1" max="30" step="0.5" 
                           value="15.0" style="width: 100px; padding: 8px;">
                </div>
                <div style="margin-bottom: 10px;">
                    <label for="num_steps">Steps:</label>
                    <input type="number" id="num_steps" name="num_steps" min="1" max="64" step="1" 
                           value="24" style="width: 100px; padding: 8px;">
                </div>
                <div style="margin-bottom: 10px;">
                    <label for="seed">Seed (optional):</label>
                    <input type="number" id="seed" name="seed" style="width: 100px; padding: 8px;">
                </div>
                <button type="submit" style="background: #4CAF50; color: white; padding: 10px 15px; border: none; 
                                             border-radius: 4px; cursor: pointer;">
                    Generate 3D Model
                </button>
            </form>
            
            <script>
                document.getElementById('generation-form').addEventListener('submit', function(e) {{
                    e.preventDefault();
                    
                    const prompt = document.getElementById('prompt').value;
                    const guidance_scale = parseFloat(document.getElementById('guidance_scale').value);
                    const num_steps = parseInt(document.getElementById('num_steps').value);
                    const seedInput = document.getElementById('seed');
                    
                    let payload = {{
                        prompt: prompt,
                        guidance_scale: guidance_scale,
                        num_steps: num_steps
                    }};
                    
                    if (seedInput.value) {{
                        payload.seed = parseInt(seedInput.value);
                    }}
                    
                    // Create a direct download
                    const form = document.createElement('form');
                    form.method = 'POST';
                    form.action = '/generate';
                    form.target = '_blank';
                    
                    const hiddenField = document.createElement('input');
                    hiddenField.type = 'hidden';
                    hiddenField.name = 'payload';
                    hiddenField.value = JSON.stringify(payload);
                    
                    form.appendChild(hiddenField);
                    document.body.appendChild(form);
                    form.submit();
                    document.body.removeChild(form);
                }});
            </script>
        </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    # Set default port to 8000 for the GenAIGenesis platform
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False) 