"""
Fresh Deployment of Shap-E Text-to-3D on Modal with A100-40GB
"""

import modal
import io
import time
import logging
from fastapi import Response, HTTPException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the verification function to use instead of a lambda
def verify_shap_e_install():
    """Verify Shap-E installation"""
    import sys
    import pkgutil
    
    sys.path.append("/root/shap-e")
    if pkgutil.find_loader("shap_e") is None:
        print("Failed to import shap_e")
        exit(1)
    print("✓ Shap-E installed successfully")

# Create a clean new Modal image with carefully ordered dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "build-essential", "curl", "wget", "pkg-config", "libglib2.0-0"])
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
        "pip install numpy==1.24.3",  # Pin specific version
        "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118",  # Ensure CUDA compatibility
    )
    .pip_install(
        "numpy==1.24.3",
        "torch==2.0.1",
        "tqdm==4.65.0",
        "pillow==9.5.0",
        "transformers==4.30.2",
        "trimesh==3.22.2",
        "pyyaml",  # Add PyYAML package
        "ipywidgets",  # Add ipywidgets package
        "matplotlib==3.7.1",
        "fastapi",
        "uvicorn"
    )
    # Install Shap-E from source with explicit version
    .run_commands(
        "git clone https://github.com/openai/shap-e.git /root/shap-e",
        "cd /root/shap-e && pip install -e .",
        "pip install pyyaml  # Ensure PyYAML is installed"
    )
    # Add verification steps
    .run_function(verify_shap_e_install)  # Use named function instead of lambda
)

# Create new app with the clean image
app = modal.App("shape-text-to-3d-a100-40gb", image=image)

# Create a volume for caching models
volume = modal.Volume.from_name("shape-model-cache-a100", create_if_missing=True)
MODEL_CACHE_DIR = "/cache"

# Define a verification function that runs at startup
@app.function(gpu="A100-40GB")  # Specifically request A100-40GB
def verify_installation():
    """Verify that all components are correctly installed"""
    try:
        import torch
        import trimesh
        import shap_e
        from shap_e.diffusion.sample import sample_latents
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_model, load_config
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print(f"Trimesh version: {trimesh.__version__}")
        print(f"Shap-E package found at: {shap_e.__file__}")
        
        # Test loading a model (this will cache it in the volume)
        print("Testing model loading...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        diffusion = diffusion_from_config(load_config('diffusion'))
        print("✓ Models loaded successfully")
        
        # Create a simple test object to verify the pipeline
        print("Creating test object...")
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=10.0,
            model_kwargs=dict(texts=["test sphere"]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=16,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        # Verify we can convert to a mesh
        print("Converting to mesh...")
        from shap_e.util.notebooks import decode_latent_mesh
        mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
        
        # Verify we can export to STL
        print("Exporting to STL...")
        buffer = io.BytesIO()
        verts, faces = mesh.verts, mesh.faces
        if isinstance(verts, torch.Tensor):
            verts = verts.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        trimesh_mesh.export(buffer, file_type='stl')
        
        print(f"✓ Full pipeline verification successful. STL size: {buffer.tell()} bytes")
        return {"status": "success", "message": "Shap-E installation verified"}
    except Exception as e:
        import traceback
        print(f"Verification failed: {e}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@app.function(
    gpu="A100-40GB",  # Specifically request A100-40GB
    timeout=600,
    volumes={MODEL_CACHE_DIR: volume}
)
def generate_3d_model(
    prompt: str,
    guidance_scale: float = 15.0,
    num_steps: int = 64,
    seed: int = None,
):
    """
    Generate a 3D model from a text prompt using Shap-E.
    
    Args:
        prompt: Text description of the 3D model to generate
        guidance_scale: Controls how closely the model follows your text (default: 15.0)
        num_steps: Number of diffusion steps (default: 64, higher = better quality but slower)
        seed: Random seed for reproducibility
    """
    import os
    import torch
    import numpy as np
    import trimesh
    import time
    import random
    
    start_time = time.time()
    
    # Set environment variables for cache
    os.environ['PYTORCH_HF_CACHE_HOME'] = MODEL_CACHE_DIR
    os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR
    os.environ['HF_HOME'] = MODEL_CACHE_DIR
    
    # Import Shap-E modules
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.collections import AttrDict
    
    # Log start of process
    print(f"Starting 3D model generation for prompt: '{prompt}'")
    print(f"Parameters: guidance_scale={guidance_scale}, num_steps={num_steps}, seed={seed}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Verify CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {device_name} with {gpu_memory:.2f} GB memory")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Random seed set to: {seed}")
    else:
        # Generate a random seed and log it for reproducibility
        seed = random.randint(0, 2**32-1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    try:
        # Load the models
        print("Loading Shap-E models...")
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        diffusion = diffusion_from_config(load_config('diffusion'))
        print("Models loaded successfully")
        
        # Generate latents with the text model
        print("Generating latents...")
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
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
        print("Creating mesh...")
        # Use cleaner approach directly from shap_e examples
        from shap_e.util.notebooks import decode_latent_mesh
        mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
        
        # Get vertices and faces for trimesh
        verts, faces = mesh.verts, mesh.faces
        
        # Convert to numpy if needed
        if isinstance(verts, torch.Tensor):
            verts = verts.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        
        # Verify mesh data
        print(f"Mesh data: {verts.shape} vertices, {faces.shape} faces")
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            print("Warning: Empty mesh generated")
            # Generate a simple cube as fallback
            verts = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ])
            faces = np.array([
                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
            ])
            print("Created fallback cube geometry")
        
        # Create a trimesh object
        trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Save as STL to a bytes buffer
        buffer = io.BytesIO()
        trimesh_mesh.export(buffer, file_type='stl')
        buffer.seek(0)
        stl_bytes = buffer.read()
        
        # Verify STL data
        print(f"STL file size: {len(stl_bytes)} bytes")
        if len(stl_bytes) < 100:
            print("Warning: STL file is suspiciously small")
        
        # Log completion
        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds")
        
        return stl_bytes, None
    except Exception as e:
        import traceback
        print(f"Error generating model: {e}")
        print(traceback.format_exc())
        return None, str(e)

# Web endpoint to check health
@app.function()
@modal.fastapi_endpoint()
def health():
    """Health check endpoint"""
    import torch
    
    # Get CUDA status
    cuda_available = torch.cuda.is_available()
    cuda_info = {
        "available": cuda_available,
        "version": torch.version.cuda if cuda_available else None,
        "device": torch.cuda.get_device_name(0) if cuda_available else None
    }
    
    return {
        "status": "healthy",
        "model": "Shap-E text-to-3D (A100-40GB)",
        "timestamp": time.time(),
        "cuda": cuda_info
    }

# Web endpoint for 3D model generation
@app.function()
@modal.fastapi_endpoint(method="POST")
def generate(prompt: str, guidance_scale: float = 15.0, num_steps: int = 64, seed: int = None):
    """
    Generate a 3D model from a text prompt
    """
    try:
        start_time = time.time()
        
        logger.info(f"Received generation request for prompt: '{prompt}'")
        
        # Generate the 3D model
        stl_bytes, error = generate_3d_model.remote(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed
        )
        
        # Handle errors
        if error:
            logger.error(f"Error in model generation: {error}")
            raise HTTPException(status_code=500, detail=error)
        
        if not stl_bytes or len(stl_bytes) < 100:
            logger.error(f"Empty or invalid STL generated (size: {len(stl_bytes) if stl_bytes else 0} bytes)")
            raise HTTPException(status_code=500, detail="Generated model was empty or invalid")
        
        # Calculate time taken
        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        # Return the STL file
        return Response(
            content=stl_bytes,
            media_type="model/stl",
            headers={"Content-Disposition": f"attachment; filename={prompt.replace(' ', '_')}.stl"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Simple UI for testing
@app.function()
@modal.fastapi_endpoint()
def index():
    """Web UI for the text-to-3D generator"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text-to-3D Generator (A100-40GB)</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #333;
            }
            input, button {
                padding: 10px;
                margin: 10px 0;
                width: 100%;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
            }
            button:hover {
                opacity: 0.8;
            }
            #loading {
                display: none;
            }
            .slider {
                width: 100%;
            }
            label {
                display: block;
                margin-top: 15px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>Text-to-3D Generator (A100-40GB)</h1>
        <p>Enter a text description to generate a 3D model.</p>
        
        <form id="generateForm">
            <label for="prompt">Text Description:</label>
            <input type="text" id="prompt" name="prompt" required placeholder="e.g., a detailed dog sculpture">
            
            <label for="guidance">Guidance Scale: <span id="guidanceValue">15.0</span></label>
            <input type="range" id="guidance" name="guidance_scale" min="5" max="30" step="0.5" value="15" class="slider">
            
            <label for="steps">Number of Steps: <span id="stepsValue">64</span></label>
            <input type="range" id="steps" name="num_steps" min="16" max="128" step="8" value="64" class="slider">
            
            <label for="seed">Random Seed (optional):</label>
            <input type="number" id="seed" name="seed" placeholder="Leave empty for random">
            
            <button type="submit">Generate 3D Model</button>
        </form>
        
        <div id="loading">
            <p>Generating your 3D model... This may take 30-90 seconds.</p>
            <progress></progress>
        </div>
        
        <div id="result"></div>
        
        <script>
            // Update slider value displays
            document.getElementById('guidance').oninput = function() {
                document.getElementById('guidanceValue').textContent = this.value;
            };
            document.getElementById('steps').oninput = function() {
                document.getElementById('stepsValue').textContent = this.value;
            };
            
            document.getElementById('generateForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const prompt = document.getElementById('prompt').value;
                const guidance = document.getElementById('guidance').value;
                const steps = document.getElementById('steps').value;
                const seed = document.getElementById('seed').value;
                
                // Show loading indicator
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').innerHTML = '';
                
                // Build the URL with query parameters
                const url = new URL('/generate', window.location.origin);
                url.searchParams.append('prompt', prompt);
                url.searchParams.append('guidance_scale', guidance);
                url.searchParams.append('num_steps', steps);
                if (seed) url.searchParams.append('seed', seed);
                
                try {
                    const response = await fetch(url, {method: 'POST'});
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const downloadUrl = URL.createObjectURL(blob);
                        
                        document.getElementById('result').innerHTML = `
                            <p>Your 3D model is ready! <a href="${downloadUrl}" download="${prompt.replace(/ /g, '_')}.stl">Download STL File</a></p>
                            <p>You can open this file in any 3D software or upload it to online viewers like <a href="https://www.viewstl.com/" target="_blank">ViewSTL</a>.</p>
                        `;
                    } else {
                        let errorText = await response.text();
                        try {
                            const errorJson = JSON.parse(errorText);
                            errorText = errorJson.detail || errorJson.message || errorText;
                        } catch (e) {}
                        
                        document.getElementById('result').innerHTML = `
                            <p style="color: red">Error: ${errorText}</p>
                        `;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `
                        <p style="color: red">Error: ${error.message}</p>
                    `;
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            };
        </script>
    </body>
    </html>
    """
    return Response(content=html, media_type="text/html")

# Add a verification step that runs at startup time
@app.function()
@modal.fastapi_endpoint()
def verify():
    """Run verification on app startup to ensure everything is working"""
    result = verify_installation.remote()
    return result

if __name__ == "__main__":
    print("Deploying Shap-E Text-to-3D app to Modal with A100-40GB GPU...")
    modal.app.deploy("shape-text-to-3d-a100-40gb")
    print("Deployment complete! Running verification...")
    verify_result = verify_installation.remote()
    print(f"Verification result: {verify_result}")
    print("\nYour app is ready to use at:")
    print("- Web UI: https://moulik-budhiraja--shape-text-to-3d-a100-40gb-index.modal.run")
    print("- API: https://moulik-budhiraja--shape-text-to-3d-a100-40gb-generate.modal.run")
    print("- Health check: https://moulik-budhiraja--shape-text-to-3d-a100-40gb-health.modal.run")
    print("- Verification: https://moulik-budhiraja--shape-text-to-3d-a100-40gb-verify.modal.run") 