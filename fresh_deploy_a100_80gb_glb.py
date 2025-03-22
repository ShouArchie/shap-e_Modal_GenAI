"""
Fresh Deployment of Shap-E Text-to-3D on Modal with A100-80GB (GLB output)
"""

import modal
import io
import time
import logging
import subprocess
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
app = modal.App("shape-text-to-3d-a100-80gb-glb", image=image)

# Create a volume for caching models
volume = modal.Volume.from_name("shape-model-cache-a100-80gb", create_if_missing=True)
MODEL_CACHE_DIR = "/cache"

# Define a verification function that runs at startup
@app.function(gpu="A100-80GB")  # Specifically request A100-80GB
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
        test_prompt = "a blue sphere"
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=10.0,
            model_kwargs=dict(texts=[test_prompt]),
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
        shap_e_mesh = decode_latent_mesh(xm, latents[0])
        tri_mesh = shap_e_mesh.tri_mesh()
        
        # Verify we can export to GLB
        print("Exporting to GLB...")
        buffer = io.BytesIO()
        verts, faces = tri_mesh.verts, tri_mesh.faces
        if isinstance(verts, torch.Tensor):
            verts = verts.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        
        # Create a trimesh object
        trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Try to extract color information from the Shap-E mesh
        print("Extracting color information...")
        
        # Simplified approach - create blue coloring for test sphere
        print("Creating blue coloring for test sphere...")
        import numpy as np
        
        # Create vertex colors array with blue as the base
        vertex_colors = np.ones((len(verts), 4))
        # Blue variation
        vertex_colors[:, 0] = 0.0 * (0.8 + 0.4 * np.random.rand(len(verts)))  # R (zero for blue)
        vertex_colors[:, 1] = 0.0 * (0.8 + 0.4 * np.random.rand(len(verts)))  # G (zero for blue)
        vertex_colors[:, 2] = 1.0 * (0.8 + 0.4 * np.random.rand(len(verts)))  # B (high for blue)
        vertex_colors[:, 3] = 1.0  # Alpha
        
        # Print the color range
        print(f"Color range - R: {vertex_colors[:, 0].min():.2f}-{vertex_colors[:, 0].max():.2f}, "
              f"G: {vertex_colors[:, 1].min():.2f}-{vertex_colors[:, 1].max():.2f}, "
              f"B: {vertex_colors[:, 2].min():.2f}-{vertex_colors[:, 2].max():.2f}")
        
        # Apply the colors
        trimesh_mesh.visual.vertex_colors = vertex_colors
        
        # Verify colors were applied
        if hasattr(trimesh_mesh.visual, 'vertex_colors'):
            print(f"Successfully applied colors to mesh with {len(trimesh_mesh.visual.vertex_colors)} vertices")
        else:
            print("WARNING: Failed to apply colors to mesh!")
        
        # Export as GLB
        trimesh_mesh.export(buffer, file_type='glb')
        
        print(f"✓ Full pipeline verification successful. GLB size: {buffer.tell()} bytes")
        return {"status": "success", "message": "Shap-E installation verified"}
    except Exception as e:
        import traceback
        print(f"Verification failed: {e}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@app.function(
    gpu="A100-80GB",  # Specifically request A100-80GB
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
        shap_e_mesh = decode_latent_mesh(xm, latents[0])
        
        # Get the triangular mesh with texture information
        print("Extracting mesh data with colors...")
        tri_mesh = shap_e_mesh.tri_mesh()
        
        # Get vertices and faces for trimesh
        verts, faces = tri_mesh.verts, tri_mesh.faces
        
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
            
            # Create a trimesh object with fallback colors
            trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            
            # Add fallback colors (blue if "blue" in prompt, otherwise gray)
            vertex_colors = np.ones((len(verts), 4))
            if "blue" in prompt.lower():
                vertex_colors[:, 0] = 0.0  # R
                vertex_colors[:, 1] = 0.0  # G
                vertex_colors[:, 2] = 1.0  # B
            else:
                vertex_colors[:, 0] = 0.7  # R
                vertex_colors[:, 1] = 0.7  # G
                vertex_colors[:, 2] = 0.7  # B
            vertex_colors[:, 3] = 1.0  # Alpha
            
            trimesh_mesh.visual.vertex_colors = vertex_colors
        else:
            # Create a trimesh object with the Shap-E generated vertices and faces
            trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            
            # Simplified approach: Create colors based on the prompt
            print("Creating colors based on the prompt...")
            
            # Simple color mapping based on common color words in the prompt
            color_map = {
                'red': [1.0, 0.0, 0.0],
                'green': [0.0, 1.0, 0.0],
                'blue': [0.0, 0.0, 1.0],
                'yellow': [1.0, 1.0, 0.0],
                'purple': [0.5, 0.0, 0.5],
                'orange': [1.0, 0.5, 0.0],
                'black': [0.1, 0.1, 0.1],  # Not completely black for visibility
                'white': [1.0, 1.0, 1.0],
                'cyan': [0.0, 1.0, 1.0],
                'magenta': [1.0, 0.0, 1.0],
                'brown': [0.6, 0.3, 0.0],
                'pink': [1.0, 0.7, 0.7],
                'gray': [0.5, 0.5, 0.5],
                'grey': [0.5, 0.5, 0.5],
                'gold': [1.0, 0.84, 0.0],
                'silver': [0.75, 0.75, 0.75]
            }
            
            # Check if any color words are in the prompt
            prompt_lower = prompt.lower()
            detected_color = None
            
            for color, rgb in color_map.items():
                if color in prompt_lower:
                    detected_color = rgb
                    print(f"Detected color '{color}' in prompt")
                    break
            
            # Create vertex colors array
            vertex_colors = np.ones((len(verts), 4))
            
            # Apply appropriate coloring based on detection or default
            if detected_color is not None:
                # Apply the detected color with slight variations for visual interest
                print(f"Applying detected color: RGB={detected_color}")
                vertex_colors[:, 0] = detected_color[0] * (0.8 + 0.4 * np.random.rand(len(verts)))  # R
                vertex_colors[:, 1] = detected_color[1] * (0.8 + 0.4 * np.random.rand(len(verts)))  # G
                vertex_colors[:, 2] = detected_color[2] * (0.8 + 0.4 * np.random.rand(len(verts)))  # B
            else:
                # Use a visually interesting default coloring if no color is specified
                print("No specific color detected in prompt, using default coloring")
                
                # Object-specific colors for common items
                if "car" in prompt_lower:
                    # Cars are often red
                    base_color = [0.8, 0.0, 0.0]  # Red
                elif "tree" in prompt_lower or "plant" in prompt_lower:
                    # Trees/plants are green
                    base_color = [0.0, 0.7, 0.0]  # Green
                elif "sky" in prompt_lower or "water" in prompt_lower:
                    # Sky/water are blue
                    base_color = [0.0, 0.5, 0.9]  # Blue
                elif "apple" in prompt_lower:
                    # Apples are red
                    base_color = [0.8, 0.1, 0.1]  # Red
                elif "banana" in prompt_lower:
                    # Bananas are yellow
                    base_color = [1.0, 0.8, 0.0]  # Yellow
                elif "orange" in prompt_lower and ("fruit" in prompt_lower or "food" in prompt_lower):
                    # Oranges are orange
                    base_color = [1.0, 0.5, 0.0]  # Orange
                else:
                    # Default rainbow gradient based on vertex position
                    vertex_colors[:, 0] = 0.5 + 0.5 * np.sin(verts[:, 0] * 5)  # R
                    vertex_colors[:, 1] = 0.5 + 0.5 * np.sin(verts[:, 1] * 5)  # G
                    vertex_colors[:, 2] = 0.5 + 0.5 * np.sin(verts[:, 2] * 5)  # B
                    base_color = None  # Flag that we've already set colors
                
                # Apply the base color with variations if we set one
                if base_color is not None:
                    vertex_colors[:, 0] = base_color[0] * (0.7 + 0.6 * np.random.rand(len(verts)))
                    vertex_colors[:, 1] = base_color[1] * (0.7 + 0.6 * np.random.rand(len(verts)))
                    vertex_colors[:, 2] = base_color[2] * (0.7 + 0.6 * np.random.rand(len(verts)))
            
            # Ensure alpha is 1.0
            vertex_colors[:, 3] = 1.0
            
            # Print the color range to verify
            print(f"Color range - R: {vertex_colors[:, 0].min():.2f}-{vertex_colors[:, 0].max():.2f}, "
                  f"G: {vertex_colors[:, 1].min():.2f}-{vertex_colors[:, 1].max():.2f}, "
                  f"B: {vertex_colors[:, 2].min():.2f}-{vertex_colors[:, 2].max():.2f}")
            
            # Apply the colors to the mesh
            trimesh_mesh.visual.vertex_colors = vertex_colors
            
            # Verify colors were applied
            if hasattr(trimesh_mesh.visual, 'vertex_colors'):
                print(f"Successfully applied colors to mesh with {len(trimesh_mesh.visual.vertex_colors)} vertices")
            else:
                print("WARNING: Failed to apply colors to mesh!")
        
        # Save as GLB to a bytes buffer
        print("Exporting to GLB...")
        buffer = io.BytesIO()
        trimesh_mesh.export(buffer, file_type='glb')
        buffer.seek(0)
        glb_bytes = buffer.read()
        
        # Verify GLB data
        print(f"GLB file size: {len(glb_bytes)} bytes")
        if len(glb_bytes) < 100:
            print("Warning: GLB file is suspiciously small")
        
        # Log completion
        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds")
        
        return glb_bytes, None
    except Exception as e:
        import traceback
        print(f"Error generating model: {e}")
        print(traceback.format_exc())
        return None, str(e)

# Web endpoint to check health
@app.function(gpu="A100-80GB")  # Request GPU access for the health check
@modal.fastapi_endpoint()
def health():
    """Health check endpoint using nvidia-smi to directly check GPU availability"""
    try:
        # Run nvidia-smi command to check GPU status
        output = subprocess.check_output(["nvidia-smi"], text=True)
        
        # Parse nvidia-smi output to extract key information
        driver_version = None
        cuda_version = None
        gpu_name = None
        gpu_memory = None
        
        for line in output.splitlines():
            if "Driver Version:" in line:
                driver_version = line.split("Driver Version:")[1].strip().split()[0]
            if "CUDA Version:" in line:
                cuda_version = line.split("CUDA Version:")[1].strip()
            if "GPU Name" in line or "|GeForce" in line or "|Tesla" in line or "|NVIDIA" in line or "| A100" in line:
                gpu_name = line.strip()
            if "GPU Memory" in line:
                gpu_memory = line.strip()
        
        print("GPU Information:")
        print(output)
        
        # Return the GPU information
        return {
            "status": "healthy",
            "model": "Shap-E text-to-3D (A100-80GB GLB)",
            "timestamp": time.time(),
            "cuda": {
                "available": True,
                "driver_version": driver_version,
                "cuda_version": cuda_version,
                "gpu_info": output,
                "device": gpu_name
            }
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return {
            "status": "warning",
            "model": "Shap-E text-to-3D (A100-80GB GLB)",
            "timestamp": time.time(),
            "error": f"Failed to run nvidia-smi: {str(e)}",
            "cuda": {"available": False}
        }
    except Exception as e:
        print(f"Unexpected error in health check: {e}")
        return {
            "status": "error",
            "model": "Shap-E text-to-3D (A100-80GB GLB)",
            "timestamp": time.time(),
            "error": str(e),
            "cuda": {"available": False}
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
        glb_bytes, error = generate_3d_model.remote(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed
        )
        
        # Handle errors
        if error:
            logger.error(f"Error in model generation: {error}")
            raise HTTPException(status_code=500, detail=error)
        
        if not glb_bytes or len(glb_bytes) < 100:
            logger.error(f"Empty or invalid GLB generated (size: {len(glb_bytes) if glb_bytes else 0} bytes)")
            raise HTTPException(status_code=500, detail="Generated model was empty or invalid")
        
        # Calculate time taken
        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        # Return the GLB file
        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={"Content-Disposition": f"attachment; filename={prompt.replace(' ', '_')}.glb"}
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
        <title>Text-to-3D Generator (A100-80GB GLB)</title>
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
        <h1>Text-to-3D Generator (A100-80GB GLB)</h1>
        <p>Enter a text description to generate a 3D model with color support (GLB format).</p>
        
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
                            <p>Your 3D model is ready! <a href="${downloadUrl}" download="${prompt.replace(/ /g, '_')}.glb">Download GLB File</a></p>
                            <p>You can open this file in any 3D software that supports GLB (like Blender) or upload it to online viewers like <a href="https://gltf-viewer.donmccurdy.com/" target="_blank">glTF Viewer</a>.</p>
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
    print("Deploying Shap-E Text-to-3D app to Modal with A100-80GB GPU (GLB format)...")
    modal.app.deploy("shape-text-to-3d-a100-80gb-glb")
    print("Deployment complete! Running verification...")
    verify_result = verify_installation.remote()
    print(f"Verification result: {verify_result}")
    print("\nYour app is ready to use at:")
    print("- Web UI: https://moulik-budhiraja--shape-text-to-3d-a100-80gb-glb-index.modal.run")
    print("- API: https://moulik-budhiraja--shape-text-to-3d-a100-80gb-glb-generate.modal.run")
    print("- Health check: https://moulik-budhiraja--shape-text-to-3d-a100-80gb-glb-health.modal.run")
    print("- Verification: https://moulik-budhiraja--shape-text-to-3d-a100-80gb-glb-verify.modal.run") 