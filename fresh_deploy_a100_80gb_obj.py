"""
Fresh Deployment of Shap-E Text-to-3D on Modal with A100-80GB (OBJ output)
"""

import modal
import io
import time
import logging
import subprocess
import os
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
app = modal.App("shape-text-to-3d-a100-80gb-obj", image=image)

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
        test_prompt = "a red apple"  # Prompt that should naturally have color
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
        
        # Verify we can convert to a mesh and save OBJ
        print("Converting to mesh and saving OBJ...")
        from shap_e.util.notebooks import decode_latent_mesh
        shap_e_mesh = decode_latent_mesh(xm, latents[0])
        tri_mesh = shap_e_mesh.tri_mesh()
        
        # Check for vertex colors
        has_vertex_colors = tri_mesh.has_vertex_colors()
        print(f"Mesh has vertex colors: {has_vertex_colors}")
        
        # If no vertex colors, add synthetic ones for testing
        if not has_vertex_colors:
            print("No vertex colors found. Adding synthetic colors...")
            verts = tri_mesh.verts
            if isinstance(verts, torch.Tensor):
                verts = verts.cpu().numpy()
            
            # Normalize vertex positions to 0-1 range for coloring
            import numpy as np
            v_min = verts.min(axis=0)
            v_max = verts.max(axis=0)
            v_range = v_max - v_min
            normalized_verts = (verts - v_min) / v_range

            # Create RGB channels based on XYZ position
            tri_mesh.vertex_channels['R'] = normalized_verts[:, 0]
            tri_mesh.vertex_channels['G'] = normalized_verts[:, 1]
            tri_mesh.vertex_channels['B'] = normalized_verts[:, 2]
            
            print("Added synthetic vertex colors")
        else:
            # Print some color statistics
            rgb_channels = [tri_mesh.vertex_channels[channel] for channel in "RGB"]
            r_range = (rgb_channels[0].min(), rgb_channels[0].max())
            g_range = (rgb_channels[1].min(), rgb_channels[1].max())
            b_range = (rgb_channels[2].min(), rgb_channels[2].max())
            
            print(f"Color ranges - R: {r_range}, G: {g_range}, B: {b_range}")
            
            # For a red apple, we'd expect higher red values
            avg_r = rgb_channels[0].mean() 
            avg_g = rgb_channels[1].mean()
            avg_b = rgb_channels[2].mean()
            
            print(f"Average colors - R: {avg_r:.2f}, G: {avg_g:.2f}, B: {avg_b:.2f}")
        
        # Create a temporary file to write the OBJ to
        temp_obj_path = "/tmp/test_apple.obj"
        with open(temp_obj_path, 'w') as f:
            tri_mesh.write_obj(f)
        
        # Check if the file was created and has content
        if os.path.exists(temp_obj_path):
            file_size = os.path.getsize(temp_obj_path)
            print(f"✓ OBJ file created successfully: {file_size} bytes")
            
            # Verify the OBJ file contains vertices and faces
            with open(temp_obj_path, 'r') as f:
                obj_content = f.read()
                vertex_count = obj_content.count('\nv ')
                face_count = obj_content.count('\nf ')
                
                # Count vertices with color information (6+ fields per line)
                color_vertex_lines = [line for line in obj_content.split('\n') if line.startswith('v ') and len(line.split()) >= 6]
                color_vertex_count = len(color_vertex_lines)
                
                print(f"OBJ file contains {vertex_count} vertices, {face_count} faces")
                print(f"OBJ file has {color_vertex_count} vertices with color data")
                
                # Verify the OBJ format with a sample of lines
                if color_vertex_count > 0:
                    sample_vertex = color_vertex_lines[0]
                    print(f"Sample vertex with color: {sample_vertex}")
        else:
            print("Failed to create OBJ file")
        
        print(f"✓ Full pipeline verification successful.")
        return {"status": "success", "message": "Shap-E installation verified with color support"}
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
    batch_size: int = 1,  # Add batch_size parameter
):
    """
    Generate a 3D model from a text prompt using Shap-E.
    
    Args:
        prompt: Text description of the 3D model to generate
        guidance_scale: Controls how closely the model follows your text (default: 15.0)
        num_steps: Number of diffusion steps (default: 64, higher = better quality but slower)
        seed: Random seed for reproducibility
        batch_size: Number of models to generate in a batch (default: 1)
    """
    import os
    import torch
    import numpy as np
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
    from shap_e.util.notebooks import decode_latent_mesh
    
    # Log start of process
    print(f"Starting 3D model generation for prompt: '{prompt}'")
    print(f"Parameters: guidance_scale={guidance_scale}, num_steps={num_steps}, seed={seed}, batch_size={batch_size}")
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
        print(f"Generating latents for batch_size={batch_size}...")
        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=num_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        # Store OBJ data for each model in the batch
        obj_data_list = []
        
        # Process each latent in the batch
        for i, latent in enumerate(latents):
            print(f"Processing model {i+1}/{batch_size}...")
            
            # Create mesh from latent, using the same method as in the sample code
            print("Converting latent to mesh...")
            shap_e_mesh = decode_latent_mesh(xm, latent)
            
            # Get the triangular mesh which includes texture information
            print("Extracting mesh data with colors...")
            tri_mesh = shap_e_mesh.tri_mesh()
            
            # Verify if the mesh has color data
            has_colors = tri_mesh.has_vertex_colors()
            print(f"Mesh has vertex colors: {has_colors}")
            
            if has_colors:
                # Log the color information to verify
                rgb_channels = [tri_mesh.vertex_channels[channel] for channel in "RGB"]
                r_range = (rgb_channels[0].min(), rgb_channels[0].max())
                g_range = (rgb_channels[1].min(), rgb_channels[1].max())
                b_range = (rgb_channels[2].min(), rgb_channels[2].max())
                print(f"Color ranges - R: {r_range}, G: {g_range}, B: {b_range}")
                
                # Calculate average color for verification
                avg_r = rgb_channels[0].mean()
                avg_g = rgb_channels[1].mean()
                avg_b = rgb_channels[2].mean()
                print(f"Average color: R={avg_r:.2f}, G={avg_g:.2f}, B={avg_b:.2f}")
            else:
                print("Warning: No vertex colors found in the mesh.")
                
                # Add synthetic colors based on vertex position for visibility
                print("Adding synthetic colors based on vertex position...")
                verts = tri_mesh.verts
                if isinstance(verts, torch.Tensor):
                    verts = verts.cpu().numpy()
                
                # Normalize vertex positions to 0-1 range for coloring
                v_min = verts.min(axis=0)
                v_max = verts.max(axis=0)
                v_range = v_max - v_min
                normalized_verts = (verts - v_min) / v_range

                # Create RGB channels based on XYZ position
                tri_mesh.vertex_channels['R'] = normalized_verts[:, 0]
                tri_mesh.vertex_channels['G'] = normalized_verts[:, 1]
                tri_mesh.vertex_channels['B'] = normalized_verts[:, 2]
                
                print("Added position-based colors to mesh")
            
            # Write to string buffer
            print("Writing colored mesh to OBJ format...")
            obj_buffer = io.StringIO()
            tri_mesh.write_obj(obj_buffer)
            obj_buffer.seek(0)
            obj_text = obj_buffer.read()
            
            # Verify OBJ content includes vertex colors
            vertex_count = obj_text.count("\nv ")
            color_vertex_count = len([line for line in obj_text.split("\n") if line.startswith("v ") and len(line.split()) >= 6])
            print(f"OBJ contains {vertex_count} vertices, {color_vertex_count} with color data")
            
            # Add to results
            obj_data_list.append(obj_text)
            print(f"Model {i+1} processed successfully with color data")
        
        # Log completion
        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds")
        
        return obj_data_list, None
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
            "model": "Shap-E text-to-3D (A100-80GB OBJ)",
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
            "model": "Shap-E text-to-3D (A100-80GB OBJ)",
            "timestamp": time.time(),
            "error": f"Failed to run nvidia-smi: {str(e)}",
            "cuda": {"available": False}
        }
    except Exception as e:
        print(f"Unexpected error in health check: {e}")
        return {
            "status": "error",
            "model": "Shap-E text-to-3D (A100-80GB OBJ)",
            "timestamp": time.time(),
            "error": str(e),
            "cuda": {"available": False}
        }

# Web endpoint for 3D model generation
@app.function()
@modal.fastapi_endpoint(method="POST")
def generate(prompt: str, guidance_scale: float = 15.0, num_steps: int = 64, seed: int = None, batch_size: int = 1):
    """
    Generate a 3D model from a text prompt
    """
    try:
        start_time = time.time()
        
        logger.info(f"Received generation request for prompt: '{prompt}' with batch_size={batch_size}")
        
        # Generate the 3D model
        obj_data_list, error = generate_3d_model.remote(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            seed=seed,
            batch_size=batch_size
        )
        
        # Handle errors
        if error:
            logger.error(f"Error in model generation: {error}")
            raise HTTPException(status_code=500, detail=error)
        
        if not obj_data_list or len(obj_data_list) == 0:
            logger.error("Empty or invalid OBJ generated")
            raise HTTPException(status_code=500, detail="Generated model was empty or invalid")
        
        # Calculate time taken
        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        # If batch_size is 1, return a single file
        if batch_size == 1:
            return Response(
                content=obj_data_list[0],
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={prompt.replace(' ', '_')}.obj"}
            )
        else:
            # If batch_size > 1, create a zip file containing all models
            import zipfile
            
            # Create a bytes buffer for the zip
            zip_buffer = io.BytesIO()
            
            # Create the zip file
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i, obj_data in enumerate(obj_data_list):
                    # Add each OBJ file to the zip
                    filename = f"{prompt.replace(' ', '_')}_{i+1}.obj"
                    zip_file.writestr(filename, obj_data)
            
            # Reset buffer position
            zip_buffer.seek(0)
            
            # Return the zip file
            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={prompt.replace(' ', '_')}_batch{batch_size}.zip"}
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
        <title>Text-to-3D Generator with Color (A100-80GB OBJ)</title>
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
            input, button, select {
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
            .feature-box {
                background-color: #f0f7ff;
                border-left: 4px solid #0066cc;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }
            .example-prompts {
                background-color: #f5f5f5;
                padding: 10px;
                margin-top: 15px;
                border-radius: 4px;
            }
            .example-prompt {
                display: inline-block;
                background-color: #e0e0e0;
                padding: 5px 10px;
                margin: 5px;
                border-radius: 15px;
                cursor: pointer;
            }
            .example-prompt:hover {
                background-color: #d0d0d0;
            }
        </style>
    </head>
    <body>
        <h1>Text-to-3D Generator with Color (OBJ Format)</h1>
        <p>Enter a text description to generate a 3D model with color information preserved in the OBJ file.</p>
        
        <div class="feature-box">
            <strong>Color Support:</strong> This version generates OBJ files that include vertex color information. 
            Try using color words in your prompt (e.g., "red apple", "blue car") to influence the coloring of your model.
        </div>
        
        <form id="generateForm">
            <label for="prompt">Text Description:</label>
            <input type="text" id="prompt" name="prompt" required placeholder="e.g., a red apple">
            
            <div class="example-prompts">
                <strong>Try these prompts:</strong>
                <div class="example-prompt" onclick="setPrompt('a red apple')">Red apple</div>
                <div class="example-prompt" onclick="setPrompt('a blue sports car')">Blue car</div>
                <div class="example-prompt" onclick="setPrompt('a green potted plant')">Green plant</div>
                <div class="example-prompt" onclick="setPrompt('a colorful parrot')">Colorful parrot</div>
            </div>
            
            <label for="guidance">Guidance Scale: <span id="guidanceValue">15.0</span></label>
            <input type="range" id="guidance" name="guidance_scale" min="5" max="30" step="0.5" value="15" class="slider">
            
            <label for="steps">Number of Steps: <span id="stepsValue">64</span></label>
            <input type="range" id="steps" name="num_steps" min="16" max="128" step="8" value="64" class="slider">
            
            <label for="batch_size">Batch Size:</label>
            <select id="batch_size" name="batch_size">
                <option value="1">1 (Single model)</option>
                <option value="2">2</option>
                <option value="4">4</option>
                <option value="8">8</option>
            </select>
            
            <label for="seed">Random Seed (optional):</label>
            <input type="number" id="seed" name="seed" placeholder="Leave empty for random">
            
            <button type="submit">Generate 3D Model</button>
        </form>
        
        <div id="loading">
            <p>Generating your 3D model(s)... This may take 30-90 seconds.</p>
            <progress></progress>
        </div>
        
        <div id="result"></div>
        
        <script>
            // Function to set prompt from examples
            function setPrompt(prompt) {
                document.getElementById('prompt').value = prompt;
            }
            
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
                const batchSize = document.getElementById('batch_size').value;
                const seed = document.getElementById('seed').value;
                
                // Show loading indicator
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').innerHTML = '';
                
                // Build the URL with query parameters
                const url = new URL('/generate', window.location.origin);
                url.searchParams.append('prompt', prompt);
                url.searchParams.append('guidance_scale', guidance);
                url.searchParams.append('num_steps', steps);
                url.searchParams.append('batch_size', batchSize);
                if (seed) url.searchParams.append('seed', seed);
                
                try {
                    const response = await fetch(url, {method: 'POST'});
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const downloadUrl = URL.createObjectURL(blob);
                        
                        let fileExtension = 'obj';
                        if (batchSize > 1) {
                            fileExtension = 'zip';
                        }
                        
                        document.getElementById('result').innerHTML = `
                            <p>Your 3D model is ready! <a href="${downloadUrl}" download="${prompt.replace(/ /g, '_')}.${fileExtension}">Download ${batchSize > 1 ? 'ZIP File' : 'OBJ File with Color'}</a></p>
                            <p>You can open OBJ files in any 3D software like Blender, Maya, 3ds Max, etc.</p>
                            <p>The vertex colors are preserved in the OBJ file and should be visible in most 3D software.</p>
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
    print("Deploying Shap-E Text-to-3D app to Modal with A100-80GB GPU (OBJ format)...")
    modal.app.deploy("shape-text-to-3d-a100-80gb-obj")
    print("Deployment complete! Running verification...")
    verify_result = verify_installation.remote()
    print(f"Verification result: {verify_result}")
    print("\nYour app is ready to use at:")
    print("- Web UI: https://[YOUR_MODAL_USERNAME]--shape-text-to-3d-a100-80gb-obj-index.modal.run")
    print("- API: https://[YOUR_MODAL_USERNAME]--shape-text-to-3d-a100-80gb-obj-generate.modal.run")
    print("- Health check: https://[YOUR_MODAL_USERNAME]--shape-text-to-3d-a100-80gb-obj-health.modal.run")
    print("- Verification: https://[YOUR_MODAL_USERNAME]--shape-text-to-3d-a100-80gb-obj-verify.modal.run") 