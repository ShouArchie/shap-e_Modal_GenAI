import os
import sys
import torch
import argparse
import time
from pathlib import Path
import numpy as np

# Add the local shap-e directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'shap-e'))

# Set environment variables for HuggingFace cache
os.environ['PYTORCH_HF_CACHE_HOME'] = os.path.join(os.getcwd(), 'shap_e_model_cache')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'shap_e_model_cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'shap_e_model_cache')

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)
print(f"Output directory: {os.path.abspath('outputs')}")

def generate_3d_model(
    prompt,
    output_path,
    guidance_scale=15.0,
    num_steps=24, #default 64
    seed=None,
    use_gpu=True
):
    """Generate a 3D model from a text prompt using Shap-E."""
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using GPU: {gpu_name}")
        else:
            device = torch.device('cpu')
            print("Using CPU (GPU not available or not selected)")
        
        # Import Shap-E modules (here to avoid import errors if not installed)
        from shap_e.diffusion.sample import sample_latents
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_model, load_config
        from shap_e.util.collections import AttrDict
        from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
        import trimesh
        
        print("Successfully imported Shap-E modules")
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            print(f"Random seed set to: {seed}")
        
        print(f"Loading text to 3D model on {device}...")
        start_time = time.time()
        
        # Load models
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        diffusion = diffusion_from_config(load_config('diffusion'))
        
        # Report memory usage after model loading
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**2)
            memory_reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"GPU memory after loading models - Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")
        
        print(f"Generating 3D model for prompt: '{prompt}'")
        print(f"Parameters: guidance_scale={guidance_scale}, num_steps={num_steps}")
        
        # Generate latents
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
        t = time.time()
        print("Creating mesh...")
        
        # This is our own implementation of decode_latent_mesh function from notebooks.py
        # Create a simple camera setup (2x2 resolution is enough since we just need the mesh)
        @torch.no_grad()
        def create_pan_cameras(size, device):
            import numpy as np
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
            if hasattr(xm, 'encoder'):
                params = xm.encoder.bottleneck_to_params(latents[0][None])
            else:
                params = xm.bottleneck_to_params(latents[0][None])
            
            # Render views with STF mode to get the mesh
            decoded = xm.renderer.render_views(
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
        print(f"Saving mesh to {output_path}...")
        mesh.export(output_path)
        
        end_time = time.time()
        print(f"✅ Done! Total generation time: {end_time - start_time:.2f} seconds")
        print(f"File saved to: {os.path.abspath(output_path)}")
        
        # Report final memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**2)
            memory_reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"Final GPU memory - Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")
            
            # Clean up CUDA memory
            torch.cuda.empty_cache()
            print("CUDA memory cache cleared")
        
        return True, output_path
    
    except Exception as e:
        import traceback
        print(f"❌ Error generating 3D model: {e}")
        traceback.print_exc()
        return False, str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D models with Shap-E")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the 3D model")
    parser.add_argument("--output", type=str, default="outputs/model.stl", help="Output STL file path")
    parser.add_argument("--guidance_scale", type=float, default=15.0, help="Guidance scale (higher = more prompt adherence)")
    parser.add_argument("--steps", type=int, default=32, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("Shap-E 3D Model Generator")
    print("="*50 + "\n")
    
    success, result = generate_3d_model(
        prompt=args.prompt,
        output_path=args.output,
        guidance_scale=args.guidance_scale,
        num_steps=args.steps,
        seed=args.seed,
        use_gpu=not args.cpu
    )
    
    if success:
        print("\n" + "="*50)
        print(f"SUCCESS: 3D model saved to {result}")
        print("="*50)
    else:
        print("\n" + "="*50)
        print(f"FAILED: {result}")
        print("="*50) 