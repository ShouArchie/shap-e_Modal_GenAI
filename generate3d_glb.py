"""
Script to generate a 3D model by sending a request to the deployed Modal app (GLB format)
"""

import requests
import argparse
import sys
import os
from pathlib import Path

# The endpoint URL for your fresh A100-80GB GLB deployment
API_ENDPOINT = "https://moulik-budhiraja--shape-text-to-3d-a100-80gb-glb-generate.modal.run"

def generate_3d_model(prompt, guidance_scale=15.0, num_steps=24, seed=None, output_file=None):
    """
    Generate a 3D model from a text prompt by calling the deployed Modal app.
    
    Args:
        prompt (str): Text description of the 3D model
        guidance_scale (float): Guidance scale for model generation
        num_steps (int): Number of diffusion steps
        seed (int, optional): Random seed for reproducibility
        output_file (str, optional): Path to save the output GLB file
    
    Returns:
        bool: True if generation was successful, False otherwise
    """
    print(f"Generating 3D model for prompt: '{prompt}'")
    print(f"Parameters: guidance_scale={guidance_scale}, num_steps={num_steps}, seed={seed}")
    
    # Prepare parameters
    params = {
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "num_steps": num_steps
    }
    
    if seed is not None:
        params["seed"] = seed
    
    # If no output file specified, create one based on the prompt
    if output_file is None:
        # Replace spaces and special characters with underscores
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)
        safe_prompt = safe_prompt[:30]  # Limit length
        output_file = f"{safe_prompt}.glb"  # Use .glb extension instead of .stl
    
    print(f"Sending request to {API_ENDPOINT}")
    try:
        # Send POST request to the API
        response = requests.post(API_ENDPOINT, params=params, timeout=300)  # 5 minute timeout
        
        # Check if request was successful
        if response.status_code == 200:
            # Get content size
            content_size = len(response.content)
            print(f"Received response: {content_size/1024:.1f} KB")
            
            # Ensure content is not empty
            if content_size < 100:  # Arbitrarily small size threshold
                print(f"Warning: Received very small file ({content_size} bytes)")
                print(f"Content: {response.content[:100]}")
                return False
            
            # Save the content to a file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully saved 3D model to {output_file}")
            
            # Get file size
            file_size = os.path.getsize(output_file)
            print(f"File size: {file_size/1024:.1f} KB")
            
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The model generation is taking too long.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def check_health():
    """
    Check the health of the deployed endpoint.
    
    Returns:
        bool: True if the endpoint is healthy, False otherwise
    """
    health_endpoint = API_ENDPOINT.replace("-generate.", "-health.")
    
    try:
        response = requests.get(health_endpoint, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("Endpoint health status:")
            print(f"  Status: {data.get('status', 'unknown')}")
            print(f"  Model: {data.get('model', 'unknown')}")
            
            # Check CUDA status
            cuda_available = False
            if 'cuda' in data:
                cuda = data['cuda']
                cuda_available = cuda.get('available', False)
                print(f"  CUDA available: {cuda_available}")
                print(f"  CUDA version: {cuda.get('version', 'unknown')}")
                print(f"  GPU device: {cuda.get('device', 'unknown')}")
            
            # Print warning if CUDA is not available
            if not cuda_available:
                print("\nWARNING: CUDA is not showing as available in the health check.")
                print("This might be normal during deployment initialization, but could")
                print("indicate that the GPU is not properly configured if it persists.")
                print("Try generating a model anyway - the GPU may still be working correctly.")
                
            return True
        else:
            print(f"Error: Health check failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error checking health: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D models from text descriptions (GLB format with color)")
    parser.add_argument("prompt", type=str, help="Text description of the 3D model", nargs="?")
    parser.add_argument("--guidance", type=float, default=15.0, help="Guidance scale (default: 15.0)")
    parser.add_argument("--steps", type=int, default=24, help="Number of diffusion steps (default: 64)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, help="Output file path (default: based on prompt)")
    parser.add_argument("--health", action="store_true", help="Check the health of the endpoint")
    
    args = parser.parse_args()
    
    # Check health if requested
    if args.health:
        success = check_health()
        if not success:
            sys.exit(1)
        
        # Exit if only health check was requested
        if not args.prompt:
            sys.exit(0)
    
    # Ensure a prompt was provided
    if not args.prompt:
        parser.print_help()
        print("\nError: A prompt is required unless using --health")
        sys.exit(1)
    
    # Generate the 3D model
    success = generate_3d_model(
        prompt=args.prompt,
        guidance_scale=args.guidance,
        num_steps=args.steps,
        seed=args.seed,
        output_file=args.output
    )
    
    if not success:
        sys.exit(1) 