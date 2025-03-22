import argparse
import requests
import time
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Test the Shap-E Text-to-3D API")
    parser.add_argument("--url", type=str, default="http://localhost:9200", help="URL of the API")
    parser.add_argument("--prompt", type=str, default="a detailed sculpture of a dragon", 
                        help="Text prompt for generating the 3D model")
    parser.add_argument("--output", type=str, default="output.stl", help="Output file name")
    parser.add_argument("--guidance_scale", type=float, default=15.0, help="Guidance scale for generation")
    parser.add_argument("--num_steps", type=int, default=24, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create the payload
    payload = {
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps
    }
    
    if args.seed is not None:
        payload["seed"] = args.seed
    
    api_url = args.url.rstrip('/')
    
    print(f"Sending request to {api_url}/generate")
    print(f"Prompt: '{args.prompt}'")
    print(f"Parameters: guidance_scale={args.guidance_scale}, num_steps={args.num_steps}, seed={args.seed}")
    
    # Check health endpoint first
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("✅ Health check successful")
            print(f"   GPU: {'Available - ' + health_data.get('gpu_name', 'Unknown') if health_data.get('gpu', False) else 'Not available'}")
            print(f"   PyTorch: {health_data.get('pytorch_version', 'Unknown')}")
        else:
            print(f"⚠️ Health check returned {health_response.status_code}")
    except Exception as e:
        print(f"⚠️ Could not reach health endpoint: {e}")
        print("   The API server might not be running.")
        return
    
    # Make the API request
    try:
        start_time = time.time()
        
        print("\nGenerating 3D model...")
        
        response = requests.post(
            f"{api_url}/generate", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=300  # 5 minutes timeout for complex models
        )
        
        if response.status_code == 200:
            # Save the STL file
            with open(args.output, "wb") as f:
                downloaded_bytes = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes % (1024*1024) == 0:  # Every 1MB
                        print(f"   Downloaded {downloaded_bytes / (1024*1024):.1f} MB...")
            
            end_time = time.time()
            file_size = os.path.getsize(args.output)
            
            print(f"\n✅ Successfully saved 3D model to: {os.path.abspath(args.output)}")
            print(f"   File size: {file_size / 1024:.1f} KB")
            print(f"   Total time: {end_time - start_time:.1f} seconds")
        else:
            print(f"\n❌ Error: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error details: {error_data.get('error', response.text)}")
            except:
                print(f"   Error content: {response.text[:500]}...")
    
    except requests.RequestException as e:
        print(f"\n❌ Connection error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 