import requests
import os
import argparse
import time
import json

def main():
    parser = argparse.ArgumentParser(description="Test the Shap-E Text-to-3D API")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="URL of the API")
    parser.add_argument("--job_id", type=str, help="Job ID for the /site/ format (if using AMD platform)")
    parser.add_argument("--prompt", type=str, default="A detailed unicorn", help="Text prompt for generating the 3D model")
    parser.add_argument("--output", type=str, default="output.stl", help="Output file name")
    parser.add_argument("--guidance_scale", type=float, default=15.0, help="Guidance scale for generation")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of diffusion steps")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create the payload
    payload = {
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps
    }
    
    # Format the URL properly based on whether we're using the AMD platform or direct access
    api_url = args.url
    if args.job_id:
        # Format for AMD platform: http://100.66.69.43:5000/site/[job_id]
        if not api_url.endswith('/'):
            api_url += '/'
        api_url = f"{api_url}site/{args.job_id}"
    
    print(f"Sending request to {api_url}/generate")
    print(f"Prompt: '{args.prompt}'")
    print(f"Parameters: guidance_scale={args.guidance_scale}, num_steps={args.num_steps}")
    
    # Check health endpoint first
    try:
        health_response = requests.get(f"{api_url}/health", timeout=30)
        if health_response.status_code == 200:
            print("Health check successful ✓")
        else:
            print(f"Warning: Health check returned {health_response.status_code}")
    except Exception as e:
        print(f"Warning: Could not reach health endpoint: {e}")
    
    # Make the API request
    try:
        start_time = time.time()
        
        if args.verbose:
            print(f"Request payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{api_url}/generate", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            stream=True,  # Stream the response to handle large files
            timeout=120   # Increased timeout for the AMD platform
        )
        
        if response.status_code == 200:
            # Save the STL file
            with open(args.output, "wb") as f:
                downloaded_bytes = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    if args.verbose and downloaded_bytes % (1024*1024) == 0:
                        print(f"Downloaded {downloaded_bytes / (1024*1024):.2f} MB...")
            
            end_time = time.time()
            file_size = os.path.getsize(args.output)
            
            print(f"✅ Successfully saved 3D model to: {args.output}")
            print(f"File size: {file_size / 1024:.2f} KB")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            print(f"Download rate: {(file_size / 1024) / (end_time - start_time):.2f} KB/s")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error content: {response.text[:500]}...")
    
    except requests.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 