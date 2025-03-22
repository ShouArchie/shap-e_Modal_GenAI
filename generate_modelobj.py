import requests
import os
import time
from datetime import datetime

# Your Modal API endpoint
ENDPOINT = "https://moulik-budhiraja--shape-text-to-3d-a100-80gb-obj-generate.modal.run"

# Test prompt (change this to whatever you want to generate)
prompt = "a colourful teddy bear"

# Parameters
params = {
    "prompt": prompt,
    "guidance_scale": 15.0,  # Controls how closely the model follows your text
    "num_steps": 16,         # Higher values give better quality but take longer
    # "seed": 42,            # Optional: Uncomment to set a specific seed for reproducibility
    "batch_size": 1          # Number of models to generate (keep at 1 for single model)
}

# Create output directory if it doesn't exist
output_dir = "generated_models"
os.makedirs(output_dir, exist_ok=True)

# Start timing
start_time = time.time()
print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending request to generate: '{prompt}'")
print(f"Using {params['num_steps']} steps, guidance scale {params['guidance_scale']}")

try:
    # Send request to the API
    response = requests.post(ENDPOINT, params=params, timeout=600)  # 10-minute timeout
    
    # Check for successful response
    response.raise_for_status()
    
    # Calculate time taken
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Create a clean filename
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)
    safe_prompt = safe_prompt[:50]  # Limit length for filename
    
    # Define output file path
    output_file = f"{output_dir}/{safe_prompt}.obj"
    
    # Save the file
    with open(output_file, "wb") as f:
        f.write(response.content)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Success! Generated in {elapsed:.2f} seconds")
    print(f"Model saved to: {os.path.abspath(output_file)}")
    
    # Calculate performance metrics
    steps_per_second = params['num_steps'] / elapsed
    print(f"Performance: {steps_per_second:.2f} steps/second")
    
except requests.exceptions.RequestException as e:
    # Handle API errors
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Error: {str(e)}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        print(f"Response text: {e.response.text}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Failed after {elapsed:.2f} seconds")