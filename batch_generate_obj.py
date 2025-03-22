"""
Batch processing script for generating multiple 3D models using Modal A100-80GB (OBJ format)
"""

import requests
import argparse
import sys
import os
import time
import json
import concurrent.futures
from pathlib import Path
from datetime import datetime

# The endpoint URL for your A100-80GB OBJ deployment
API_ENDPOINT = "https://[YOUR_MODAL_USERNAME]--shape-text-to-3d-a100-80gb-obj-generate.modal.run"

def generate_model(prompt, guidance_scale=15.0, num_steps=64, seed=None, batch_size=1, output_dir="output"):
    """
    Generate a single 3D model or batch of models and save to the output directory
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a safe filename based on the prompt
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)
    safe_prompt = safe_prompt[:30]  # Limit length
    
    # Prepare parameters
    params = {
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "num_steps": num_steps,
        "batch_size": batch_size
    }
    
    if seed is not None:
        params["seed"] = seed
        output_file_base = f"{output_dir}/{safe_prompt}_seed{seed}"
    else:
        output_file_base = f"{output_dir}/{safe_prompt}"
    
    # Log start time
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating: '{prompt}' with {num_steps} steps, batch_size={batch_size}...")
    
    try:
        # Make request to the API
        response = requests.post(API_ENDPOINT, params=params, timeout=600)
        response.raise_for_status()
        
        # Check content type to determine if it's a single OBJ or ZIP of multiple models
        content_type = response.headers.get('Content-Type', '')
        
        if 'application/zip' in content_type:
            # Save as ZIP file
            output_file = f"{output_file_base}_batch{batch_size}.zip"
            with open(output_file, 'wb') as f:
                f.write(response.content)
                
            # Extract the ZIP file
            import zipfile
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                
            # Log the individual files
            extracted_files = []
            with zipfile.ZipFile(output_file) as z:
                extracted_files = z.namelist()
            
            # Calculate time and metrics
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate steps per second (across all models)
            steps_per_second = (num_steps * batch_size) / elapsed
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Generated {batch_size} models in {elapsed:.2f}s ({steps_per_second:.2f} steps/sec)")
            print(f"  → Saved ZIP to: {output_file}")
            print(f"  → Extracted {len(extracted_files)} OBJ files to {output_dir}")
            
            return {
                "prompt": prompt, 
                "success": True, 
                "time": elapsed, 
                "steps_per_second": steps_per_second,
                "batch_size": batch_size,
                "output_file": output_file,
                "extracted_files": extracted_files
            }
        else:
            # Save as single OBJ file
            output_file = f"{output_file_base}.obj"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Calculate time and metrics
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Calculate steps per second
            steps_per_second = num_steps / elapsed
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Generated in {elapsed:.2f}s ({steps_per_second:.2f} steps/sec)")
            print(f"  → Saved to: {output_file}")
            
            return {
                "prompt": prompt, 
                "success": True, 
                "time": elapsed, 
                "steps_per_second": steps_per_second,
                "batch_size": 1,
                "output_file": output_file
            }
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed: {str(e)}")
        return {"prompt": prompt, "success": False, "error": str(e), "time": elapsed}

def load_prompts_from_file(file_path):
    """Load a list of prompts from a text file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def process_batch(prompts, guidance_scale=15.0, num_steps=64, seed=None, output_dir="output", max_workers=3, models_per_batch=1):
    """
    Process a batch of prompts in parallel
    
    Args:
        prompts: List of text prompts
        guidance_scale: Guidance scale for the model
        num_steps: Number of diffusion steps
        seed: Starting random seed (will increment for each prompt if specified)
        output_dir: Output directory
        max_workers: Maximum number of concurrent API calls
        models_per_batch: Number of models to generate per API call
    """
    results = []
    
    # Use timestamp for the output directory to keep batches separate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = f"{output_dir}/batch_{timestamp}"
    
    print(f"\n===== BATCH PROCESSING =====")
    print(f"Starting batch with {len(prompts)} prompts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {batch_dir}")
    print(f"Max parallel workers: {max_workers}")
    print(f"Models per API call: {models_per_batch}")
    print(f"Steps per model: {num_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Seed: {seed if seed is not None else 'random'}")
    print("=" * 40)
    
    # Create batch directory
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    
    # Start timing for the whole batch
    batch_start_time = time.time()
    
    # Process prompts in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = []
        for i, prompt in enumerate(prompts):
            # Use sequential seeds if a base seed is provided
            task_seed = seed + i if seed is not None else None
            futures.append(
                executor.submit(
                    generate_model, 
                    prompt, 
                    guidance_scale, 
                    num_steps, 
                    task_seed, 
                    models_per_batch,
                    batch_dir
                )
            )
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
    
    # End timing
    batch_end_time = time.time()
    batch_elapsed = batch_end_time - batch_start_time
    
    # Generate report
    successful = [r for r in results if r["success"]]
    
    print("\n===== BATCH COMPLETED =====")
    print(f"Total time: {batch_elapsed:.2f} seconds")
    print(f"Prompts processed: {len(successful)}/{len(prompts)}")
    
    total_models_generated = sum(r.get("batch_size", 1) for r in successful)
    print(f"Total models generated: {total_models_generated}")
    
    if successful:
        avg_time = sum(r["time"] for r in successful) / len(successful)
        avg_steps_per_second = sum(r["steps_per_second"] for r in successful) / len(successful)
        print(f"Average time per API call: {avg_time:.2f} seconds")
        print(f"Average performance: {avg_steps_per_second:.2f} steps/second")
        
        # Calculate effective throughput
        effective_steps_per_second = (total_models_generated * num_steps) / batch_elapsed
        print(f"Effective throughput: {effective_steps_per_second:.2f} steps/second")
    
    # Save report
    report = {
        "timestamp": timestamp,
        "total_time": batch_elapsed,
        "total_prompts": len(prompts),
        "successful_prompts": len(successful),
        "total_models_generated": total_models_generated,
        "parameters": {
            "guidance_scale": guidance_scale,
            "num_steps": num_steps,
            "seed": seed,
            "max_workers": max_workers,
            "models_per_batch": models_per_batch
        },
        "results": results
    }
    
    with open(f"{batch_dir}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {batch_dir}/report.json")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate 3D models using Modal A100-80GB (OBJ format)")
    parser.add_argument("--prompts", type=str, help="Text file with prompts (one per line)")
    parser.add_argument("--guidance", type=float, default=15.0, help="Guidance scale (default: 15.0)")
    parser.add_argument("--steps", type=int, default=64, help="Number of diffusion steps (default: 64)")
    parser.add_argument("--seed", type=int, help="Starting random seed (will increment for each prompt if specified)")
    parser.add_argument("--output", type=str, default="output", help="Output directory (default: 'output')")
    parser.add_argument("--parallel", type=int, default=3, help="Maximum number of parallel API calls (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of models to generate per API call (default: 1)")
    parser.add_argument("--endpoint", type=str, help="Override the default API endpoint")
    parser.add_argument("prompt", type=str, nargs="*", help="One or more text prompts for 3D models")
    
    args = parser.parse_args()
    
    # Override API endpoint if provided
    if args.endpoint:
        API_ENDPOINT = args.endpoint
        print(f"Using custom API endpoint: {API_ENDPOINT}")
    
    # Collect prompts from arguments and/or file
    prompts = []
    
    if args.prompt:
        prompts.extend(args.prompt)
    
    if args.prompts:
        if not os.path.exists(args.prompts):
            print(f"Error: Prompts file '{args.prompts}' not found!")
            sys.exit(1)
        file_prompts = load_prompts_from_file(args.prompts)
        prompts.extend(file_prompts)
        print(f"Loaded {len(file_prompts)} prompts from '{args.prompts}'")
    
    if not prompts:
        parser.print_help()
        print("\nError: No prompts provided. Please provide prompts as arguments or in a file.")
        sys.exit(1)
    
    # Process the batch
    results = process_batch(
        prompts,
        guidance_scale=args.guidance,
        num_steps=args.steps,
        seed=args.seed,
        output_dir=args.output,
        max_workers=args.parallel,
        models_per_batch=args.batch_size
    ) 