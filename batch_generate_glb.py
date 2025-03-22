"""
Batch processing script for generating multiple 3D models using Modal A100-80GB
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

# The endpoint URL for your A100-80GB deployment
API_ENDPOINT = "https://moulik-budhiraja--shape-text-to-3d-a100-80gb-glb-generate.modal.run"

def generate_model(prompt, guidance_scale=15.0, num_steps=64, seed=None, output_dir="output"):
    """
    Generate a single 3D model and save it to the output directory
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a safe filename based on the prompt
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)
    safe_prompt = safe_prompt[:30]  # Limit length
    
    # Add seed to filename if provided
    if seed is not None:
        output_file = f"{output_dir}/{safe_prompt}_seed{seed}.glb"
    else:
        output_file = f"{output_dir}/{safe_prompt}.glb"
    
    # Prepare parameters
    params = {
        "prompt": prompt,
        "guidance_scale": guidance_scale,
        "num_steps": num_steps
    }
    
    if seed is not None:
        params["seed"] = seed
    
    # Log start time
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating: '{prompt}' with {num_steps} steps...")
    
    try:
        # Send request
        response = requests.post(API_ENDPOINT, params=params, timeout=300)
        
        # End timing
        end_time = time.time()
        elapsed = end_time - start_time
        
        if response.status_code == 200:
            # Get content size
            content_size = len(response.content)
            
            # Save the model to a file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] '{prompt}': Done in {elapsed:.2f}s ({num_steps/elapsed:.2f} steps/s) - {content_size/1024:.1f} KB")
            
            return {
                "prompt": prompt,
                "success": True,
                "file": output_file,
                "time": elapsed,
                "size": content_size,
                "steps_per_second": num_steps/elapsed
            }
        else:
            print(f"❌ [{datetime.now().strftime('%H:%M:%S')}] '{prompt}': Failed with status {response.status_code} in {elapsed:.2f}s")
            print(f"   Response: {response.text[:100]}...")
            
            return {
                "prompt": prompt,
                "success": False,
                "error": f"Status code: {response.status_code}",
                "time": elapsed
            }
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"❌ [{datetime.now().strftime('%H:%M:%S')}] '{prompt}': Error after {elapsed:.2f}s - {str(e)}")
        
        return {
            "prompt": prompt,
            "success": False,
            "error": str(e),
            "time": elapsed
        }

def process_batch(prompts, guidance_scale=15.0, num_steps=64, seed=None, output_dir="output", max_workers=3):
    """
    Process a batch of prompts in parallel
    """
    results = []
    
    # Use timestamp for the output directory to keep batches separate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = f"{output_dir}/batch_{timestamp}"
    
    print(f"\n===== BATCH PROCESSING =====")
    print(f"Starting batch with {len(prompts)} prompts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {batch_dir}")
    print(f"Max parallel workers: {max_workers}")
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
    print(f"Models generated: {len(successful)}/{len(prompts)}")
    
    if successful:
        avg_time = sum(r["time"] for r in successful) / len(successful)
        avg_steps_per_second = sum(r["steps_per_second"] for r in successful) / len(successful)
        print(f"Average time per model: {avg_time:.2f} seconds")
        print(f"Average performance: {avg_steps_per_second:.2f} steps/second")
    
    # Save report
    report = {
        "timestamp": timestamp,
        "total_time": batch_elapsed,
        "total_prompts": len(prompts),
        "successful": len(successful),
        "parameters": {
            "guidance_scale": guidance_scale,
            "num_steps": num_steps,
            "seed": seed,
            "max_workers": max_workers
        },
        "results": results
    }
    
    with open(f"{batch_dir}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {batch_dir}/report.json")
    
    return results

def load_prompts_from_file(file_path):
    """
    Load prompts from a text file (one prompt per line)
    """
    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate 3D models using Modal A100-80GB")
    parser.add_argument("--prompts", type=str, help="Text file with prompts (one per line)")
    parser.add_argument("--guidance", type=float, default=15.0, help="Guidance scale (default: 15.0)")
    parser.add_argument("--steps", type=int, default=64, help="Number of diffusion steps (default: 64)")
    parser.add_argument("--seed", type=int, help="Starting random seed (will increment for each prompt if specified)")
    parser.add_argument("--output", type=str, default="output", help="Output directory (default: 'output')")
    parser.add_argument("--parallel", type=int, default=3, help="Maximum number of parallel generations (default: 3)")
    parser.add_argument("prompt", type=str, nargs="*", help="One or more text prompts for 3D models")
    
    args = parser.parse_args()
    
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
        max_workers=args.parallel
    ) 