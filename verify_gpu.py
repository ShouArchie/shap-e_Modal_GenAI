"""
Verification script for Modal GPU allocation and performance testing
"""

import requests
import time
import json
import sys
import argparse
from datetime import datetime

# The URL of your A100-80GB deployment
BASE_URL = "https://moulik-budhiraja--shape-text-to-3d-a100-80gb-glb"
GENERATE_URL = f"{BASE_URL}-generate.modal.run"
HEALTH_URL = f"{BASE_URL}-health.modal.run" 
VERIFY_URL = f"{BASE_URL}-verify.modal.run"

def check_health():
    """Check the health of the endpoint and GPU availability"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking health at {HEALTH_URL}...")
    
    try:
        response = requests.get(HEALTH_URL, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("\nHealth check response:")
            print(json.dumps(data, indent=2))
            
            # Extract CUDA info
            cuda_available = False
            if 'cuda' in data:
                cuda = data['cuda']
                cuda_available = cuda.get('available', False)
                
            if cuda_available:
                print("\n✅ CUDA is available! GPU is properly configured.")
                print(f"   GPU: {data.get('cuda', {}).get('device', 'Unknown')}")
                return True
            else:
                print("\n⚠️ CUDA is not showing as available in the health check.")
                print("   This could mean the GPU is not properly allocated.")
                return False
        else:
            print(f"\n❌ Error: Health check failed with status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"\n❌ Error checking health: {e}")
        return False

def run_verification():
    """Run the verification endpoint to test full pipeline"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running verification at {VERIFY_URL}...")
    
    try:
        # The verification endpoint might take a while to complete
        response = requests.get(VERIFY_URL, timeout=120)
        if response.status_code == 200:
            data = response.json()
            print("\nVerification response:")
            print(json.dumps(data, indent=2))
            
            if data.get('status') == 'success':
                print("\n✅ Verification successful! The full pipeline is working.")
                return True
            else:
                print(f"\n⚠️ Verification completed but reported status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
                return False
        else:
            print(f"\n❌ Error: Verification failed with status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print("\n⚠️ Verification request timed out. This might be normal if the server is initializing.")
        print("   Try running the script again in a few minutes.")
        return False
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        return False

def test_generation(prompt="test sphere", num_steps=16):
    """Test model generation performance with a simple prompt"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Testing generation with prompt: '{prompt}'...")
    
    params = {
        "prompt": prompt,
        "guidance_scale": 15.0,
        "num_steps": num_steps,
        "seed": 42  # Use fixed seed for consistent comparison
    }
    
    try:
        # Start timing
        start_time = time.time()
        
        # Send request
        print(f"Sending request to {GENERATE_URL}...")
        response = requests.post(GENERATE_URL, params=params, timeout=300)
        
        # End timing
        end_time = time.time()
        elapsed = end_time - start_time
        
        if response.status_code == 200:
            content_size = len(response.content)
            print(f"\n✅ Generation successful!")
            print(f"   Response size: {content_size/1024:.1f} KB")
            print(f"   Time taken: {elapsed:.2f} seconds")
            print(f"   Performance: ~{num_steps/elapsed:.2f} steps per second")
            
            # Save the model to a file for inspection
            output_file = f"test_sphere_{num_steps}steps.glb"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"   Saved test model to: {output_file}")
            
            return True
        else:
            print(f"\n❌ Generation failed with status code {response.status_code}")
            print(f"   Response: {response.text}")
            print(f"   Time taken: {elapsed:.2f} seconds")
            return False
    except requests.exceptions.Timeout:
        print("\n❌ Generation request timed out.")
        print("   This might indicate that the GPU is underpowered or experiencing issues.")
        return False
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        return False

def measure_gpu_performance():
    """Test generation with different step counts to measure GPU performance"""
    print("\n===== PERFORMANCE TESTING =====")
    print("Running multiple tests with different step counts to measure GPU performance...")
    
    results = []
    step_counts = [16, 32, 64]
    
    for steps in step_counts:
        print(f"\n----- Testing with {steps} steps -----")
        start_time = time.time()
        prompt = f"test sphere with {steps} steps"
        
        try:
            params = {
                "prompt": prompt,
                "guidance_scale": 15.0,
                "num_steps": steps,
                "seed": 42
            }
            
            response = requests.post(GENERATE_URL, params=params, timeout=300)
            end_time = time.time()
            elapsed = end_time - start_time
            
            if response.status_code == 200:
                content_size = len(response.content)
                results.append({
                    "steps": steps,
                    "time": elapsed,
                    "size": content_size,
                    "steps_per_second": steps/elapsed
                })
                print(f"✅ Success - {elapsed:.2f} seconds, {steps/elapsed:.2f} steps/sec")
            else:
                print(f"❌ Failed - Status code: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Print performance summary
    if results:
        print("\n===== PERFORMANCE SUMMARY =====")
        print("Steps | Time (s) | Steps/sec | Size (KB)")
        print("------+----------+-----------+----------")
        for r in results:
            print(f"{r['steps']:5d} | {r['time']:.2f} | {r['steps_per_second']:.2f} | {r['size']/1024:.1f}")
    else:
        print("\n❌ No successful performance tests to report.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify GPU allocation and performance for Modal deployment")
    parser.add_argument("--health", action="store_true", help="Run health check only")
    parser.add_argument("--verify", action="store_true", help="Run verification endpoint test")
    parser.add_argument("--test", action="store_true", help="Test generation with a simple prompt")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    
    args = parser.parse_args()
    
    # If no specific options, run health check
    if not (args.health or args.verify or args.test or args.performance or args.all):
        args.health = True
    
    # Run all checks if --all is specified
    if args.all:
        args.health = True
        args.verify = True 
        args.test = True
        args.performance = True
    
    print("===== MODAL A100-80GB GPU VERIFICATION =====")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests in sequence
    if args.health:
        health_ok = check_health()
        print()
    
    if args.verify:
        verify_ok = run_verification()
        print()
    
    if args.test:
        test_ok = test_generation()
        print()
    
    if args.performance:
        measure_gpu_performance()
    
    print(f"\nVerification completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 