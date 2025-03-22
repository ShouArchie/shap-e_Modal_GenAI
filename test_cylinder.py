import requests
import sys

# The URL where your API is running
url = "http://localhost:9200/generate"  # Change this if your API is running elsewhere

print(f"Sending request to {url}")
print("Prompt: 'Create a cylinder'")

try:
    # Send the request
    response = requests.post(
        url,
        json={"prompt": "Create a cylinder"},
        timeout=30
    )
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        # Save the response content to a file
        with open("cylinder.stl", "wb") as f:
            f.write(response.content)
        print(f"Successfully saved 3D model to cylinder.stl")
        print(f"File size: {len(response.content) / 1024:.2f} KB")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}") 