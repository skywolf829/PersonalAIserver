# smoke_test.py
import requests
import base64
from PIL import Image
import io
import time

def run_smoke_test(base_url="http://localhost:8000", username="admin", password="admin"):
    print("Running smoke test...")
    
    # Test 1: Server health check
    print("\n1. Testing server health...")
    try:
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        print("✅ Server is running")
    except:
        print("❌ Server health check failed")
        return
    
    # Test 2: Authentication
    print("\n2. Testing authentication...")
    try:
        auth_response = requests.post(
            f"{base_url}/token",
            data={"username": username, "password": password}
        )
        assert auth_response.status_code == 200
        token = auth_response.json()["access_token"]
        print("✅ Authentication successful")
    except:
        print("❌ Authentication failed")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test 3: Text Generation
    print("\n3. Testing text generation...")
    try:
        text_response = requests.post(
            f"{base_url}/generate/text",
            headers=headers,
            json={
                "prompt": "Write a one-line test message."
            }
        )
        assert text_response.status_code == 200
        print("✅ Text generation successful")
        print(f"Prompt: Write a one-line test message.")
        print(f"Generated text: {text_response.json()['generated_content']}")
    except Exception as e:
        print(f"❌ Text generation failed: {str(e)}")
    
    # Test 4: Image Generation
    print("\n4. Testing image generation...")
    try:
        image_response = requests.post(
            f"{base_url}/generate/image",
            headers=headers,
            json={
                "prompt": "A black porsche 911 convertible rolling through a fall forest, image shot from the top down, cinematic, high quality, 8k"
            }
        )
        assert image_response.status_code == 200
        
        # Try to decode and save the image
        image_data = image_response.json()["generated_content"]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image.save("test_output.png")
        
        print("✅ Image generation successful")
        print("Image saved as 'test_output.png'")
    except Exception as e:
        print(f"❌ Image generation failed: {str(e)}")
    
    print("\nSmoke test completed!")

if __name__ == "__main__":
    run_smoke_test()