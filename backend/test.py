# test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app, create_access_token, get_user
import base64
from PIL import Image
import io
import json

client = TestClient(app)

# Test data
TEST_USER = "admin"
TEST_PASSWORD = "your-password-here"  # Same as in main.py
VALID_TOKEN = create_access_token({"sub": TEST_USER})

@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {VALID_TOKEN}"}

# Authentication Tests
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "device" in response.json()

def test_login_success():
    response = client.post(
        "/token",
        data={"username": TEST_USER, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "token_type" in response.json()

def test_login_invalid_credentials():
    response = client.post(
        "/token",
        data={"username": "wrong", "password": "wrong"}
    )
    assert response.status_code == 401
    assert "Invalid credentials" in response.json()["detail"]

# Text Generation Tests
def test_text_generation_unauthorized():
    response = client.post(
        "/generate/text",
        json={"prompt": "Test prompt"}
    )
    assert response.status_code == 401

def test_text_generation_authorized(auth_headers):
    response = client.post(
        "/generate/text",
        headers=auth_headers,
        json={
            "prompt": "Once upon a time"
        }
    )
    assert response.status_code == 200
    assert "generated_content" in response.json()
    assert "content_type" in response.json()
    assert response.json()["content_type"] == "text"

def test_text_generation_invalid_params(auth_headers):
    response = client.post(
        "/generate/text",
        headers=auth_headers,
        json={
            "prompt": "",  # Empty prompt
        }
    )
    assert response.status_code != 200

# Image Generation Tests
def test_image_generation_unauthorized():
    response = client.post(
        "/generate/image",
        json={"prompt": "Test prompt"}
    )
    assert response.status_code == 401

def test_image_generation_authorized(auth_headers):
    response = client.post(
        "/generate/image",
        headers=auth_headers,
        json={"prompt": "A beautiful sunset"}
    )
    assert response.status_code == 200
    assert "generated_content" in response.json()
    assert "content_type" in response.json()
    assert response.json()["content_type"] == "image"
    
    # Verify the base64 image
    image_data = response.json()["generated_content"]
    try:
        # Try to decode and open the image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        assert image.size[0] > 0
        assert image.size[1] > 0
    except Exception as e:
        pytest.fail(f"Failed to decode image: {str(e)}")

# Error Handling Tests
def test_invalid_token():
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.post(
        "/generate/text",
        headers=headers,
        json={"prompt": "Test"}
    )
    assert response.status_code == 401

def test_missing_prompt(auth_headers):
    response = client.post(
        "/generate/text",
        headers=auth_headers,
        json={}
    )
    assert response.status_code == 422  # Validation error

def test_invalid_generation_type(auth_headers):
    response = client.post(
        "/generate/invalid_type",
        headers=auth_headers,
        json={"prompt": "Test"}
    )
    assert response.status_code == 404

# Load Testing
def test_concurrent_requests(auth_headers):
    import concurrent.futures
    
    def make_request():
        return client.post(
            "/generate/text",
            headers=auth_headers,
            json={"prompt": "Test prompt"}
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(make_request) for _ in range(3)]
        responses = [future.result() for future in futures]
        
    assert all(response.status_code == 200 for response in responses)

# Performance Testing
def test_response_time(auth_headers):
    import time
    
    start_time = time.time()
    response = client.post(
        "/generate/text",
        headers=auth_headers,
        json={"prompt": "Quick test"}
    )
    end_time = time.time()
    
    assert response.status_code == 200
    assert end_time - start_time < 30  # Should respond within 30 seconds

if __name__ == "__main__":
    pytest.main(["-v"])