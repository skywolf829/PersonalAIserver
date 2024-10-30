from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import pipeline
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
import uvicorn
from typing import List, Optional
from datetime import datetime, timedelta
import json
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
import base64
from io import BytesIO
from pathlib import Path
import argparse
import psutil

# Security configuration
SECRET_KEY = "your-secret-key-here"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup model cache directory
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Server")

# Configure CORS - Replace with your GitHub Pages URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://skywolf829.github.io", 
                   "http://localhost:3000"
                   "https://swwurster.com"
                   "https://api.swwurster.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7

class GenerationResponse(BaseModel):
    generated_content: str
    content_type: str
    model_used: str

# User database - Replace with your desired username/password
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin"),  # Change this!
        "disabled": False
    }
}

class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model_configs = {
            "text": {
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "model_type": "text",
                "cache_dir": CACHE_DIR / "Llama-3.2-1B-Instruct",
                "system_prompt": Path(__file__).parent.joinpath("system_prompt.txt").read_text()
            },
            "image": {
                "model_id": "stabilityai/stable-diffusion-3.5-large",
                "model_type": "image",
                "cache_dir": CACHE_DIR / "stable-diffusion-3.5-large"
            }
        }
        self.user_chat_history = {}

    def is_model_cached(self, modality: str) -> bool:
        """Check if the model is already cached locally."""
        cache_dir = self.model_configs[modality]["cache_dir"]
        return cache_dir.exists() and any(cache_dir.iterdir())

    def load_model(self, modality: str):
        if modality in self.loaded_models:
            return

        config = self.model_configs[modality]
        model_id = config["model_id"]
        cache_dir = config["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {modality} model: {model_id}")
        is_cached = self.is_model_cached(modality)
        load_path = str(cache_dir) if is_cached else model_id
        
        try:
            if modality == "text":
                logger.info(f"Loading text model from: {'cache' if is_cached else 'online'}")
                self.loaded_models[modality] = {
                    "pipeline": pipeline(
                        "text-generation",
                        model=load_path,
                        device=0 if self.device == "cuda" else -1
                    )
                }
                logger.info(f"Loaded text model from: {'cache' if is_cached else 'online'}")
                # Save model if it wasn't cached
                if not is_cached:
                    logger.info("Saving text model to cache")
                    self.loaded_models[modality]["pipeline"].save_pretrained(str(cache_dir))

            elif modality == "image":
                logger.info(f"Loading image model from: {'cache' if is_cached else 'online'}")
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_nf4 = SD3Transformer2DModel.from_pretrained(
                    load_path,
                    subfolder="transformer",
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16,
                    cache_dir=str(cache_dir) if not is_cached else None
                )
                model = StableDiffusion3Pipeline.from_pretrained(
                    load_path,
                    transformer=model_nf4,
                    torch_dtype=torch.bfloat16,
                    cache_dir=str(cache_dir) if not is_cached else None
                ).to(self.device)
                self.loaded_models[modality] = {"pipeline": model}
                logger.info(f"Loaded image model from: {'cache' if is_cached else 'online'}")
                # Save model if it wasn't cached
                if not is_cached:
                    logger.info("Saving image model to cache")
                    model.save_pretrained(str(cache_dir))

        except Exception as e:
            logger.error(f"Error loading {modality} model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    async def generate_text(self, prompt: str, current_user: User):
        if "text" not in self.loaded_models:
            self.load_model("text")
        
        if current_user.username not in self.user_chat_history:
            self.user_chat_history[current_user.username] = [{"role": "system", "content": self.model_configs["text"]["system_prompt"]}]

        try:
            pipeline = self.loaded_models["text"]["pipeline"]
            chat = self.user_chat_history[current_user.username]
            chat.append({"role": "user", "content": prompt})
            result = pipeline(chat, max_length=1024)
            self.user_chat_history[current_user.username] = result[0]["generated_text"]
            return result[0]["generated_text"][-1]['content']
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def generate_image(self, prompt: str):
        if "image" not in self.loaded_models:
            self.load_model("image")
        
        try:
            pipeline = self.loaded_models["image"]["pipeline"]
            image = pipeline(prompt, num_inference_steps=50, guidance_scale=3.5).images[0]
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize model manager
model_manager = ModelManager()

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    if username in USERS_DB:
        user_dict = USERS_DB[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = get_user(username)
        if user is None:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception

# Endpoints
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.username})
    model_manager.user_chat_history[user.username] = [{"role": "system", "content": model_manager.model_configs["text"]["system_prompt"]}]
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
async def root():
    return {
        "status": "running",
        "device": model_manager.device,
        "cached_models": {
            modality: model_manager.is_model_cached(modality)
            for modality in model_manager.model_configs
        }
    }

@app.post("/generate/text")
async def generate_text(
    request: GenerationRequest,
    current_user: User = Depends(get_current_user)
):
    text = await model_manager.generate_text(
        request.prompt,
        current_user
    )
    return GenerationResponse(
        generated_content=text,
        content_type="text",
        model_used="llama-3.2"
    )

@app.post("/generate/image")
async def generate_image(
    request: GenerationRequest,
    current_user: User = Depends(get_current_user)
):
    image_b64 = await model_manager.generate_image(request.prompt)
    return GenerationResponse(
        generated_content=image_b64,
        content_type="image",
        model_used="stable-diffusion-3.5"
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "device": model_manager.device
    }
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", action="store_true", default=False, help="Run server with public access")
    args = parser.parse_args()
    
    host = "0.0.0.0" if args.public else "127.0.0.1"
    uvicorn.run("backend:app", host=host, port=8000)