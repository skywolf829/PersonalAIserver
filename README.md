# PersonalAIserver
Stand up a GenAI server on your own available hardware. This project provides a web interface to interact with LLaMA and Stable Diffusion models (among others)for text, image, video, and 3D generation.

## Features
- Text generation using LLaMA 3.2-1B
- Image generation using Stable Diffusion 3.5
- User authentication and JWT token-based security
- React-based frontend with modern UI components
- FastAPI backend with CORS support

## Setup

Fork this repository, setup GitHub Pages, and set the repository secret `NEXT_PUBLIC_API_URL` to the URL of the backend.

The frontend will be available at `https://[username].github.io/PersonalAIserver`. Follow the instructions below to set up the backend.

### Backend
1. Create conda environment
```
conda create -n personalai python=3.11
conda activate personalai
```
2. Install dependencies
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

3. Hugging Face setup
- Login to Hugging Face
- Agree to model's terms of use: https://huggingface.co/stabilityai/stable-diffusion-3.5-large, https://huggingface.co/meta-llama/Llama-3.2-1B
- Create a read-only personal access token: https://huggingface.co/settings/tokens

## Local Frontend

In case you want to run the frontend locally, follow the instructions below.

1. Setup app environment
```
npx create-next-app@latest frontend --typescript --tailwind --eslint
cd frontend
npm install -D @shadcn/ui
npx shadcn-ui init
npx shadcn@latest add alert button card input textarea
npm install react-markdown remark-gfm react-syntax-highlighter @types/react-syntax-highlighter lucide-react --legacy-peer-deps
npm install -D @tailwindcss/typography
```

## Running the backend
1. Log in to Hugging Face with `huggingface-cli login` and use your personal access token
2. Run the backend with `python backend/backend.py`

## Running the frontend
1. Run the frontend with `npm run dev`

