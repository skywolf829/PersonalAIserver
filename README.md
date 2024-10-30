# PersonalAIserver
Stand up a GenAI server on your own available hardware. This project provides a web interface to interact with LLaMA and Stable Diffusion models (among others)for text, image, video, and 3D generation.

Many people have a nice GPU on hand and are willing to use that instead of pay for subscriptions from OpenAI, Anthropic, etc. Here, you can host your own AI models, albeit with many limitations as compared to those great services.

## Features
- Text generation using LLaMA 3.2-1B-Instruct
- Image generation using Stable Diffusion 3.5 medium
- User authentication and JWT token-based security
- React-based frontend with modern UI components
- FastAPI backend with CORS support

## Setup

The frontend for this is hosted on GitHub Pages, while the backend is hosted on your available server machine.

After setup, the frontend will be available at `https://[username].github.io/PersonalAIserver`. Follow the instructions below to set up the backend.

### Frontend

For the frontend, only a GitHub action needs to be setup to build the project page.
In your forked repository, navigate to `Settings -> Pages -> Source`, and set it to "GitHub Actions".
Next, navigate to the `Actions` tab, and 

### Backend
This requires python 3.11 and conda. Non-cuda devices may also be supported, but will generally be slower, especially for image generation.

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
- Agree to model's terms of use: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium, https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Create a read-only personal access token: https://huggingface.co/settings/tokens

4. Local environment setup
- Create a `users.json` file in the backend directory with the following format:
```
{
    "username1": {
        "username": "username1",
        "password": "password1",
        "disabled": false
    },
    "username2": {
        "username": "username2",
        "password": "password2",
        "disabled": false
    }
    ...
}
```
- Create a `.secret.key` file in the backend directory with a random string of your choosing. This is used to encrypt the JWT token. I recommend generating one with 
```
import secrets
secret_key = secrets.token_hex(32)
print(secret_key)
```
and then copying the output.

5. GitHub secrets
- Navigate to `Settings -> Secrets and variables -> Actions` in the repository
- Add a repository secret with the name `NEXT_PUBLIC_API_URL` and the value being the URL of the backend server (more details below). Example: `https://api.example.com`.

6. Running the backend
- Log in to Hugging Face with `huggingface-cli login` and use your personal access token created above as the password.
- Run the backend with `python backend/backend.py --public`, or remove `--public` if you want to run the server on localhost only. Running the server on localhost is recommended for testing purposes like testing VRAM use for the models. Localhost requires you run the frontend locally as well.

### Hosting the backend
Many options exist for hosting the backend at a public URL, but I recommend using Cloudflare Tunnel.
Cloudflare Tunnel routes traffic from the internet to your server without exposing a port or your local IP.
It also supports HTTPS, SSL encryption, and DDoS protection, among other security features you would otherwise need to setup and maintain yourself.

Other options include:
- Ngrok
- Local IP address + port forwarding

For Cloudflare Tunnel, you will need a domain name and Cloudflare account.

1. Install cloudflared on the backend machine: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
2. Run `cloudflared tunnel login`. This step opens a browser window to log in with your Cloudflare account.
3. Run `cloudflared tunnel create genai-api`. This step generates a tunnel ID, note this for next steps.
4. Create a config file ~/.cloudflared/config.yml:
```
tunnel: <your-tunnel-id>
credentials-file: /home/user/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: <your-api-url>
    service: http://localhost:8000
  - service: http_status:404
```
The `<your-api-url>` should be the domain name you have registered with Cloudflare's nameservers, and can be a subdomain. For example, if you own `example.com`, you could use `genai.example.com` or `api.example.com`.
5. Create DNS record `cloudflared tunnel route dns <tunnel-id> <your-api-url>`. This creates the tunnel for Cloudflare to route traffic from the internet to your server without exposing a port or your local IP.

## Local Frontend

In case you want to run the frontend locally, follow the instructions below.

1. Setup app environment
```
npm install -D @shadcn/ui
npx shadcn@latest init
npx shadcn@latest add alert button card input textarea
npm install lucide-react
npm install -D @tailwindcss/typography
npm install clsx tailwind-merge
npm install
npm install sharp
```

2. Running the local frontend
```
npm run dev
```


