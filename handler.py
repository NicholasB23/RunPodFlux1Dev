import runpod
from diffusers import FluxPipeline
from huggingface_hub import login, HfFolder
import torch
from io import BytesIO
import os
import base64

HF_TOKEN = os.environ["HUGGING_FACE_HUB_TOKEN"]
login(token=HF_TOKEN)

# Verify token is set
if not HfFolder.get_token():
    raise EnvironmentError("Hugging Face token not found. Please check authentication.")


def load_model():
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and move it to GPU
    model = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,  # Using float16 for better GPU performance
        device_map="balanced",  # Auto changed to balanced for this model
        cache_dir="/tmp/huggingface",  # Explicit cache directory
        resume_download=True,  # Resume interrupted downloads
        local_files_only=False  # Force new download if needed
    )

    return model

def generate_image_from_prompt(event):
    global model

    # Ensure the model is loaded
    if "model" not in globals():
        model = load_model()

    # Get the input text from the event
    prompt = event["input"].get("prompt")

    # Validate input
    if not prompt:
        return {
            "statusCode": 400,
            "error": "No text provided for analysis."
            }

    try:
        output = model(
            prompt=prompt,
            guidance_scale=3.5,
            height=768,
            width=1360,
            num_inference_steps=50,
        )
        pil_image = output.images[0]

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "statusCode": 200,
            "output": {
                "image": img_str,
                "type": "base64"
            }
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "output": {
                "error": str(e)
            }
        }


runpod.serverless.start({"handler": generate_image_from_prompt})
