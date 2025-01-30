import runpod
from diffusers import FluxPipeline
from huggingface_hub import login, HfFolder
import torch
from io import BytesIO
import os
import base64

# # Adjustable paramaters for model configuration

# How closely the model follows the prompt
model_guidance_scale = 3.5
# Height in pixels of the generated image
model_height = 768
# Width in pixels of the generated image
model_width = 1360
# Number of times a model attempts to improve its prediction of an image. Higher is better but slower to generate
model_num_inference_steps = 50


# Get the Hugging Face access token from an environment variable for access to restricted models
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
        cache_dir="/tmp/huggingface",  # Explicit cache directory
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
            "error": "No prompt provided for image generation."
            }

    try:
        output = model(
            prompt=prompt,
            guidance_scale=model_guidance_scale,
            height=model_height,
            width=model_width,
            num_inference_steps=model_num_inference_steps,
        )
        pil_image = output.images[0]

        # Convert pil_image to Base64 to be returned in the API response
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
