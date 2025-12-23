import torch
from diffusers import StableDiffusionXLPipeline


class StableDiffusionEngine:
    """
    Stable Diffusion XL Engine (HF Spaces safe version)
    """

    def __init__(
        self,
        model_id = "stabilityai/stable-diffusion-xl-base-1.0",
        hf_token: str | None = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            safety_checker=True,
            token=hf_token,   # ✅ CRITICAL FIX
        )

        self.pipe.to(self.device)

    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
    ):
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]

        return image
