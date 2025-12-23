import os
import gradio as gr
from main import StableDiffusionEngine

# Load Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Stable Diffusion engine
sd_engine = StableDiffusionEngine(hf_token=HF_TOKEN)


def generate(prompt, steps, cfg, width, height):
    """
    Wrapper function for Gradio UI.
    """
    return sd_engine.generate_image(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=width,
        height=height,
    )


examples = [
    "A futuristic African city at sunset, ultra realistic, cinematic lighting",
    "A microscopic view of malaria parasites, scientific illustration",
    "A humanoid AI researcher working in a high-tech laboratory",
]


with gr.Blocks(theme=gr.themes.Soft(), title="Stable Diffusion Research Lab") as demo:
    gr.Markdown(
        """
        # 🧠 Stable Diffusion Research Lab
        **Text-to-Image Generation using SDXL**
        
        This app is designed for **research and experimentation** with Stable Diffusion models.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=4,
            )

            steps = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=1,
                label="Inference Steps",
            )

            cfg = gr.Slider(
                minimum=5.0,
                maximum=9.0,
                value=6.0,
                step=0.5,
                label="Guidance Scale (CFG)",
            )

            width = gr.Dropdown(
                choices=[512, 768, 1024],
                value=768,
                label="Image Width",
            )

            height = gr.Dropdown(
                choices=[512, 768, 1024],
                value=512,
                label="Image Height",
            )

            generate_btn = gr.Button("🚀 Generate Image", variant="primary")

        with gr.Column(scale=1):
            output = gr.Image(
                label="Generated Image",
                type="pil",
            )

    gr.Examples(examples=examples, inputs=prompt)

    generate_btn.click(
        fn=generate,
        inputs=[prompt, steps, cfg, width, height],
        outputs=output,
    )

demo.launch(debug=True)
