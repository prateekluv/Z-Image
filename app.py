"""Gradio UI for Z-Image image generation."""

from __future__ import annotations

import importlib.util
import os
import time
from functools import lru_cache
import gradio as gr
import torch

from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

DEFAULT_PROMPT = (
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
    "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
    "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
    "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if importlib.util.find_spec("torch_xla") is not None:
        return "tpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device_type: str):
    if device_type == "tpu":
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    return device_type


def resolve_dtype(device_type: str) -> torch.dtype:
    if device_type in {"cuda", "tpu"}:
        return torch.bfloat16
    return torch.float32


def available_backends() -> list[str]:
    return [backend.value for backend in AttentionBackend.__members__.values()]


@lru_cache
def load_components(device_type: str, compile_model: bool, attn_backend: str):
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    device = resolve_device(device_type)
    dtype = resolve_dtype(device_type)
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile_model)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    return components


def generate_image(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    compile_model: bool,
    attn_backend: str,
):
    device_type = pick_device()
    components = load_components(device_type, compile_model, attn_backend)
    device = resolve_device(device_type)

    if seed < 0:
        seed = torch.seed()
    generator = torch.Generator(device).manual_seed(int(seed))

    start_time = time.time()
    images = generate(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        **components,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    elapsed = time.time() - start_time
    return images[0], f"Generated in {elapsed:.2f}s on {device_type}."


def build_ui():
    with gr.Blocks(title="Z-Image UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# ⚡️ Z-Image UI\n"
            "Generate images with the Z-Image Turbo checkpoint using the native pipeline."
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    lines=6,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (optional)",
                    value="",
                    lines=2,
                )
                with gr.Row():
                    width = gr.Dropdown(
                        label="Width",
                        choices=[512, 768, 1024],
                        value=1024,
                    )
                    height = gr.Dropdown(
                        label="Height",
                        choices=[512, 768, 1024],
                        value=1024,
                    )
                with gr.Row():
                    steps = gr.Slider(1, 16, value=8, step=1, label="Steps")
                    guidance_scale = gr.Slider(0.0, 12.0, value=0.0, step=0.1, label="Guidance Scale")
                with gr.Row():
                    seed = gr.Number(label="Seed (-1 for random)", value=42, precision=0)
                    compile_model = gr.Checkbox(label="Compile model", value=False)
                attn_backend = gr.Dropdown(
                    label="Attention Backend",
                    choices=available_backends(),
                    value=os.environ.get("ZIMAGE_ATTENTION", "_native_flash"),
                )
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                output = gr.Image(label="Output", type="pil")
                status = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                steps,
                guidance_scale,
                seed,
                compile_model,
                attn_backend,
            ],
            outputs=[output, status],
        )

        gr.Markdown(
            "**Tips**\n"
            "- Guidance scale should remain 0 for Turbo.\n"
            "- Use 512 or 768 if you are running on limited VRAM.\n"
            "- Set seed to -1 for a random result."
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
