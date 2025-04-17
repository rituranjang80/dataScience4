# Requires diffusers package: pip install diffusers transformers torch

from diffusers import StableDiffusionPipeline
import torch

def generate_diffusion_images(prompt, num_images=1):
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16
    ).to("cuda")
    
    images = pipe(
        prompt=prompt,
        num_images_per_prompt=num_images,
        guidance_scale=7.5
    ).images
    
    for i, img in enumerate(images):
        img.save(f"generated_mango_{i}.png")
    
    return images

# Example usage
generate_diffusion_images(
    prompt="A healthy mango leaf on white background, high resolution, realistic",
    num_images=3
)