from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
import torch

# now 

from datetime import datetime
import os

def model_fn(model_dir):
    print(f"Loading model ({model_dir}) via custom model_fn")

    model = StableDiffusionXLPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        use_safetensors=True,
        device_map="auto",
        variant="fp16",
    )

    model.safety_checker = None

    # TODO: Remove one of these
    # model.enable_attention_slicing()
    # model.enable_xformers_memory_efficient_attention()

    return model


def predict_fn(data, model: StableDiffusionXLPipeline):
    print(f"Running inference with data: {data}")

    prompt = data.get("inputs")
    assert prompt, "Prompt is required"

    base_dir = os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output")


    parameters = data.get("parameters", {})
    width = parameters.get("width", 1024)
    num_inference_steps = parameters.get("num_inference_steps", 30)
    guidance_scale = parameters.get("guidance_scale", 7.5)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    filename = f"sdxl_async_{timestamp}.png"
    output_path = os.path.join(base_dir, filename)
    os.makedirs(base_dir, exist_ok=True)

    image = model(
        prompt,
        height=width,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]  # type: ignore
    image.save(output_path)

    return {"output_path": output_path}


if __name__ == "__main__":
    model = model_fn("stabilityai/stable-diffusion-xl-base-1.0")
    os.environ["SM_OUTPUT_DATA_DIR"] = "./output/"

    data = {
        "inputs": "A beautiful landscape with mountains and a river",
        "parameters": {"width": 1024, "num_inference_steps": 30},
    }
    result = predict_fn(data, model)
    print(result)
