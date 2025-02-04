import os
import uuid
import torch
from diffusers import StableDiffusionPipeline
from fastapi import HTTPException
from fastapi.responses import FileResponse

OUTPUT_DIR = "./outputs"
MODEL_DIR = "./models"


def generate_image(
    model_name: str, prompt: str, num_inference_steps: int, guidance_scale: float
) -> str:
    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    # Cargar el modelo
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float32
    )
    pipe.to("cpu")

    # Deshabilitar el safety_checker con un dummy correcto
    def dummy_safety_checker(images, clip_input):
        return images, [False] * len(images)

    pipe.safety_checker = dummy_safety_checker

    # Generar una imagen
    image = pipe(
        prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
    ).images[0]

    # Generar un nombre único para la imagen
    unique_image_name = f"{uuid.uuid4()}.png"
    output_path = os.path.join(OUTPUT_DIR, unique_image_name)
    image.save(output_path)

    return output_path


def download_image(image_name: str) -> FileResponse:
    image_path = os.path.join(OUTPUT_DIR, image_name)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    return FileResponse(image_path)
