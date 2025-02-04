from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import torch
from diffusers import StableDiffusionPipeline
import uuid

router = APIRouter()

OUTPUT_DIR = "./outputs"
MODEL_DIR = "./models"
DEFAULT_MODEL_NAME = "stable_modified"


class GenerateRequest(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


@router.post("/generate/")
async def generate_image(request: GenerateRequest):
    model_path = os.path.join(MODEL_DIR, request.model_name)

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
        request.prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
    ).images[0]

    # Generar un nombre único para la imagen
    unique_image_name = f"{uuid.uuid4()}.png"
    output_path = os.path.join(OUTPUT_DIR, unique_image_name)
    image.save(output_path)

    return {"message": "Imagen generada exitosamente", "image_path": output_path}


@router.get("/download/{image_name}")
async def download_image(image_name: str):
    image_path = os.path.join(OUTPUT_DIR, image_name)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    return FileResponse(image_path)
