from fastapi import APIRouter
from pydantic import BaseModel

from api.utils.image_operations import download_image, generate_image

router = APIRouter()

DEFAULT_MODEL_NAME = "stable_modified"


class GenerateRequest(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5


@router.post("/generate/")
async def generate_image_endpoint(request: GenerateRequest):
    image_path = generate_image(
        model_name=request.model_name,
        prompt=request.prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
    )
    return {"message": "Imagen generada exitosamente", "image_path": image_path}


@router.get("/download/{image_name}")
async def download_image_endpoint(image_name: str):
    return download_image(image_name)
