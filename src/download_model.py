from diffusers import StableDiffusionPipeline

# Nombre del modelo en Hugging Face
model_name = "CompVis/stable-diffusion-v1-4"

# Descargar nuevamente el modelo con la ruta especificada
pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    cache_dir="./data/stable",
    force_download=True,
    local_files_only=False,
)

print("Modelo descargado correctamente.")
