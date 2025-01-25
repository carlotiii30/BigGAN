from diffusers import StableDiffusionPipeline
import torch

# Ruta corregida al modelo
model_path = "./data/stable/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"

# Cargar el modelo desde la ubicación correcta
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)

# Mover el modelo a CPU
pipe.to("cpu")

# Texto de entrada para la generación de la imagen
prompt = "A beautiful landscape with mountains and a river"

# Generar la imagen
print("Generando imagen, esto puede tardar un poco...")
image = pipe(prompt).images[0]

# Guardar la imagen generada
image.save("output_image.png")

print("Imagen generada y guardada como 'output_image.png'")
