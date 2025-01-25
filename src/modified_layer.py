import os
from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn

model_path = "./data/stable/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
modified_model_path = "./data/stable_modified"

# Cargar el modelo preentrenado
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
pipe.to("cpu")

# Acceder al modelo UNet
unet = pipe.unet

# Guardar los pesos originales de la capa de salida
original_weights = unet.conv_out.weight.clone()
original_bias = unet.conv_out.bias.clone()

# Modificación: cambiar la capa de salida de la UNet asegurando compatibilidad con el modelo original
new_conv_layer = nn.Conv2d(
    in_channels=unet.conv_out.in_channels,  # Mantener el mismo número de canales de entrada
    out_channels=unet.conv_out.out_channels,  # Mantener salida en 4 canales
    kernel_size=3,  # Tamaño de kernel óptimo
    padding=1,
)

# Inicializar la nueva capa con los pesos originales para evitar pérdida de información
new_conv_layer.weight.data = original_weights
new_conv_layer.bias.data = original_bias

# Reemplazar la capa de salida de la UNet
unet.conv_out = new_conv_layer

# Guardar el modelo modificado
pipe.save_pretrained(modified_model_path)
print(f"Modelo modificado guardado en {modified_model_path}")

# Cargar el modelo modificado
pipe = StableDiffusionPipeline.from_pretrained(
    modified_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=False
)
pipe.to("cpu")


# Deshabilitar el safety_checker para evitar bloqueos
def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)


pipe.safety_checker = dummy_safety_checker

# Generar una imagen con el modelo modificado con más pasos y ajuste de escala
prompt = "A beautiful landscape with mountains and a river"
print(f"Generando imagen para el prompt: '{prompt}'")
image = pipe(prompt, num_inference_steps=75, guidance_scale=7.5).images[0]

# Guardar la imagen generada
output_path = "modified_output_image_corrected.png"
image.save(output_path)
print(f"Imagen generada con el modelo modificado guardada como '{output_path}'")
