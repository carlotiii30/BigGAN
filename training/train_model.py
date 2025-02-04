# Importar librerías necesarias
import os
import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.onnx

import nltk
nltk.download("wordnet")

# Configuración de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Cargar BigGAN preentrenado
model = BigGAN.from_pretrained("biggan-deep-128").to(device)

# Congelar capas iniciales
for param in model.parameters():
    param.requires_grad = False

# Permitir entrenamiento en capas específicas
for param in model.generator.parameters():
    param.requires_grad = True

# Dataset personalizado
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []

        # Recorrer recursivamente los subdirectorios
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Configurar dataset y DataLoader
dataset = CustomDataset("data/stanford_dogs/Images", transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Optimización
optimizer = AdamW(model.generator.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# Entrenamiento
model.train()
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = batch.to(device)

        # Generar entrada de ruido
        noise = torch.tensor(truncated_noise_sample(batch_size=batch.size(0), dim_z=128, truncation=0.4), dtype=torch.float).to(device)
        class_labels = one_hot_from_names(["golden retriever"] * batch.size(0), batch_size=batch.size(0))
        class_labels = torch.tensor(class_labels).to(device)


        # Forward y pérdida
        output = model(noise, class_labels, truncation=0.4)
        loss = criterion(output, batch)

        # Backprop y actualización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Loss en epoch {epoch+1}: {epoch_loss / len(dataloader):.4f}")

# Exportar a ONNX
onnx_path = "biggan_finetuned.onnx"
dummy_noise = torch.tensor(truncated_noise_sample(batch_size=1, dim_z=128, truncation=0.4), dtype=torch.float)
dummy_class = one_hot_from_names(["golden retriever"], batch_size=1)

model.eval()
torch.onnx.export(
    model,
    (dummy_noise.to(device), dummy_class.to(device), torch.tensor(0.4)),
    onnx_path,
    input_names=["noise", "class", "truncation"],
    output_names=["output"],
    dynamic_axes={"noise": {0: "batch_size"}, "class": {0: "batch_size"}},
    opset_version=14
)

print(f"Modelo exportado a ONNX en: {onnx_path}")
