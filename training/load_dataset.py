import os
import tarfile
import requests
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# URL del dataset
dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
dataset_dir = "./data/stanford_dogs"

# Crear directorio para el dataset
os.makedirs(dataset_dir, exist_ok=True)

# Descargar el dataset
tar_path = os.path.join(dataset_dir, "images.tar")
if not os.path.exists(tar_path):
    print("Descargando el dataset...")
    response = requests.get(dataset_url, stream=True)
    with open(tar_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Descarga completada.")

# Extraer el dataset
if not os.path.exists(os.path.join(dataset_dir, "Images")):
    print("Extrayendo el dataset...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dataset_dir)
    print("Extracción completada.")

# Clase personalizada para el dataset
class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Recorrer las carpetas y agregar imágenes y etiquetas
        for breed_idx, breed in enumerate(os.listdir(self.root_dir)):
            breed_dir = os.path.join(self.root_dir, breed)
            if os.path.isdir(breed_dir):
                for img_name in os.listdir(breed_dir):
                    self.image_paths.append(os.path.join(breed_dir, img_name))
                    self.labels.append(breed_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label}

# Transformaciones para el dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionar imágenes
    transforms.ToTensor(),         # Convertir a tensores
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalizar
])

# Crear el dataset y DataLoader
dataset = StanfordDogsDataset(os.path.join(dataset_dir, "Images"), transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Total de imágenes en el dataset: {len(dataset)}")
