from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil
import zipfile
from api.utils.validation import validate_model_structure

router = APIRouter()

MODEL_DIR = "./models"


@router.post("/upload/")
async def upload_model(model_zip: UploadFile = File(...)):
    model_zip_path = os.path.join(MODEL_DIR, model_zip.filename)

    # Guardar el archivo ZIP subido
    with open(model_zip_path, "wb") as buffer:
        shutil.copyfileobj(model_zip.file, buffer)

    # Descomprimir el archivo ZIP
    with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    # Obtener el nombre del directorio descomprimido
    model_dir_name = model_zip.filename.replace(".zip", "")
    model_path = os.path.join(MODEL_DIR, model_dir_name)

    # Validar la estructura del modelo
    validate_model_structure(model_path)

    # Eliminar el archivo ZIP después de descomprimir
    os.remove(model_zip_path)

    return {"filename": model_zip.filename, "message": "Modelo subido exitosamente"}


@router.delete("/delete/{model_name}")
async def delete_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    shutil.rmtree(model_path)
    return {"message": f"Modelo '{model_name}' eliminado correctamente"}


@router.get("/list/")
def list_models():
    models = os.listdir(MODEL_DIR)
    return {"models": models}
