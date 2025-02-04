import os
import shutil
import zipfile

from fastapi import HTTPException

from api.utils.validation import validate_model_structure

MODEL_DIR = "./models"


def upload_model(model_zip_path: str, model_zip_file) -> str:
    with open(model_zip_path, "wb") as buffer:
        shutil.copyfileobj(model_zip_file, buffer)

    with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    model_dir_name = os.path.basename(model_zip_path).replace(".zip", "")
    model_path = os.path.join(MODEL_DIR, model_dir_name)

    validate_model_structure(model_path)

    os.remove(model_zip_path)

    return model_dir_name


def delete_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    shutil.rmtree(model_path)


def list_models() -> list:
    return os.listdir(MODEL_DIR)
