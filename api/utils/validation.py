import os
from fastapi import HTTPException

EXPECTED_STRUCTURE = {
    "feature_extractor": ["preprocessor_config.json"],
    "safety_checker": ["config.json", "model.safetensors"],
    "scheduler": ["scheduler_config.json"],
    "text_encoder": ["config.json", "model.safetensors"],
    "tokenizer": [
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
    "unet": ["config.json", "diffusion_pytorch_model.safetensors"],
    "vae": ["config.json", "diffusion_pytorch_model.safetensors"],
    "model_index.json": [],
}


def validate_model_structure(model_path: str):
    for dir_name, files in EXPECTED_STRUCTURE.items():
        dir_path = os.path.join(model_path, dir_name)
        if not os.path.exists(dir_path):
            raise HTTPException(
                status_code=400, detail=f"Missing directory: {dir_name}"
            )
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing file: {file_name} in directory {dir_name}",
                )
