"""
Utility functions for file operations
"""
import os
import hashlib
from typing import List
from fastapi import HTTPException
from fastapi.responses import StreamingResponse


def calculate_file_hash(file_path: str) -> str:
    """Calculates MD5 hash to detect file changes"""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return "unknown"


def list_images_local(base_path: str) -> List[str]:
    """Recursively finds all images in a local folder"""
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = []
    
    if not os.path.exists(base_path):
        raise HTTPException(status_code=404, detail="Path not found")

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                images.append(full_path)
    
    return images


def get_image_stream_local(file_path: str):
    """Streams a local file"""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return StreamingResponse(open(file_path, "rb"), media_type="image/jpeg")
