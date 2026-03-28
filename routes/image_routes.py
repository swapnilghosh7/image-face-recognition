"""
Image routes - handles serving cropped face images
"""
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

import database

router = APIRouter(prefix="/images", tags=["Images"])


@router.get("/faces/{project_name}/{face_id}")
async def get_face_image(
    project_name: str,
    face_id: int,
    db: Session = Depends(database.get_db)
):
    """
    Direct endpoint to access cropped face image
    URL format: /images/faces/{projectName}/{id}.jpg
    """
    from models import FaceRecord, Project
    
    # Verify the face record exists
    record = db.query(FaceRecord).filter(FaceRecord.id == face_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Face record not found")
    
    # Verify project matches
    project = db.query(Project).filter(Project.id == record.project_id).first()
    if not project or project.name != project_name:
        raise HTTPException(status_code=404, detail="Project mismatch")
    
    # Build path to cropped face image
    face_path = os.path.join("faces", project_name, f"{face_id}.jpg")
    
    if not os.path.exists(face_path):
        raise HTTPException(status_code=404, detail="Cropped face image not found on disk")
    
    return StreamingResponse(open(face_path, "rb"), media_type="image/jpeg")


@router.get("/{record_id}")
async def get_image(record_id: int, db: Session = Depends(database.get_db)):
    """
    Streams the cropped face image from faces/{project_name}/{id}.jpg
    Legacy endpoint - use /images/faces/{project_name}/{face_id} instead
    """
    from models import FaceRecord, Project
    
    record = db.query(FaceRecord).filter(FaceRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Get project name to build the path
    project = db.query(Project).filter(Project.id == record.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Build path to cropped face image
    face_path = os.path.join("faces", project.name, f"{record_id}.jpg")
    
    if not os.path.exists(face_path):
        raise HTTPException(status_code=404, detail="Cropped face image not found on disk")
    
    return StreamingResponse(open(face_path, "rb"), media_type="image/jpeg")
