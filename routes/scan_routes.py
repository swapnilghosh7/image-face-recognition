"""
Scan routes - handles scanning folders for faces
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

import database
from services.face_service import process_new_faces_background

router = APIRouter(prefix="/scan", tags=["Scanning"])


class ScanRequest(BaseModel):
    project_name: str
    source_path: str
    storage_type: str = "local"  # local, s3, gdrive


@router.post("")
async def scan_folder(request: ScanRequest, background_tasks: BackgroundTasks, db: Session = Depends(database.get_db)):
    """
    Triggers a scan of a folder/cloud bucket.
    Runs in background to avoid timeout.
    Does NOT merge duplicate faces.
    """
    from models import Project
    
    # Create or Get Project
    project = db.query(Project).filter(Project.name == request.project_name).first()
    if not project:
        project = Project(
            name=request.project_name,
            source_path=request.source_path,
            storage_type=request.storage_type
        )
        db.add(project)
        db.commit()
        db.refresh(project)
    
    # Add Background Task (without auto-merge)
    background_tasks.add_task(
        process_new_faces_background,
        project.id,
        request.source_path,
        request.storage_type,
        db,
        False
    )
    
    return {
        "status": "scanning_started",
        "project_id": project.id,
        "message": "Scanning initiated in background. Check status later."
    }


@router.post("-merge")
async def scan_and_merge(request: ScanRequest, background_tasks: BackgroundTasks, db: Session = Depends(database.get_db)):
    """
    Triggers a scan of a folder/cloud bucket AND automatically merges duplicate faces.
    Runs in background to avoid timeout.
    """
    from models import Project
    
    # Create or Get Project
    project = db.query(Project).filter(Project.name == request.project_name).first()
    if not project:
        project = Project(
            name=request.project_name,
            source_path=request.source_path,
            storage_type=request.storage_type
        )
        db.add(project)
        db.commit()
        db.refresh(project)
    
    # Add Background Task (with auto-merge)
    background_tasks.add_task(
        process_new_faces_background,
        project.id,
        request.source_path,
        request.storage_type,
        db,
        True
    )
    
    return {
        "status": "scanning_and_merging_started",
        "project_id": project.id,
        "message": "Scanning and auto-merging initiated in background. Check status later."
    }
