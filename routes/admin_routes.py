"""
Admin routes - handles database reset and project cleanup
"""
import os
import shutil
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import database
from services.face_service import delete_cropped_face

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/reset")
def reset_database(db: Session = Depends(database.get_db)):
    """
    ⚠️ DANGER: Deletes ALL data from the database
    Use this to start fresh
    """
    from models import FaceRecord, Person, Project
    
    # Delete all face records and their cropped images
    faces = db.query(FaceRecord).all()
    for face in faces:
        project = db.query(Project).filter(Project.id == face.project_id).first()
        if project:
            delete_cropped_face(project.name, face.id)
        db.delete(face)
    
    # Delete all persons
    db.query(Person).delete()
    
    # Delete all projects
    db.query(Project).delete()
    
    db.commit()
    
    # Optionally delete faces folder
    if os.path.exists("faces"):
        shutil.rmtree("faces")
        print("Deleted faces/ folder")
    
    return {"message": "Database reset successfully. All data deleted."}


@router.delete("/clear-project/{project_name}")
def clear_project(project_name: str, db: Session = Depends(database.get_db)):
    """
    Delete all data for a specific project
    """
    from models import Project, FaceRecord
    
    project = db.query(Project).filter(Project.name == project_name).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Delete all faces for this project
    faces = db.query(FaceRecord).filter(FaceRecord.project_id == project.id).all()
    for face in faces:
        delete_cropped_face(project_name, face.id)
        db.delete(face)
    
    # Delete persons associated only with this project
    # (This is simplified - in production you'd check associations more carefully)
    
    db.delete(project)
    db.commit()
    
    # Delete project's face folder
    project_faces_dir = os.path.join("faces", project_name)
    if os.path.exists(project_faces_dir):
        shutil.rmtree(project_faces_dir)
        print(f"Deleted {project_faces_dir}")
    
    return {"message": f"Project '{project_name}' and all its data deleted successfully"}
