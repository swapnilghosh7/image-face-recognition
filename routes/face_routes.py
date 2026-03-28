"""
Face routes - handles face listing, assignment, and deletion
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

import database
from services.face_service import delete_cropped_face

router = APIRouter(prefix="/faces", tags=["Faces"])


@router.get("")
def list_faces(project_name: Optional[str] = None, db: Session = Depends(database.get_db)):
    """List all faces, optionally filtered by project"""
    from models import FaceRecord, Project
    
    query = db.query(FaceRecord)
    if project_name:
        project = db.query(Project).filter(Project.name == project_name).first()
        if project:
            query = query.filter(FaceRecord.project_id == project.id)
    
    results = []
    for rec in query.all():
        # Get project name for URL
        project = db.query(Project).filter(Project.id == rec.project_id).first()
        project_name_str = project.name if project else "unknown"
        
        results.append({
            "id": rec.id,
            "file_path": rec.file_path,
            "person_id": rec.person_id,
            "person_name": rec.person.name if rec.person else "Unknown",
            "face_image_url": f"/images/faces/{project_name_str}/{rec.id}.jpg",
            "created_at": rec.created_at.isoformat() if rec.created_at else None
        })
    
    return results


@router.patch("/{face_id}/assign-person")
def assign_person_to_face(
    face_id: int,
    person_id: Optional[int],
    db: Session = Depends(database.get_db)
):
    """Manually assign a person to a face record"""
    from models import FaceRecord, Person
    
    face_record = db.query(FaceRecord).filter(FaceRecord.id == face_id).first()
    if not face_record:
        raise HTTPException(status_code=404, detail="Face record not found")
    
    if person_id:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
    
    face_record.person_id = person_id
    db.commit()
    
    return {"message": "Person assigned successfully", "face_id": face_id, "person_id": person_id}


@router.delete("/{face_id}")
def delete_face(face_id: int, db: Session = Depends(database.get_db)):
    """Delete a single face record and its cropped image"""
    from models import FaceRecord, Project
    
    face_record = db.query(FaceRecord).filter(FaceRecord.id == face_id).first()
    if not face_record:
        raise HTTPException(status_code=404, detail="Face record not found")
    
    # Get project name for file cleanup
    project = db.query(Project).filter(Project.id == face_record.project_id).first()
    project_name = project.name if project else f"project_{face_record.project_id}"
    
    # Delete cropped face file
    delete_cropped_face(project_name, face_id)
    
    # Delete database record
    db.delete(face_record)
    db.commit()
    
    return {"message": f"Face {face_id} deleted successfully"}
