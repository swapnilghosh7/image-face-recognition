"""
Person routes - handles person listing, creation, merging, and deletion
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

import database
from services.face_service import delete_cropped_face

router = APIRouter(prefix="/persons", tags=["Persons"])


@router.get("")
def list_persons(project_name: Optional[str] = None, db: Session = Depends(database.get_db)):
    """List all persons, optionally filtered by project"""
    from models import Person, FaceRecord, Project
    
    if project_name:
        project = db.query(Project).filter(Project.name == project_name).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get all persons who have faces in this project
        persons = db.query(Person).join(FaceRecord).filter(
            FaceRecord.project_id == project.id
        ).distinct().all()
    else:
        persons = db.query(Person).all()
    
    results = []
    for person in persons:
        # Count how many faces this person has
        face_count = db.query(FaceRecord).filter(
            FaceRecord.person_id == person.id
        ).count()
        
        # Get one representative face image
        representative_face = db.query(FaceRecord).filter(
            FaceRecord.person_id == person.id
        ).first()
        
        # Get project name for URL
        project_for_face = None
        if representative_face:
            project_for_face = db.query(Project).filter(
                Project.id == representative_face.project_id
            ).first()
        project_name_str = project_for_face.name if project_for_face else "unknown"
        
        results.append({
            "id": person.id,
            "name": person.name,
            "face_count": face_count,
            "representative_image_url": f"/images/faces/{project_name_str}/{representative_face.id}.jpg" if representative_face else None
        })
    
    return results


@router.get("/{person_id}")
def get_person_details(person_id: int, db: Session = Depends(database.get_db)):
    """Get person details with all their face images"""
    from models import Person, FaceRecord, Project
    
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Get all faces for this person
    faces = db.query(FaceRecord).filter(
        FaceRecord.person_id == person_id
    ).all()
    
    face_list = []
    for face in faces:
        # Get project name for URL
        project = db.query(Project).filter(Project.id == face.project_id).first()
        project_name_str = project.name if project else "unknown"
        
        face_list.append({
            "id": face.id,
            "file_path": face.file_path,
            "face_image_url": f"/images/faces/{project_name_str}/{face.id}.jpg",
            "project_id": face.project_id
        })
    
    return {
        "id": person.id,
        "name": person.name,
        "face_count": len(face_list),
        "faces": face_list
    }


@router.post("")
def create_person(name: str, db: Session = Depends(database.get_db)):
    """Create a new person"""
    from models import Person
    
    existing = db.query(Person).filter(Person.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Person with this name already exists")
    
    person = Person(name=name)
    db.add(person)
    db.commit()
    db.refresh(person)
    
    return {"id": person.id, "name": person.name}


@router.post("/merge")
def merge_persons(
    person_id_keep: int,
    person_id_remove: int,
    db: Session = Depends(database.get_db)
):
    """
    Merge two persons (e.g., if 'Unknown_1' and 'Unknown_2' are actually the same person)
    All faces from person_id_remove will be moved to person_id_keep
    """
    from models import Person, FaceRecord
    
    person_keep = db.query(Person).filter(Person.id == person_id_keep).first()
    person_remove = db.query(Person).filter(Person.id == person_id_remove).first()
    
    if not person_keep or not person_remove:
        raise HTTPException(status_code=404, detail="One or both persons not found")
    
    # Update all faces from remove to keep
    db.query(FaceRecord).filter(
        FaceRecord.person_id == person_id_remove
    ).update({"person_id": person_id_keep})
    
    # Delete the removed person
    db.delete(person_remove)
    db.commit()
    
    return {"message": f"Merged person {person_id_remove} into {person_id_keep}"}


@router.delete("/{person_id}")
def delete_person(person_id: int, db: Session = Depends(database.get_db)):
    """Delete a person and ALL their face records (including cropped images from disk)"""
    from models import Person, FaceRecord, Project
    
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Get all faces for this person to delete cropped files
    faces = db.query(FaceRecord).filter(
        FaceRecord.person_id == person_id
    ).all()
    
    # Get project name for file cleanup
    if faces:
        project = db.query(Project).filter(Project.id == faces[0].project_id).first()
        project_name = project.name if project else f"project_{faces[0].project_id}"
        
        # Delete cropped face files and database records
        for face in faces:
            delete_cropped_face(project_name, face.id)
            db.delete(face)
        
        db.commit()
        print(f"Deleted person {person.name} and {len(faces)} face(s)")
    
    db.delete(person)
    db.commit()
    
    return {"message": f"Person {person.name} deleted, faces marked as unknown"}
