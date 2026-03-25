from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import os
import json
import cv2
import numpy as np
import hashlib
from typing import List, Optional
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

import database, models, schemas # Ensure you update schemas too if needed

# Initialize DB
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Universal Face Scanner API")

# Initialize AI
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# --- Helper Functions for Storage ---

def calculate_file_hash(file_path: str) -> str:
    """Calculates MD5 hash to detect file changes"""
    # Note: For S3/GDrive, you might use ETag or modified_time instead of reading whole file
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

# TODO: Implement list_images_s3 and get_image_stream_s3 using boto3
# TODO: Implement list_images_gdrive using google-api-python-client

# --- Pydantic Schemas for Request ---

class ScanRequest(BaseModel):
    project_name: str
    source_path: str
    storage_type: str = "local" # local, s3, gdrive

# --- Endpoints ---

@app.post("/scan")
async def scan_folder(request: ScanRequest, background_tasks: BackgroundTasks, db: Session = Depends(database.get_db)):
    """
    Triggers a scan of a folder/cloud bucket.
    Runs in background to avoid timeout.
    """
    
    # 1. Create or Get Project
    project = db.query(models.Project).filter(models.Project.name == request.project_name).first()
    if not project:
        project = models.Project(
            name=request.project_name,
            source_path=request.source_path,
            storage_type=request.storage_type
        )
        db.add(project)
        db.commit()
        db.refresh(project)
    
    # 2. Add Background Task
    background_tasks.add_task(process_new_faces, project.id, request.source_path, request.storage_type, db)
    
    return {
        "status": "scanning_started",
        "project_id": project.id,
        "message": "Scanning initiated in background. Check status later."
    }

# --- Face Matching Helpers ---

MATCH_THRESHOLD = 0.5  # Adjust based on testing (lower = stricter matching)

def find_matching_person(embedding_list: list, db: Session, project_id: int = None) -> Optional[int]:
    """
    Compare new face embedding against all existing faces in DB.
    Returns person_id if match found, else None.
    """
    # Get all processed faces with embeddings (optionally filter by project)
    query = db.query(models.FaceRecord).filter(
        models.FaceRecord.person_id != None,  # Only check faces that are already linked to a person
        models.FaceRecord.embedding != None
    )
    
    if project_id:
        # Optionally match across all projects, or limit to current project
        pass  # Remove this if you want cross-project matching
    
    existing_faces = query.all()
    
    if not existing_faces:
        return None
    
    # Convert new embedding to numpy array
    new_embedding = np.array(embedding_list).reshape(1, -1)
    
    # Check against each existing face
    for existing_face in existing_faces:
        try:
            existing_embedding = np.array(json.loads(existing_face.embedding)).reshape(1, -1)
            
            # Calculate cosine similarity (1.0 = identical, 0.0 = completely different)
            similarity = cosine_similarity(new_embedding, existing_embedding)[0][0]
            
            # Convert similarity to distance (optional, some prefer direct similarity threshold)
            # For cosine similarity, higher is better. Threshold ~0.6+ is usually a match
            if similarity >= (1.0 - MATCH_THRESHOLD):  # e.g., 0.5 threshold means 0.5+ similarity
                return existing_face.person_id
        except Exception as e:
            print(f"Error comparing embeddings: {e}")
            continue
    
    return None

def create_or_get_person(name: str, db: Session) -> int:
    """Create a new person if not exists, return their ID"""
    person = db.query(models.Person).filter(models.Person.name == name).first()
    if not person:
        person = models.Person(name=name)
        db.add(person)
        db.commit()
        db.refresh(person)
    return person.id

def process_new_faces(project_id: int, source_path: str, storage_type: str, db: Session):
    """Background worker to find and process new images WITH matching"""
    print(f"Starting scan for project {project_id} at {source_path}")
    
    if storage_type == "local":
        try:
            image_paths = list_images_local(source_path)
        except Exception as e:
            print(f"Error listing files: {e}")
            return
    else:
        print(f"Storage type {storage_type} not implemented yet.")
        return

    processed_count = 0
    skipped_count = 0
    matched_count = 0
    new_person_count = 0

    for full_path in image_paths:
        # Check if already processed
        file_hash = calculate_file_hash(full_path)
        
        existing = db.query(models.FaceRecord).filter(
            models.FaceRecord.file_path == full_path,
            models.FaceRecord.file_hash == file_hash
        ).first()
        
        if existing:
            skipped_count += 1
            continue
        
        # Process New Image
        try:
            img = cv2.imread(full_path)
            if img is None: 
                continue
            
            faces = face_app.get(img)
            
            for face in faces:
                embedding_list = face.embedding.tolist()
                embedding_json = json.dumps(embedding_list)
                
                # --- MATCHING LOGIC HERE ---
                matched_person_id = find_matching_person(embedding_list, db, project_id)
                
                if matched_person_id:
                    # Face matches an existing person
                    person_id = matched_person_id
                    matched_count += 1
                    print(f"  ✓ Matched existing person ID: {person_id}")
                else:
                    # New unknown person - create a placeholder
                    # Option 1: Leave person_id as NULL (user labels later)
                    # Option 2: Auto-create "Unknown_Person_123"
                    person_id = None  
                    new_person_count += 1
                    print(f"  ? New unknown face detected")
                
                # Save Face Record
                new_record = models.FaceRecord(
                    project_id=project_id,
                    file_path=full_path,
                    file_hash=file_hash,
                    image_url=full_path,
                    embedding=embedding_json,
                    person_id=person_id
                )
                db.add(new_record)
            
            processed_count += len(faces)
            db.commit()
            
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            db.rollback()

    print(f"=== Scan Complete ===")
    print(f"Processed: {processed_count} faces")
    print(f"Matched: {matched_count} faces")
    print(f"New Unknown: {new_person_count} faces")
    print(f"Skipped: {skipped_count} files")

@app.get("/images/{record_id}")
async def get_image(record_id: int, db: Session = Depends(database.get_db)):
    """
    Streams the actual image based on the stored path.
    Works for local files. For S3, this would redirect to a Signed URL.
    """
    record = db.query(models.FaceRecord).filter(models.FaceRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    if record.project.storage_type == "local":
        return get_image_stream_local(record.image_url)
    
    # Logic for S3/GDrive redirection would go here
    raise HTTPException(status_code=501, detail="Streaming for this storage type not implemented yet")

@app.get("/faces")
def list_faces(project_name: Optional[str] = None, db: Session = Depends(database.get_db)):
    query = db.query(models.FaceRecord)
    if project_name:
        project = db.query(models.Project).filter(models.Project.name == project_name).first()
        if project:
            query = query.filter(models.FaceRecord.project_id == project.id)
    
    results = []
    for rec in query.all():
        results.append({
            "id": rec.id,
            "file_path": rec.file_path,
            "person_name": rec.person.name if rec.person else "Unknown",
            "image_preview_url": f"/images/{rec.id}" # Endpoint to view/download
        })
    return results

# For assigning names
# --- Manual Person Assignment Endpoints ---

@app.patch("/faces/{face_id}/assign-person")
def assign_person_to_face(face_id: int, person_id: Optional[int], db: Session = Depends(database.get_db)):
    """Manually assign a person to a face record"""
    face_record = db.query(models.FaceRecord).filter(models.FaceRecord.id == face_id).first()
    if not face_record:
        raise HTTPException(status_code=404, detail="Face record not found")
    
    if person_id:
        person = db.query(models.Person).filter(models.Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
    
    face_record.person_id = person_id
    db.commit()
    
    return {"message": "Person assigned successfully", "face_id": face_id, "person_id": person_id}

#merging two persons (e.g., if 'Unknown_1' and 'Unknown_2' are actually the same person)

@app.post("/persons/merge")
def merge_persons(person_id_keep: int, person_id_remove: int, db: Session = Depends(database.get_db)):
    """
    Merge two persons (e.g., if 'Unknown_1' and 'Unknown_2' are actually the same person)
    All faces from person_id_remove will be moved to person_id_keep
    """
    person_keep = db.query(models.Person).filter(models.Person.id == person_id_keep).first()
    person_remove = db.query(models.Person).filter(models.Person.id == person_id_remove).first()
    
    if not person_keep or not person_remove:
        raise HTTPException(status_code=404, detail="One or both persons not found")
    
    # Update all faces from remove to keep
    db.query(models.FaceRecord).filter(
        models.FaceRecord.person_id == person_id_remove
    ).update({"person_id": person_id_keep})
    
    # Delete the removed person
    db.delete(person_remove)
    db.commit()
    
    return {"message": f"Merged person {person_id_remove} into {person_id_keep}"}

#for deleting a person (marking their faces as unknown)

@app.delete("/persons/{person_id}")
def delete_person(person_id: int, db: Session = Depends(database.get_db)):
    """Delete a person (faces become unknown)"""
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Set all their faces to NULL (unknown)
    db.query(models.FaceRecord).filter(
        models.FaceRecord.person_id == person_id
    ).update({"person_id": None})
    
    db.delete(person)
    db.commit()
    
    return {"message": f"Person {person.name} deleted, faces marked as unknown"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)