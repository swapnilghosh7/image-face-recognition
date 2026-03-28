"""
Face processing service - handles face detection, cropping, matching, and merging
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

import models


# Initialize FaceAnalysis (singleton)
_face_app = None

def get_face_analyzer() -> FaceAnalysis:
    """Get or create the face analyzer instance"""
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def crop_and_save_face(img: np.ndarray, face: any, record_id: int, project_name: str) -> str:
    """
    Crop face from image with dynamic size maintaining 2:3 aspect ratio.
    Face is centered with padding, then resized proportionally.
    Saves to faces/{project_name}/{id}.jpg
    """
    ASPECT_RATIO = 2 / 3
    
    # InsightFace bbox format: [x1, y1, x2, y2]
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Calculate face dimensions
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Add padding (20% of face size, minimum 10px)
    padding_x = max(int(face_width * 0.2), 10)
    padding_y = max(int(face_height * 0.2), 10)
    
    x1_padded = max(0, x1 - padding_x)
    y1_padded = max(0, y1 - padding_y)
    x2_padded = min(img.shape[1], x2 + padding_x)
    y2_padded = min(img.shape[0], y2 + padding_y)
    
    # Calculate crop dimensions maintaining 2:3 aspect ratio
    crop_width = x2_padded - x1_padded
    crop_height = y2_padded - y1_padded
    
    # Adjust to maintain 2:3 aspect ratio
    current_ratio = crop_width / crop_height if crop_height > 0 else ASPECT_RATIO
    
    if current_ratio > ASPECT_RATIO:
        crop_width = int(crop_height * ASPECT_RATIO)
    else:
        crop_height = int(crop_width / ASPECT_RATIO)
    
    # Center the crop on the face
    face_center_x = (x1_padded + x2_padded) // 2
    face_center_y = (y1_padded + y2_padded) // 2
    
    half_w = crop_width // 2
    half_h = crop_height // 2
    
    crop_x1 = face_center_x - half_w
    crop_y1 = face_center_y - half_h
    crop_x2 = face_center_x + half_w
    crop_y2 = face_center_y + half_h
    
    # Adjust if crop goes outside image bounds
    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > img.shape[1]:
        crop_x1 -= (crop_x2 - img.shape[1])
        crop_x2 = img.shape[1]
    if crop_y2 > img.shape[0]:
        crop_y1 -= (crop_y2 - img.shape[0])
        crop_y2 = img.shape[0]
    
    # Ensure dimensions are valid
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img.shape[1], max(crop_x1 + 1, crop_x2))
    crop_y2 = min(img.shape[0], max(crop_y1 + 1, crop_y2))
    
    # Crop face region
    face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Resize to standard size
    OUTPUT_WIDTH = 200
    OUTPUT_HEIGHT = 300
    face_crop_resized = cv2.resize(face_crop, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
    
    # Create faces directory structure
    faces_dir = os.path.join("faces", project_name)
    os.makedirs(faces_dir, exist_ok=True)
    
    # Save cropped face
    face_path = os.path.join(faces_dir, f"{record_id}.jpg")
    abs_face_path = os.path.abspath(face_path)
    
    # Convert BGR to RGB for PIL
    face_crop_rgb = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
    face_img = Image.fromarray(face_crop_rgb)
    face_img.save(face_path, "JPEG", quality=95)
    
    print(f"  ✓ Saved cropped face: {abs_face_path} ({OUTPUT_WIDTH}x{OUTPUT_HEIGHT}) [face:{face_width}x{face_height}, crop:{crop_width}x{crop_height}]")
    return face_path


def delete_cropped_face(project_name: str, record_id: int):
    """Delete cropped face image from disk"""
    face_path = os.path.join("faces", project_name, f"{record_id}.jpg")
    if os.path.exists(face_path):
        try:
            os.remove(face_path)
            print(f"  ✓ Deleted cropped face: {face_path}")
        except Exception as e:
            print(f"  ⚠ Error deleting {face_path}: {e}")


def find_matching_person(embedding_list: list, db: Session, project_id: int = None) -> Optional[int]:
    """
    Compare new face embedding against all existing faces in DB.
    Returns person_id if match found, else None.
    """
    MATCH_THRESHOLD = 0.5  # Adjust based on testing
    
    # Get all processed faces with embeddings
    query = db.query(models.FaceRecord).filter(
        models.FaceRecord.person_id != None,
        models.FaceRecord.embedding != None
    )
    
    if project_id:
        query = query.filter(models.FaceRecord.project_id == project_id)
    
    existing_faces = query.all()
    
    if not existing_faces:
        return None
    
    # Convert new embedding to numpy array
    new_embedding = np.array(embedding_list).reshape(1, -1)
    
    # Check against each existing face
    for existing_face in existing_faces:
        try:
            existing_embedding = np.array(json.loads(existing_face.embedding)).reshape(1, -1)
            similarity = cosine_similarity(new_embedding, existing_embedding)[0][0]
            
            if similarity >= (1.0 - MATCH_THRESHOLD):
                return existing_face.person_id
        except Exception as e:
            print(f"Error comparing embeddings: {e}")
            continue
    
    return None


def process_faces_in_image(
    img: np.ndarray,
    image_path: str,
    file_hash: str,
    project_id: int,
    project_name: str,
    db: Session
) -> Tuple[int, int, int]:
    """
    Process all faces in a single image.
    Returns: (processed_count, matched_count, new_person_count)
    """
    face_app = get_face_analyzer()
    faces = face_app.get(img)
    
    if not faces:
        return 0, 0, 0
    
    matched_count = 0
    new_person_count = 0
    
    for face in faces:
        embedding_list = face.embedding.tolist()
        embedding_json = json.dumps(embedding_list)
        
        # Find matching person
        matched_person_id = find_matching_person(embedding_list, db, project_id)
        
        if matched_person_id:
            person_id = matched_person_id
            matched_count += 1
            print(f"  ✓ Matched existing person ID: {person_id}")
        else:
            person_id = None
            new_person_count += 1
            print(f"  ? New unknown face detected")
        
        # Save Face Record
        new_record = models.FaceRecord(
            project_id=project_id,
            file_path=image_path,
            file_hash=file_hash,
            image_url=image_path,  # Temporary, will be updated after cropping
            embedding=embedding_json,
            person_id=person_id
        )
        db.add(new_record)
        db.commit()  # Commit to get the record ID
        
        # Crop and save face
        try:
            face_path = crop_and_save_face(img, face, new_record.id, project_name)
            new_record.image_url = face_path  # Update to point to cropped face
            db.commit()
        except Exception as crop_error:
            print(f"  ⚠ Error cropping face: {crop_error}")
    
    return len(faces), matched_count, new_person_count


def merge_duplicate_faces(project_id: int, db: Session):
    """
    Find duplicate faces (same person appearing multiple times) and merge them.
    Uses embedding similarity to group faces of the same person.
    Keeps only ONE face record per unique person, deletes duplicates.
    """
    # Get project name for cleanup
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    project_name = project.name if project else f"project_{project_id}"
    
    # Get all face records for this project with embeddings
    faces = db.query(models.FaceRecord).filter(
        models.FaceRecord.project_id == project_id,
        models.FaceRecord.embedding != None
    ).all()
    
    if len(faces) < 2:
        print("Not enough faces to merge")
        return
    
    # Group faces by similarity
    face_groups = []
    processed_face_ids = set()
    
    for i, face_a in enumerate(faces):
        if face_a.id in processed_face_ids:
            continue
        
        current_group = [face_a.id]
        processed_face_ids.add(face_a.id)
        
        try:
            embedding_a = np.array(json.loads(face_a.embedding)).reshape(1, -1)
        except Exception as e:
            print(f"Error loading embedding for face {face_a.id}: {e}")
            continue
        
        # Compare with all remaining faces
        for face_b in faces[i+1:]:
            if face_b.id in processed_face_ids:
                continue
            
            try:
                embedding_b = np.array(json.loads(face_b.embedding)).reshape(1, -1)
                similarity = cosine_similarity(embedding_a, embedding_b)[0][0]
                
                if similarity >= 0.65:
                    current_group.append(face_b.id)
                    processed_face_ids.add(face_b.id)
            except Exception as e:
                print(f"Error comparing embeddings: {e}")
                continue
        
        face_groups.append(current_group)
    
    # Merge faces in each group - KEEP ONLY ONE record per group
    merged_groups = 0
    total_faces_deleted = 0
    
    for group in face_groups:
        if len(group) <= 1:
            continue
        
        # Find if any face in this group already has a person_id
        existing_person_id = None
        for face_id in group:
            face = db.query(models.FaceRecord).filter(models.FaceRecord.id == face_id).first()
            if face and face.person_id:
                existing_person_id = face.person_id
                break
        
        # If no existing person, create a new one
        if not existing_person_id:
            new_person = models.Person(name=f"Person_{group[0]}")
            db.add(new_person)
            db.commit()
            existing_person_id = new_person.id
            print(f"  Created new person ID {existing_person_id} for group {group}")
        
        # Keep the first face record, delete the rest
        face_to_keep_id = group[0]
        
        for face_id in group[1:]:
            face_to_delete = db.query(models.FaceRecord).filter(models.FaceRecord.id == face_id).first()
            if face_to_delete:
                delete_cropped_face(project_name, face_id)
                db.delete(face_to_delete)
                db.commit()
                total_faces_deleted += 1
                print(f"  Deleted duplicate face record ID: {face_id}")
        
        # Update the kept face record with the person_id
        face_to_keep = db.query(models.FaceRecord).filter(models.FaceRecord.id == face_to_keep_id).first()
        if face_to_keep:
            old_person_id = face_to_keep.person_id
            face_to_keep.person_id = existing_person_id
            db.commit()
            
            # Delete old person if no faces reference it anymore
            if old_person_id and old_person_id != existing_person_id:
                remaining = db.query(models.FaceRecord).filter(
                    models.FaceRecord.person_id == old_person_id
                ).count()
                if remaining == 0:
                    old_person = db.query(models.Person).filter(models.Person.id == old_person_id).first()
                    if old_person:
                        db.delete(old_person)
                        db.commit()
        
        merged_groups += 1
    
    print(f"Deleted {total_faces_deleted} duplicate faces, kept {merged_groups} unique persons")


def process_new_faces_background(project_id: int, source_path: str, storage_type: str, db: Session, auto_merge: bool = False):
    """
    Background worker to find and process new images.
    This is called by the scan routes as a background task.
    """
    from utils.file_utils import calculate_file_hash, list_images_local
    
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
    
    # Get project name for saving cropped faces
    from models import Project
    project = db.query(Project).filter(Project.id == project_id).first()
    project_name = project.name if project else f"project_{project_id}"
    
    processed_count = 0
    skipped_count = 0
    matched_count = 0
    new_person_count = 0
    cropped_count = 0
    
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
            
            # Use the refactored process_faces_in_image function
            face_count, matched, new_faces = process_faces_in_image(
                img, full_path, file_hash, project_id, project_name, db
            )
            
            processed_count += face_count
            matched_count += matched
            new_person_count += new_faces
            cropped_count += face_count
            
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            db.rollback()
    
    print(f"=== Scan Complete ===")
    print(f"Processed: {processed_count} faces")
    print(f"Matched: {matched_count} faces")
    print(f"New Unknown: {new_person_count} faces")
    print(f"Skipped: {skipped_count} files")
    print(f"Cropped faces saved: {cropped_count}")
    
    # Auto-merge if requested
    if auto_merge:
        print("\n=== Starting Auto-Merge ===")
        merge_duplicate_faces(project_id, db)
