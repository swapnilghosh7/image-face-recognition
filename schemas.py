from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Request/Response for Persons
class PersonCreate(BaseModel):
    name: str

class PersonResponse(PersonCreate):
    id: int
    class Config:
        from_attributes = True

# Response for Face Records
class FaceRecordResponse(BaseModel):
    id: int
    file_path: str  # Original image path
    person_id: Optional[int]
    person_name: Optional[str] = None
    face_image_url: str  # URL to cropped face image
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True