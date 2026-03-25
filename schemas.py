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
    image_path: str
    person_id: Optional[int]
    person_name: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True