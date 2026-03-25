from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean, DateTime, Text, Table
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Project(Base):
    """Represents a specific folder/source being scanned"""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True) # e.g., "Client_A_Drive"
    source_path = Column(String, nullable=False)   # e.g., "s3://bucket/folder" or "/local/path"
    storage_type = Column(String, default="local") # "local", "s3", "gdrive"
    
    faces = relationship("FaceRecord", back_populates="project")

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    projects = relationship("Project", secondary="person_project_association") # Many-to-many if needed
    faces = relationship("FaceRecord", back_populates="person")

# Association table for Many-to-Many (Optional, if one person spans multiple projects)
person_project_association = Table('person_project_association', Base.metadata,
    Column('person_id', Integer, ForeignKey('persons.id')),
    Column('project_id', Integer, ForeignKey('projects.id'))
)

class FaceRecord(Base):
    __tablename__ = "face_records"

    id = Column(Integer, primary_key=True, index=True)
    
    # Project Link
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    project = relationship("Project", back_populates="faces")
    
    # File Identification (Crucial for skipping processed files)
    file_path = Column(String, nullable=False, index=True) # Relative or Full path
    file_hash = Column(String, index=True) # MD5/SHA256 to detect changes
    
    # Image Access
    image_url = Column(String, nullable=False) # Direct URL or Path to stream
    
    # AI Data
    embedding = Column(Text, nullable=False) # JSON string of the vector
    
    # Person Link (Null if unknown)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    person = relationship("Person", back_populates="faces")
    
    is_processed = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)