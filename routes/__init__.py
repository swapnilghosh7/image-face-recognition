from .scan_routes import router as scan_router
from .face_routes import router as face_router
from .person_routes import router as person_router
from .admin_routes import router as admin_router
from .image_routes import router as image_router

__all__ = [
    "scan_router",
    "face_router", 
    "person_router",
    "admin_router",
    "image_router"
]
