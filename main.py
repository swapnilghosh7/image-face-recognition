"""
Face Recognition API - Main Entry Point

This is the main FastAPI application that wires together all routes and services.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import database
from routes import (
    scan_router,
    face_router,
    person_router,
    admin_router,
    image_router
)

# Initialize DB
database.Base.metadata.create_all(bind=database.engine)

# Create FastAPI app
app = FastAPI(
    title="Universal Face Scanner API",
    description="Face recognition and management API with automatic face detection, cropping, and person matching",
    version="2.0.0"
)

# Enable CORS (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(scan_router)
app.include_router(face_router)
app.include_router(person_router)
app.include_router(admin_router)
app.include_router(image_router)


# Health check endpoint
@app.get("/")
def root():
    """Root endpoint - API health check"""
    return {
        "status": "ok",
        "message": "Face Recognition API is running",
        "docs": "/docs"
    }


# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
