# image-face-recognition

This project is a backend project in which we can find all the people present in the photos of a specific project.

## Features

- **Face Detection & Recognition**: Automatically detects faces in images and groups them by person
- **Auto-Merge Duplicates**: `/scan-merge` endpoint automatically merges multiple appearances of the same person
- **Face Cropping**: Automatically crops detected faces and saves them as `faces/{project_name}/{id}.jpg`
- **Background Processing**: Long-running tasks run in the background to avoid timeouts
- **Incremental Scanning**: Only processes new images (skips already processed files)
- **Person Management**: Create, merge, delete, and assign persons to faces

## Project Architecture

```
image-face-recognition/
│
├── main.py                    # FastAPI app entry point (routes wiring)
├── models.py                  # Database models (SQLAlchemy)
├── schemas.py                 # Pydantic schemas for validation
├── database.py                # Database connection
├── faceMatchingHelpers.py     # (Legacy - can be removed)
│
├── routes/                    # API Route handlers
│   ├── __init__.py
│   ├── scan_routes.py         # /scan, /scan-merge endpoints
│   ├── face_routes.py         # /faces endpoints
│   ├── person_routes.py       # /persons endpoints
│   ├── image_routes.py         # /images endpoints
│   └── admin_routes.py        # /admin endpoints (reset, cleanup)
│
├── services/                  # Business logic
│   ├── __init__.py
│   └── face_service.py        # Face detection, cropping, matching, merging
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── file_utils.py          # File hashing, listing, streaming
│
├── faces/                     # Auto-created (stores cropped face images)
│   └── {project_name}/
│       └── {id}.jpg
│
├── test/
│   ├── test_face.py           # Test script for face detection
│   ├── test_crop.py           # Test script for face cropping
│   └── test.jpg               # Test image
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup Instructions

### 1. Create a Virtual Environment:
```bash
python -m venv venv
```

### 2. Activate the Environment:
**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install insightface onnxruntime fastapi uvicorn python-multipart pillow sqlalchemy aiofiles opencv-python-headless scikit-learn
```

## API Endpoints

### Scanning Endpoints

#### `POST /scan`
Scan a folder for faces (without auto-merging duplicates)
```json
{
  "project_name": "my_project",
  "source_path": "path/to/images",
  "storage_type": "local"
}
```
**Note:** This keeps ALL face detections. If the same person appears in 5 photos, you get 5 records.

#### `POST /scan-merge` ⭐ NEW
Scan a folder AND automatically merge duplicate faces of the same person
```json
{
  "project_name": "my_project",
  "source_path": "path/to/images",
  "storage_type": "local"
}
```
**Note:** This DELETES duplicate face records. If the same person appears in 5 photos, you get ONLY 1 record (the first one found). Cropped faces are saved to `faces/{project_name}/{id}.jpg`.

### Face Endpoints

#### `GET /faces?project_name=my_project`
List all face records with cropped face images
- Returns: `id`, `file_path`, `person_id`, `person_name`, `face_image_url`, `created_at`
- **After /scan-merge:** Returns only unique persons (duplicates removed)

#### `GET /faces` (without filter)
List all face records across all projects

#### `DELETE /faces/{face_id}` ⭐ NEW
Delete a single face record and its cropped image from disk

#### `PATCH /faces/{face_id}/assign-person`
Manually assign a person to a face
```json
{ "person_id": 1 }
```

### Person Endpoints

#### `GET /persons?project_name=my_project`
List all persons with face count and representative image

#### `GET /persons/{person_id}`
Get person details with all their face images

#### `POST /persons`
Create a new person
```json
{ "name": "John Doe" }
```

#### `POST /persons/merge`
Merge two persons (e.g., Unknown_1 and Unknown_2 are the same)
```json
{ "person_id_keep": 1, "person_id_remove": 2 }
```

#### `DELETE /persons/{person_id}`
Delete a person and ALL their face records (including cropped images from disk)

### Image Endpoints

#### `GET /images/{record_id}`
Stream the cropped face image from `faces/{project_name}/{id}.jpg`

#### `GET /images/faces/{project_name}/{face_id}` ⭐ NEW
Direct access to cropped face image
- URL format: `/images/faces/{projectName}/{id}.jpg`
- Example: `/images/faces/test/1.jpg`

## How It Works

### Option 1: Scan with Auto-Merge (Recommended)

1. **Scan**: Call `/scan-merge` with a folder path
2. **Processing**:
   - System scans all images in the folder
   - Detects faces using InsightFace AI
   - Crops each face and saves to `faces/{project_name}/{id}.jpg`
   - Groups similar faces (same person)
   - **Deletes duplicate records** - keeps only 1 face per unique person
3. **View Results**: Call `/faces?project_name=my_project` to get unique persons
4. **View Person**: Call `/persons/{person_id}` to see all photos of a specific person

### Option 2: Scan without Merge

1. **Scan**: Call `/scan` with a folder path
2. **Processing**:
   - System scans all images
   - Detects faces and crops them
   - **Keeps ALL detections** (same person in 5 photos = 5 records)
3. **Manual Merge**: Use `/persons/merge` endpoint to merge duplicates manually

## Testing the Flow

```bash
# 1. Start server
python main.py

# 2. Scan and merge (in another terminal or via curl)
curl -X POST "http://localhost:8000/scan-merge" \
  -H "Content-Type: application/json" \
  -d '{"project_name":"test","source_path":"./uploads","storage_type":"local"}'

# 3. Wait for processing to complete (check console logs)

# 4. Get all faces (should show only unique persons)
curl "http://localhost:8000/faces?project_name=test"

# 5. Get all persons
curl "http://localhost:8000/persons?project_name=test"

# 6. View a cropped face (both URLs work)
curl "http://localhost:8000/images/1" > face.jpg
curl "http://localhost:8000/images/faces/test/1.jpg" > face_direct.jpg

# 7. Delete a person (removes cropped files too)
curl -X DELETE "http://localhost:8000/persons/1"
```

## Running the Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access API docs at: `http://localhost:8000/docs`