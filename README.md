# Universal Face Recognition & Auto-Deduplication API

A high-performance FastAPI backend for scanning photo directories, automatically detecting human faces, grouping unique individuals, and extracting normalized cropped face thumbnails using InsightFace AI.

---

## 🌟 Key Features

- **Folder Face Scanning**: Scan any local directory (or cloud bucket placeholders) recursively for images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
- **Deep Learning Face Detection & Recognition**: Powered by **InsightFace** (ArcFace model with ONNX Runtime) to detect faces and extract 512-dimensional vector embeddings.
- **Auto-Deduplication (`/scan-merge`)**: Groups duplicate appearances of the same person using Cosine Similarity matching and retains only unique face records while removing duplicates.
- **Normalized Face Cropping**: Dynamically calculates face bounding boxes with padding, enforces a 2:3 aspect ratio, and resizes cropped face images to a standard `200x300` resolution stored under `faces/{project_name}/{id}.jpg`.
- **Background Task Execution**: Runs long scanning and face matching processes asynchronously using FastAPI `BackgroundTasks` to prevent HTTP request timeouts.
- **Incremental Scanning**: Hashes image files using MD5 to skip already processed files during repeated folder scans.
- **Person & Face Management API**: Full RESTful operations to view faces, assign faces to named persons, merge separate person profiles, delete specific faces/persons, and serve cropped images directly.
- **Admin Utilities**: Easily clear specific project data or perform full database resets.

---

## 📂 Project Architecture

```text
image-face-recognition/
├── main.py                    # FastAPI application entry point & route registration
├── database.py                # SQLite database setup & SQLAlchemy session dependency
├── models.py                  # SQLAlchemy ORM models (Project, Person, FaceRecord)
├── schemas.py                 # Pydantic data validation schemas
├── requirements.txt           # Python package dependencies
├── README.md                  # Comprehensive project documentation
│
├── routes/                    # Modular API Route Handlers
│   ├── __init__.py
│   ├── scan_routes.py         # /scan and /scan-merge endpoints
│   ├── face_routes.py         # /faces listing, assignment, and deletion
│   ├── person_routes.py       # /persons CRUD and manual merging
│   ├── image_routes.py        # /images serving cropped face thumbnails
│   └── admin_routes.py        # /admin reset and project cleanup
│
├── services/                  # Business & AI Logic Layer
│   ├── __init__.py
│   └── face_service.py        # InsightFace analyzer, face cropping, matching, and deduplication
│
├── utils/                     # Utility Functions
│   ├── __init__.py
│   └── file_utils.py          # File hashing (MD5), local file listing, and streaming
│
├── faces/                     # Auto-generated directory storing cropped face images
│   └── {project_name}/
│       └── {id}.jpg
│
└── test/                      # Test scripts & sample images
    ├── test_face.py           # Standalone script testing face detection & bounding boxes
    ├── test_crop.py           # Standalone script testing face cropping logic
    └── test.jpg               # Sample image for testing
```

---

## 🚀 Setup & Installation

### Prerequisites

- **Python**: Version 3.8 or higher.
- **C++ Build Tools**: Required by `onnxruntime` and `insightface` compilation on some systems.

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

- **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate
  ```

- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

*Or manually install core packages:*
```bash
pip install fastapi uvicorn insightface onnxruntime opencv-python-headless pillow sqlalchemy scikit-learn numpy python-multipart aiofiles
```

---

## 🏃 Running the Application

Start the FastAPI dev server using Uvicorn:

```bash
python main.py
```

Or run directly via Uvicorn CLI:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- **API Base URL**: `http://localhost:8000`
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## 📖 How It Works

### Option 1: Scan & Auto-Merge Unique Faces (Recommended)

1. **Trigger Scan**: Send a `POST` request to `/scan-merge` with the target folder path (`source_path`) and a `project_name`.
2. **Background Processing**:
   - Recursively finds all image files in the target directory.
   - Detects face bounding boxes and extracts vector embeddings using InsightFace.
   - Crops each detected face with 20% padding and 2:3 aspect ratio, saving it to `faces/{project_name}/{id}.jpg`.
   - Compares face embeddings via Cosine Similarity.
   - **Deduplication**: Automatically groups identical faces into single person profiles and deletes redundant face records and crop files.
3. **Retrieve Unique Results**: Query `GET /faces?project_name={project_name}` or `GET /persons?project_name={project_name}`.

### Option 2: Scan Without Auto-Merging

1. **Trigger Scan**: Send a `POST` request to `/scan`.
2. **Processing**: Keeps **all** face detections in the database (e.g., 5 appearances of Person A produce 5 separate face records).
3. **Manual Merging**: Use `POST /persons/merge` to manually combine duplicate person profiles as needed.

---

## 🛠️ API Reference

### 🔍 Scanning Endpoints

#### `POST /scan`
Initiates a background folder scan without merging duplicates.
- **Request Body**:
  ```json
  {
    "project_name": "vacation_photos",
    "source_path": "C:/Users/User/Pictures/Vacation2024",
    "storage_type": "local"
  }
  ```
- **Response**:
  ```json
  {
    "status": "scanning_started",
    "project_id": 1,
    "message": "Scanning initiated in background. Check status later."
  }
  ```

#### `POST /scan-merge` ⭐
Initiates a background folder scan and automatically merges duplicate faces of the same person.
- **Request Body**:
  ```json
  {
    "project_name": "vacation_photos",
    "source_path": "C:/Users/User/Pictures/Vacation2024",
    "storage_type": "local"
  }
  ```
- **Response**:
  ```json
  {
    "status": "scanning_and_merging_started",
    "project_id": 1,
    "message": "Scanning and auto-merging initiated in background. Check status later."
  }
  ```

---

### 👤 Face Endpoints

#### `GET /faces`
Lists face records with links to cropped face images.
- **Query Parameters**: `project_name` (optional)
- **Response Example**:
  ```json
  [
    {
      "id": 1,
      "file_path": "C:/Users/User/Pictures/Vacation2024/img01.jpg",
      "person_id": 101,
      "person_name": "Person_1",
      "face_image_url": "/images/faces/vacation_photos/1.jpg",
      "created_at": "2026-07-17T09:30:00"
    }
  ]
  ```

#### `PATCH /faces/{face_id}/assign-person`
Manually reassigns a face record to a specific person ID.
- **Query Parameter**: `person_id` (integer)

#### `DELETE /faces/{face_id}`
Deletes a single face record from the database and removes its cropped image file from disk.

---

### 🧑 Person Endpoints

#### `GET /persons`
Lists all recognized persons along with face count and representative thumbnail URL.
- **Query Parameters**: `project_name` (optional)

#### `GET /persons/{person_id}`
Returns details for a specific person, including a list of all associated face records.

#### `POST /persons`
Creates a new person profile.
- **Query Parameter**: `name` (string)

#### `POST /persons/merge`
Merges two person entries into one. Moves all face records from `person_id_remove` to `person_id_keep` and deletes `person_id_remove`.
- **Query Parameters**: `person_id_keep` (int), `person_id_remove` (int)

#### `DELETE /persons/{person_id}`
Deletes a person profile along with **all** associated face records and cropped face images.

---

### 🖼️ Image Endpoints

#### `GET /images/faces/{project_name}/{face_id}`
Directly serves the cropped JPEG thumbnail for a given face ID.

#### `GET /images/{record_id}`
Legacy endpoint to stream a face crop image by record ID.

---

### ⚙️ Admin Endpoints

#### `POST /admin/reset`
⚠️ **Danger Zone**: Wipes all database records (Projects, Persons, FaceRecords) and deletes the entire `faces/` directory.

#### `DELETE /admin/clear-project/{project_name}`
Deletes all records and face thumbnails associated with a specific project name.

---

## 🧪 Testing & Verification

You can test the API flow using `curl` or tools like Postman:

```bash
# 1. Start the backend server
python main.py

# 2. Trigger folder scan with auto-deduplication
curl -X POST "http://localhost:8000/scan-merge" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "sample_folder",
    "source_path": "./test",
    "storage_type": "local"
  }'

# 3. Retrieve extracted unique faces
curl "http://localhost:8000/faces?project_name=sample_folder"

# 4. View representative persons
curl "http://localhost:8000/persons?project_name=sample_folder"

# 5. Fetch a cropped face image
curl "http://localhost:8000/images/faces/sample_folder/1.jpg" --output face_1.jpg
```

You can also test standalone face detection and cropping directly using the scripts in `test/`:

```bash
python test/test_face.py
python test_crop.py
```

---

## 📄 License

This project is licensed under the MIT License.