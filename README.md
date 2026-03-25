# image-face-recognition
This project is a backend project in which we can find all the people present in the photos of a specific project.
# project architecture
image-face-recognition/
│
├── venv/                  # Your virtual environment
├── uploads/               # Create this folder manually (stores images)
├── database.py            # Database connection setup
├── models.py              # Database table definitions (SQLAlchemy)
├── schemas.py             # Data validation (Pydantic)
├── main.py                # The main FastAPI application
├── test_insight.py        # Your working test script
└── requirements.txt       # List of libraries (optional but good practice)
# Create a Virtual Environment:
bash
 python -m venv venv
# Activate the Environment:
Windows (Command Prompt): venv\Scripts\activate
Windows (PowerShell): venv\Scripts\Activate.ps1
Mac/Linux: source venv/bin/activate
(You should see (venv) appear at the start of your terminal line.)
# Install Command 
pip install insightface onnxruntime fastapi uvicorn python-multipart pillow sqlalchemy aiofiles opencv-python-headless

# installing skit learning
pip install scikit-learn