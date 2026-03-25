import cv2
import insightface
from insightface.app import FaceAnalysis
import os

# 1. Initialize the Face Analysis app
# 'buffalo_l' is a large, accurate model. 'antelopev2' is also good.
app = FaceAnalysis(providers=['CPUExecutionProvider']) 
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Load Image
image_path = "test.jpg"

if not os.path.exists(image_path):
    print(f"Error: {image_path} not found. Please add an image to test.")
else:
    print("Loading image...")
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not load image. Check file path.")
    else:
        print("Detecting faces...")
        # 3. Get Faces
        faces = app.get(img)
        
        print(f"Found {len(faces)} face(s)!")
        
        for i, face in enumerate(faces):
            # Get the embedding (the 512-dimension vector representing the face)
            embedding = face.embedding
            
            # Get bounding box
            bbox = face.bbox.astype(int)
            print(f"Face #{i+1}:")
            print(f"  - Location: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
            print(f"  - Embedding shape: {embedding.shape}")
            print(f"  - Gender: {face.gender}, Age: {face.age}")