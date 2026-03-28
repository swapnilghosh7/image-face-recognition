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

            # Get bounding box - InsightFace returns [x1, y1, x2, y2] (NOT x,y,width,height)
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            width = x2 - x1
            height = y2 - y1
            
            print(f"Face #{i+1}:")
            print(f"  - BBox: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
            print(f"  - Size: {width}x{height}")
            print(f"  - Embedding shape: {embedding.shape}")
            print(f"  - Gender: {face.gender}, Age: {face.age}")
            
            # Optional: Draw bbox and save test output
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Face {i+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save test output
        cv2.imwrite("test_output.jpg", img)
        print(f"\nSaved annotated image: test_output.jpg")