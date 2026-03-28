"""
Test script to verify face cropping is working correctly
Run this after scanning some images
"""
import cv2
from insightface.app import FaceAnalysis
import os

# Initialize
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load test image
test_image = "test/test.jpg"
if not os.path.exists(test_image):
    print(f"Error: {test_image} not found!")
    exit(1)

img = cv2.imread(test_image)
print(f"Image size: {img.shape[1]}x{img.shape[0]}")

# Detect faces
faces = app.get(img)
print(f"Found {len(faces)} face(s)\n")

for i, face in enumerate(faces):
    # InsightFace bbox: [x1, y1, x2, y2]
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    print(f"Face {i+1}:")
    print(f"  BBox: [{x1}, {y1}, {x2}, {y2}]")
    print(f"  Face size: {x2-x1}x{y2-y1}")
    
    # Test crop with 5px padding
    padding = 5
    x1_p = max(0, x1 - padding)
    y1_p = max(0, y1 - padding)
    x2_p = min(img.shape[1], x2 + padding)
    y2_p = min(img.shape[0], y2 + padding)
    
    print(f"  With padding: [{x1_p}, {y1_p}, {x2_p}, {y2_p}]")
    
    # Crop and save
    face_crop = img[y1_p:y2_p, x1_p:x2_p]
    output_path = f"test_face_crop_{i+1}.jpg"
    cv2.imwrite(output_path, face_crop)
    print(f"  ✓ Saved: {output_path} ({face_crop.shape[1]}x{face_crop.shape[0]})")
    print()

print("Open the saved images to verify the face is visible and centered!")
