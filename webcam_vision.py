import cv2
import os
from google.cloud import vision
from PIL import Image
import io
import time
from dotenv import load_dotenv
import numpy as np
import hashlib

# Load environment variables
load_dotenv()

def get_color_for_object(object_name):
    """Generate a consistent color for each object type using a hash function"""
    # Use hash of object name to generate color
    hash_value = int(hashlib.md5(object_name.encode()).hexdigest()[:6], 16)
    # Create bright, distinct colors
    hue = (hash_value % 255) / 255.0  # Normalize to 0-1
    # Convert HSV to BGR (using full saturation and value for vibrant colors)
    rgb = tuple(round(i * 255) for i in cv2.cvtColor(np.uint8([[[hue * 180, 255, 255]]]), 
                                                    cv2.COLOR_HSV2BGR)[0][0])
    return rgb

def detect_objects(image_content):
    """Detects objects in the image using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    
    # Perform object detection
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    
    return objects  # Return full object data including bounding boxes

def draw_objects_on_frame(frame, detected_objects):
    """Draw bounding boxes and labels for detected objects"""
    height, width = frame.shape[:2]
    
    for obj in detected_objects:
        # Get normalized vertices
        vertices = obj.bounding_poly.normalized_vertices
        
        # Convert normalized coordinates to actual pixel coordinates
        box_points = []
        for vertex in vertices:
            x_px = int(vertex.x * width)
            y_px = int(vertex.y * height)
            box_points.append((x_px, y_px))
            
        # Convert points to numpy array for drawing
        box_points = np.array(box_points)
        
        # Get color for this object type
        color = get_color_for_object(obj.name)
        
        # Draw the bounding box
        cv2.polylines(frame, [box_points], True, color, 2)
        
        # Prepare label text with object name and confidence
        label = f"{obj.name}: {obj.score:.2f}"
        
        # Calculate label position (above the box)
        label_x = int(vertices[0].x * width)
        label_y = int(vertices[0].y * height) - 10
        
        # Ensure label is within frame bounds
        if label_y < 0:
            label_y = int(vertices[3].y * height) + 20
            
        # Add background rectangle for text
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame,
                     (label_x, label_y - label_h - baseline),
                     (label_x + label_w, label_y + baseline),
                     color,
                     cv2.FILLED)
        
        # Add label text
        cv2.putText(frame, label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # White text
                    1)
    
    return frame

def test_camera_devices():
    """Test available camera devices"""
    working_ports = []
    for i in range(5):  # Test first 5 indexes
        camera = cv2.VideoCapture(i)
        if camera.isOpened():
            ret, frame = camera.read()
            if ret:
                working_ports.append(i)
                print(f"Camera index {i} is working")
            camera.release()
        else:
            print(f"Camera index {i} is not available")
    return working_ports

def main():
    print("Testing available camera devices...")
    working_ports = test_camera_devices()
    
    if not working_ports:
        print("Error: No working cameras found")
        return
        
    # Try to use the first working camera
    camera_index = working_ports[0]
    print(f"Using camera index: {camera_index}")
    
    # Initialize webcam with more detailed error reporting
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}")
        print("Camera properties:")
        print(f"Backend name: {cap.getBackendName()}")
        for prop_id in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, 
                       cv2.CAP_PROP_FPS, cv2.CAP_PROP_BRIGHTNESS]:
            print(f"Property {prop_id}: {cap.get(prop_id)}")
        return

    print("Webcam initialized successfully")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print("Press 'q' to quit.")
    
    last_detection_time = 0
    detection_interval = 2  # Seconds between detections
    last_detected_objects = []  # Store last detected objects
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Draw last detected objects on current frame
        if last_detected_objects:
            frame = draw_objects_on_frame(frame, last_detected_objects)
            
        # Show the frame
        cv2.imshow('Webcam Feed', frame)
        
        # Perform object detection every few seconds
        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            # Convert frame to bytes for Google Cloud Vision
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                image_content = buffer.tobytes()
                
                try:
                    # Detect objects
                    detected_objects = detect_objects(image_content)
                    last_detected_objects = detected_objects  # Update last detected objects
                    
                    # Print detected objects with confidence scores
                    print("\nDetected objects:")
                    for obj in detected_objects:
                        print(f"{obj.name}: {obj.score:.2f}")
                        
                except Exception as e:
                    print(f"Error during object detection: {e}")
                
                last_detection_time = current_time
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()