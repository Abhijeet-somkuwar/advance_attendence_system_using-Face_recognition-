# -*- coding: utf-8 -*-
"""
Advanced Attendance System using DeepFace
Features:
- Red box for unknown faces, Green box for recognized faces
- Real-time attendance marking with visual feedback
- Duplicate detection with warning message
- Better UI with on-screen messages
- Comprehensive logging
"""

import cv2
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
from deepface import DeepFace
import time

# ===================== CONFIGURATION =====================
CONFIDENCE_THRESHOLD = 0.65  # Higher = stricter (0.65-0.75 recommended for accuracy)
CHECK_INTERVAL = 5  # Check every N frames (5-8 for smooth performance)
DUPLICATE_CHECK_TIME = 30  # Seconds to wait before allowing same person again
DETECTION_CONFIDENCE = 0.5  # Cascade classifier threshold
MIN_FRAMES_MATCH = 1  # Reduced for faster marking (no delay)
RESIZE_SCALE = 0.75  # Reduce frame size for faster processing (0.75 = 75% of original)

# ===================== INITIALIZATION =====================
attendance = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])
marked_attendance = {}  # Dictionary to track marked attendance with timestamps
webcam_video_stream = cv2.VideoCapture(0)

# Get the script directory and set images path
script_dir = os.path.dirname(os.path.abspath(__file__))
images_folder_path = os.path.join(script_dir, 'images')

# Set camera properties for better performance
try:
    webcam_video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    webcam_video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    webcam_video_stream.set(cv2.CAP_PROP_FPS, 30)
    webcam_video_stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
except:
    pass  # Some cameras don't support all properties

# ===================== LOAD FACE DATABASE =====================
print("=" * 60)
print("Loading faces from images folder...")
print("=" * 60)

known_faces = {}
face_files = []

for filename in os.listdir(images_folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(images_folder_path, filename)
        name = os.path.splitext(filename)[0]
        face_files.append((name, file_path))

print(f"Found {len(face_files)} face image(s)")

for name, file_path in face_files:
    try:
        embedding = DeepFace.represent(img_path=file_path, 
                                      model_name="VGG-Face",
                                      enforce_detection=False)
        known_faces[name] = embedding[0]['embedding']
        print(f"✓ Loaded: {name}")
    except Exception as e:
        print(f"✗ Failed to load {name}: {str(e)[:50]}")

print("=" * 60)
print(f"Total faces loaded: {len(known_faces)}")
print("=" * 60)
print("Starting camera... Press 'q' to quit\n")

# ===================== UTILITY FUNCTIONS =====================

def cosine_distance(embedding1, embedding2):
    """Calculate cosine distance between two embeddings (0 = identical, 2 = opposite)"""
    a = np.array(embedding1)
    b = np.array(embedding2)
    
    # Normalize vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    # Cosine similarity (1 = identical, -1 = opposite)
    similarity = np.dot(a_norm, b_norm)
    
    # Convert to distance (0 = identical, 2 = opposite)
    distance = 1 - similarity
    
    return distance

def recognize_face(face_embedding):
    """Compare face embedding with known faces - returns best match and confidence"""
    best_match = None
    best_confidence = 0.0  # Confidence score (0-100)
    best_distance = float('inf')
    
    for name, known_embedding in known_faces.items():
        distance = cosine_distance(face_embedding, known_embedding)
        
        # Convert distance to confidence (higher is more confident)
        # distance range: 0 (identical) to 2 (opposite)
        confidence = max(0, (1 - distance / 2) * 100)  # Convert to 0-100%
        
        # Keep track of best match
        if confidence > best_confidence:
            best_confidence = confidence
            best_distance = distance
            best_match = name
    
    # Return match only if confidence exceeds threshold
    # CONFIDENCE_THRESHOLD = 0.6 means we need 60% or better match
    if best_confidence >= (CONFIDENCE_THRESHOLD * 100):
        return best_match, best_confidence, best_distance
    
    return None, best_confidence, best_distance

def mark_attendance(name):
    """Mark attendance with duplicate checking"""
    current_time = datetime.now()
    
    # Check if person already marked in the time window
    if name in marked_attendance:
        last_marked_time = marked_attendance[name]
        time_diff = (current_time - last_marked_time).total_seconds()
        
        if time_diff < DUPLICATE_CHECK_TIME:
            return False, f"Already marked {int(time_diff)}s ago"
    
    # Mark new attendance
    marked_attendance[name] = current_time
    attendance.loc[len(attendance)] = [
        name,
        current_time.strftime('%Y-%m-%d'),
        current_time.strftime('%H:%M:%S'),
        'Present'
    ]
    return True, "Attendance marked successfully"

def draw_label_with_background(frame, text, x, y, font_scale=0.6, thickness=1):
    """Draw text with background for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Draw background rectangle
    cv2.rectangle(frame, (x - 5, y - text_height - 10), 
                  (x + text_width + 5, y), (0, 0, 0), -1)
    
    # Put text
    cv2.putText(frame, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)

# ===================== MAIN LOOP =====================

frame_count = 0
message_time = 0
current_message = ""
message_type = ""  # 'success', 'warning', 'info'
face_match_history = {}  # Track consecutive matches for verification
last_detected_faces = []  # Cache last detected faces for stability

while True:
    ret, current_frame = webcam_video_stream.read()
    
    if not ret:
        print("Error: Could not read frame from camera")
        break
    
    frame_count += 1
    
    # Keep original frame for display at full resolution
    original_frame = current_frame.copy()
    
    # Resize frame for faster processing
    current_frame = cv2.resize(current_frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    
    # Flip frame for mirror effect
    current_frame = cv2.flip(current_frame, 1)
    original_frame = cv2.flip(original_frame, 1)
    
    # Process every N frames for performance
    if frame_count % CHECK_INTERVAL == 0:
        try:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detect faces with optimized parameters for speed
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,  # Slightly increased for faster detection
                minNeighbors=4,    # Reduced from 5 for faster detection
                minSize=(40, 40)   # Increased minimum size to reduce false positives
            )
            
            # Scale faces back to original resolution
            faces = [(int(x/RESIZE_SCALE), int(y/RESIZE_SCALE), int(w/RESIZE_SCALE), int(h/RESIZE_SCALE)) 
                    for (x, y, w, h) in faces]
            last_detected_faces = faces.copy()
            
            detected_faces_info = []
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region from original resolution
                face_roi = original_frame[y:y+h, x:x+w]
                
                try:
                    # Get embedding for detected face
                    embedding = DeepFace.represent(
                        img_path=face_roi,
                        model_name="VGG-Face",
                        enforce_detection=False
                    )
                    
                    if embedding:
                        face_embedding = embedding[0]['embedding']
                        name, confidence, distance = recognize_face(face_embedding)
                        
                        # Track face matches for verification
                        face_key = f"{x}_{y}_{w}_{h}"
                        
                        if name and confidence >= (CONFIDENCE_THRESHOLD * 100):
                            # High confidence match
                            color = (0, 255, 0)  # Green
                            thickness = 3
                            
                            # Update match history
                            if face_key not in face_match_history:
                                face_match_history[face_key] = {'name': name, 'count': 1, 'confidence': confidence}
                            else:
                                if face_match_history[face_key]['name'] == name:
                                    face_match_history[face_key]['count'] += 1
                                    face_match_history[face_key]['confidence'] = confidence
                                else:
                                    # Different person detected, reset
                                    face_match_history[face_key] = {'name': name, 'count': 1, 'confidence': confidence}
                            
                            # Mark attendance only after multiple frame matches
                            if face_match_history[face_key]['count'] >= MIN_FRAMES_MATCH:
                                success, message = mark_attendance(name)
                                
                                if success:
                                    current_message = f"✓ {name} - Attendance Marked (Confidence: {confidence:.1f}%)"
                                    message_type = "success"
                                    print(f"✓ MARKED: {name} at {datetime.now().strftime('%H:%M:%S')} - Confidence: {confidence:.1f}%")
                                else:
                                    current_message = f"⚠ {name} - {message}"
                                    message_type = "warning"
                                    print(f"⚠ WARNING: {message} - Confidence: {confidence:.1f}%")
                                
                                message_time = time.time()
                                # Reset after marking
                                face_match_history[face_key]['count'] = 0
                        else:
                            # Unknown or low confidence face - RED BOX
                            color = (0, 0, 255)  # Red
                            thickness = 2
                            
                            # Reset match history for this face region
                            if face_key in face_match_history:
                                del face_match_history[face_key]
                            
                            # Only show unknown message occasionally to avoid spam
                            if frame_count % 15 == 0:
                                current_message = f"❌ Unknown Face - Confidence: {confidence:.1f}% (Need >{CONFIDENCE_THRESHOLD*100:.0f}%)"
                                message_type = "info"
                                message_time = time.time()
                        
                        detected_faces_info.append({
                            'coords': (x, y, w, h),
                            'name': name if name else "Unknown",
                            'color': color,
                            'thickness': thickness,
                            'confidence': confidence,
                            'distance': distance
                        })
                
                except Exception as e:
                    print(f"Error processing face: {str(e)[:50]}")
            
            # Clean up face history for lost faces
            current_face_positions = set(f"{x}_{y}_{w}_{h}" for (x, y, w, h) in faces)
            lost_faces = [key for key in face_match_history if key not in current_face_positions]
            for lost_face in lost_faces:
                del face_match_history[lost_face]
            
            # Draw rectangles and labels
            for face_info in detected_faces_info:
                x, y, w, h = face_info['coords']
                name = face_info['name']
                color = face_info['color']
                thickness = face_info['thickness']
                confidence = face_info['confidence']
                
                # Draw rectangle
                cv2.rectangle(original_frame, (x, y), (x+w, y+h), color, thickness)
                
                # Draw label with background showing confidence
                label = f"{name} ({confidence:.1f}%)"
                draw_label_with_background(original_frame, label, x, y-10, 
                                          font_scale=0.6, thickness=1)
        
        except Exception as e:
            print(f"Frame processing error: {str(e)[:50]}")
    else:
        # Use cached faces for display when not processing to maintain smooth visuals
        for (x, y, w, h) in last_detected_faces:
            cv2.rectangle(original_frame, (x, y), (x+w, y+h), (100, 100, 100), 1)  # Gray for cached
    
    # Display status message
    if time.time() - message_time < 3:  # Show message for 3 seconds
        y_pos = original_frame.shape[0] - 30
        
        if message_type == "success":
            # Green message for success
            cv2.rectangle(original_frame, (10, y_pos - 30), 
                         (original_frame.shape[1] - 10, y_pos + 10), (0, 200, 0), -1)
            cv2.putText(original_frame, current_message, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif message_type == "warning":
            # Orange/Yellow message for warning
            cv2.rectangle(original_frame, (10, y_pos - 30), 
                         (original_frame.shape[1] - 10, y_pos + 10), (0, 165, 255), -1)
            cv2.putText(original_frame, current_message, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Red message for info/unknown
            cv2.rectangle(original_frame, (10, y_pos - 30), 
                         (original_frame.shape[1] - 10, y_pos + 10), (0, 0, 200), -1)
            cv2.putText(original_frame, current_message, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add header
    cv2.rectangle(original_frame, (0, 0), (original_frame.shape[1], 60), (50, 50, 50), -1)
    cv2.putText(original_frame, "ATTENDANCE SYSTEM - Press 'q' to quit", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display footer with stats
    footer_text = f"Marked Today: {len(marked_attendance)} | Total Records: {len(attendance)}"
    cv2.putText(original_frame, footer_text, (10, original_frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show frame at original resolution
    cv2.imshow("Attendance System", original_frame)
    
    # Handle key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n" + "=" * 60)
        print("Shutting down...")
        print("=" * 60)
        break

# ===================== SAVE AND CLEANUP =====================

if len(attendance) > 0:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'attendance_{timestamp}.xlsx'
    attendance.to_excel(filename, index=False)
    print(f"✓ Attendance saved to: {filename}")
    print(f"✓ Total attendance records: {len(attendance)}")
    print("\nAttendance Summary:")
    print(attendance.to_string(index=False))
else:
    print("No attendance records to save.")

print("=" * 60)
print("Program ended successfully!")
print("=" * 60)

# Release resources
webcam_video_stream.release()
cv2.destroyAllWindows()
