import os
import cv2
import numpy as np
import pandas as pd
import time
import json
import shutil
from datetime import date, datetime
import sqlite3
import traceback
import base64
import threading
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify, Response
from PIL import Image
from contextlib import contextmanager
from werkzeug.utils import secure_filename
from io import BytesIO
import csv
from flask import send_file
import pytz
from datetime import datetime # Ensure datetime is imported from datetime
# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

try:
    ist = pytz.timezone('Asia/Kolkata')
except pytz.exceptions.UnknownTimeZoneError:
    print("Error: Timezone 'Asia/Kolkata' not found. Ensure pytz is installed correctly.")
    ist = pytz.utc # Fallback to UTC if IST is not found

now_ist = datetime.now(ist)
datetoday = now_ist.strftime("%m_%d_%y")
datetoday2 = now_ist.strftime("%d-%B-%Y")

# Create required directories
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
os.makedirs('static/embeddings', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/temp_embeddings', exist_ok=True)

# Ensure today's attendance file exists
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Database connection manager to prevent locking issues
@contextmanager
def get_db_connection():
    """Context manager for database connections to ensure proper cleanup"""
    conn = None
    try:
        # Set timeout to 30 seconds for lock
        conn = sqlite3.connect('face_attendance.db', timeout=30.0)
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    except Exception as e:
        print(f"Database connection error: {e}")
        traceback.print_exc()
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.commit()
            conn.close()

# Execute database operations with retry mechanism
def execute_with_retry(operation, max_retries=5, retry_delay=1.0):
    """Execute a database operation with retries for lock issues"""
    retries = 0
    while retries < max_retries:
        try:
            return operation()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and retries < max_retries - 1:
                retries += 1
                print(f"Database locked, retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                raise
        except Exception as e:
            print(f"Database operation error: {e}")
            traceback.print_exc()
            raise

# Initialize database
def init_db():
    def _init():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create Users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                user_id TEXT NOT NULL UNIQUE,
                registration_date TEXT NOT NULL
            )
            ''')
            
            # Create Embeddings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            ''')
            
            # Create Attendance table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            ''')
        
        print("Database initialized successfully.")
        
    try:
        execute_with_retry(_init)
    except Exception as e:
        print(f"Error initializing database: {e}")
        traceback.print_exc()

# Initialize the database
init_db()

# Initialize face detection and recognition models
# Using MTCNN for face detection with more relaxed parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create separate MTCNN models for detection and extraction
# This is the key fix for the tensor shape issue
detector = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=20,
    thresholds=[0.5, 0.6, 0.6],  # Lower thresholds for easier detection
    factor=0.709,
    post_process=True,
    device=device,
    select_largest=False,  # Don't just select the largest face
    keep_all=True  # Keep all detected faces
)

# Using FaceNet for face recognition
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Face recognition models loaded successfully.")

# Get total registered users
def get_total_users():
    def _get_count():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            return count
    
    try:
        return execute_with_retry(_get_count)
    except Exception as e:
        print(f"Error getting user count: {e}")
        return 0

# Get all users
def get_all_users():
    def _get_users():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, user_id, registration_date FROM users ORDER BY name")
            
            result = cursor.fetchall()
            users = []
            
            for row in result:
                users.append({
                    'name': row[0],
                    'user_id': row[1],
                    'registration_date': row[2]
                })
                
            return users
    
    try:
        return execute_with_retry(_get_users)
    except Exception as e:
        print(f"Error getting users: {e}")
        traceback.print_exc()
        return []

# Function to add a new user to the database
def add_user_to_db(name, user_id):
    def _add_user():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Use IST for registration_date
            registration_date = datetime.now(ist).strftime("%Y-%m-%d")
            
            # Check if user ID already exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE user_id = ?", (user_id,))
            if cursor.fetchone()[0] > 0:
                print(f"User ID {user_id} already exists.")
                return False
            
            cursor.execute("INSERT INTO users (name, user_id, registration_date) VALUES (?, ?, ?)",
                            (name, user_id, registration_date))
            print(f"Added user {name} with ID {user_id} to database.")
            return True
    
    try:
        return execute_with_retry(_add_user)
    except sqlite3.IntegrityError:
        print(f"User ID {user_id} already exists (IntegrityError).")
        return False
    except Exception as e:
        print(f"Error adding user to database: {e}")
        traceback.print_exc()
        return False

# Function to delete user from database
def delete_user_from_db(user_id):
    def _delete_user():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get user name first (for directory deletion)
            cursor.execute("SELECT name FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            
            if not result:
                print(f"User ID {user_id} not found.")
                return False, None
                
            user_name = result[0]
            
            # Delete embeddings first (foreign key constraint)
            cursor.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
            
            # Delete attendance records (foreign key constraint)
            cursor.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
            
            # Delete user
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            
            deleted = cursor.rowcount > 0
            print(f"Deleted user with ID {user_id} from database: {deleted}")
            return deleted, user_name
    
    try:
        deleted, user_name = execute_with_retry(_delete_user)
        
        # Also delete the user's image directories if they exist
        if deleted and user_name:
            user_dir = f'static/faces/{user_name}_{user_id}'
            temp_embedding_dir = f'static/temp_embeddings/{user_name}_{user_id}'
            
            # Delete face images
            if os.path.exists(user_dir):
                try:
                    shutil.rmtree(user_dir)
                    print(f"Deleted user image directory: {user_dir}")
                except Exception as e:
                    print(f"Error deleting user directory {user_dir}: {e}")
            
            # Delete temp embeddings
            if os.path.exists(temp_embedding_dir):
                try:
                    shutil.rmtree(temp_embedding_dir)
                    print(f"Deleted user embedding directory: {temp_embedding_dir}")
                except Exception as e:
                    print(f"Error deleting user embedding directory {temp_embedding_dir}: {e}")
                    
            # Update attendance CSV files if they exist
            try:
                attendance_files = [f for f in os.listdir('Attendance') if f.endswith('.csv')]
                for attendance_file in attendance_files:
                    file_path = os.path.join('Attendance', attendance_file)
                    df = pd.read_csv(file_path)
                    if 'Roll' in df.columns and user_id in df['Roll'].values:
                        df = df[df['Roll'] != user_id]
                        df.to_csv(file_path, index=False)
                        print(f"Updated attendance file: {attendance_file}")
            except Exception as e:
                print(f"Error updating attendance files: {e}")
                    
        return deleted
    except Exception as e:
        print(f"Error deleting user: {e}")
        traceback.print_exc()
        return False

# Function to save face embedding to database
def save_embedding(user_id, embedding):
    def _save_embedding():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Convert tensor to bytes for storage
            embedding_bytes = embedding.cpu().numpy().tobytes()
            
            cursor.execute("INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)",
                          (user_id, embedding_bytes))
            print(f"Saved embedding for user {user_id}.")
            return True
    
    try:
        return execute_with_retry(_save_embedding)
    except Exception as e:
        print(f"Error saving embedding: {e}")
        traceback.print_exc()
        return False

# Function to get all embeddings for recognition
def get_all_embeddings():
    def _get_embeddings():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT users.name, users.user_id, embeddings.embedding 
                FROM users 
                JOIN embeddings ON users.user_id = embeddings.user_id
            """)
            
            result = cursor.fetchall()
            
            embeddings_dict = {}
            for name, user_id, embedding_bytes in result:
                try:
                    # Convert bytes back to numpy array and then to tensor
                    embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(1, -1)
                    embedding_tensor = torch.from_numpy(embedding_array).to(device)
                    key = f"{name}_{user_id}"
                    embeddings_dict[key] = embedding_tensor
                except Exception as e:
                    print(f"Error processing embedding for user {name}_{user_id}: {e}")
                    continue
            
            print(f"Loaded {len(embeddings_dict)} embeddings from database.")
            return embeddings_dict
    
    try:
        return execute_with_retry(_get_embeddings)
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        traceback.print_exc()
        return {}

# Function to add attendance record
def add_attendance(name, user_id):
    def _add_attendance():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Use IST for current_date and current_time
            current_time_obj = datetime.now(ist)
            current_date = current_time_obj.strftime("%Y-%m-%d")
            current_time = current_time_obj.strftime("%H:%M:%S")
            
            # Check if attendance already marked for today
            # Ensure the date comparison is consistent if dates are stored/compared
            # The current_date is already IST based.
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE user_id = ? AND date = ?
            """, (user_id, current_date)) # Make sure 'date' in DB is also based on consistent timezone logic
            
            if cursor.fetchone() is None:
                cursor.execute("""
                    INSERT INTO attendance (user_id, name, date, time) 
                    VALUES (?, ?, ?, ?)
                """, (user_id, name, current_date, current_time))
                
                # Also update CSV for backward compatibility
                # Ensure datetoday for CSV filename is also IST based (already done globally)
                attendance_csv_path = f'Attendance/Attendance-{datetoday}.csv'
                if not os.path.exists(attendance_csv_path):
                    with open(attendance_csv_path, 'w') as f_csv:
                        f_csv.write('Name,Roll,Time\n') # Add header if file is new

                with open(attendance_csv_path, 'a') as f_csv:
                    f_csv.write(f'{name},{user_id},{current_time}\n') # Write on new line
                
                print(f"Added attendance for {name} (ID: {user_id}) at {current_time} (IST).")
                return True, current_time
            else:
                print(f"Attendance already marked for {name} (ID: {user_id}) today (IST).")
                return False, None
    
    try:
        return execute_with_retry(_add_attendance)
    except Exception as e:
        print(f"Error adding attendance: {e}")
        traceback.print_exc()
        
        # Try to add to CSV directly as fallback
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                f.write(f'\n{name},{user_id},{current_time}')
            print(f"Added attendance to CSV for {name} (ID: {user_id}).")
            return True, current_time
        except:
            print("Failed to add attendance to CSV.")
            return False, None

# Function to extract attendance data for today
def extract_attendance():
    def _extract_attendance():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            current_date = date.today().strftime("%Y-%m-%d")
            
            cursor.execute("""
                SELECT name, user_id, time FROM attendance 
                WHERE date = ? 
                ORDER BY time
            """, (current_date,))
            
            result = cursor.fetchall()
            
            names = [row[0] for row in result]
            rolls = [row[1] for row in result]
            times = [row[2] for row in result]
            l = len(result)
            
            print(f"Retrieved {l} attendance records for today.")
            return names, rolls, times, l
    
    try:
        return execute_with_retry(_extract_attendance)
    except Exception as e:
        print(f"Error extracting attendance from database: {e}")
        traceback.print_exc()
        # Try to read from CSV as fallback
        try:
            df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
            names = df['Name'].tolist()
            rolls = df['Roll'].tolist() 
            times = df['Time'].tolist()
            l = len(df)
            print(f"Retrieved {l} attendance records from CSV.")
            return names, rolls, times, l
        except Exception as csv_e:
            print(f"Failed to read attendance from CSV: {csv_e}")
            return [], [], [], 0

# Extract face from image directly using alignment and resize
def extract_face(img, box, image_size=160, margin=20):
    """Extract face from image with more robust preprocessing"""
    try:
        # Get coordinates with margin
        x1, y1, x2, y2 = box
        
        # Calculate dimensions with margin
        size_bb = max(x2-x1, y2-y1)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate new size with margin
        size = size_bb + 2 * margin
        
        # Calculate new coordinates
        x1 = max(int(center_x - size // 2), 0)
        y1 = max(int(center_y - size // 2), 0)
        x2 = min(int(center_x + size // 2), img.width)
        y2 = min(int(center_y + size // 2), img.height)
        
        # Crop and resize
        face = img.crop((x1, y1, x2, y2))
        face = face.resize((image_size, image_size), Image.BILINEAR)
        
        # Convert to tensor
        face_tensor = torch.from_numpy(np.array(face)).permute(2, 0, 1).float() / 255.0
        return face_tensor.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error extracting face: {e}")
        traceback.print_exc()

# Process face from uploaded image
def process_face_image(image_path):
    """Extract face from uploaded image file with preprocessing"""
    try:
        # Open image
        img = Image.open(image_path).convert('RGB')
        
        # Convert to numpy array for preprocessing
        img_np = np.array(img)
        
        # Apply histogram equalization to improve lighting invariance
        if img_np.shape[2] == 3:  # Color image
            img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            # Convert back to PIL
            img = Image.fromarray(img_np)
        
        # Detect faces
        boxes, _ = detector.detect(img)
        
        if boxes is None or len(boxes) == 0:
            print(f"No face detected in {image_path}")
            return None
        
        # Use the first detected face
        box = boxes[0]
        x1, y1, x2, y2 = [int(i) for i in box]
        
        # Extract face
        face_tensor = extract_face(img, (x1, y1, x2, y2))
        return face_tensor
    
    except Exception as e:
        print(f"Error processing face image {image_path}: {e}")
        traceback.print_exc()
        return None

# Identify face using cosine similarity between embeddings
def identify_face(face_tensor, threshold=0.6, samples=3):
    """Identify face with multiple samples for better accuracy"""
    try:
        # Get embeddings database
        embeddings_dict = get_all_embeddings()
        
        if not embeddings_dict:
            print("No embeddings in database. Returning Unknown.")
            return "Unknown_0"
        
        # Get embedding for current face
        embedding = resnet(face_tensor).detach()
        
        # Create slight variations for robustness (simulate small movements)
        embeddings = [embedding]
        
        # Use the matches across all variations
        scores = {}
        
        for emb in embeddings:
            for name_id, ref_embedding in embeddings_dict.items():
                # Calculate cosine similarity (higher is better)
                similarity = torch.nn.functional.cosine_similarity(emb, ref_embedding)
                score = similarity.item()
                
                # Add to scores, keeping the best score per identity
                if name_id not in scores or score > scores[name_id]:
                    scores[name_id] = score
        
        # Find the best match
        best_match = None
        best_score = 0
        
        for name_id, score in scores.items():
            if score > best_score:
                best_score = score
                best_match = name_id
        
        print(f"Best match: {best_match} with score {best_score:.4f}")
        
        # If similarity is below threshold, mark as unknown
        if best_score < threshold:
            return "Unknown_0"
        
        return best_match
    except Exception as e:
        print(f"Error identifying face: {e}")
        traceback.print_exc()
        return "Unknown_0"
    
# Function to get image count for a user
def get_temp_embedding_count(username, userid):
    """Get the number of temporary embeddings saved for a user"""
    try:
        temp_dir = f'static/temp_embeddings/{username}_{userid}'
        if not os.path.exists(temp_dir):
            return 0
        
        # Count .pt files
        count = len([f for f in os.listdir(temp_dir) if f.endswith('.pt')])
        return count
    except Exception as e:
        print(f"Error getting embedding count: {e}")
        return 0

# Global variable to store camera reference
camera = None
camera_lock = threading.Lock()  # Add threading import at the top
last_frame = None
camera_error = None

# Function to safely access the camera
def get_camera():
    global camera, camera_error
    
    if camera is not None and camera.isOpened():
        return camera
    
    # Try to initialize camera with error handling
    with camera_lock:
        # First release any existing camera
        if camera is not None:
            try:
                camera.release()
            except:
                pass
            camera = None
        
        # Try different camera indices
        for camera_idx in [0, 1, 2, -1]:
            try:
                print(f"Trying camera index {camera_idx}")
                cam = cv2.VideoCapture(camera_idx)
                if cam.isOpened():
                    # Set lower resolution for better performance
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera = cam
                    camera_error = None
                    print(f"Successfully opened camera with index {camera_idx}")
                    return camera
            except Exception as e:
                print(f"Error with camera index {camera_idx}: {e}")
                camera_error = str(e)
                continue
        
        # Try a different method on Linux (v4l2)
        try:
            import subprocess
            result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
            print("Available video devices:")
            print(result.stdout)
            
            # Try /dev/video0 directly
            cam = cv2.VideoCapture('/dev/video0')
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera = cam
                camera_error = None
                print("Successfully opened camera with /dev/video0")
                return camera
        except Exception as e:
            print(f"Error with v4l2 method: {e}")
            camera_error = f"Failed to access camera: {str(e)}"
        
        print("Failed to open any camera")
        camera_error = "Failed to access camera. Please check camera connections and permissions."
        return None

# Function to generate camera frames with better error handling
def generate_frames():
    global last_frame, camera_error
    
    # Create a blank frame with error message
    def create_error_frame(message):
        blank_frame = np.zeros((480, 640, 3), np.uint8)
        # Draw background rectangle
        cv2.rectangle(blank_frame, (0, 200), (640, 280), (50, 50, 50), -1)
        # Add text
        cv2.putText(blank_frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return blank_frame
    
    # Try to get camera
    cam = get_camera()
    if cam is None:
        # Return error frame
        error_frame = create_error_frame(camera_error or "Camera not available")
        _, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    frame_count = 0
    recognition_frequency = 3  # Only process every 3rd frame
    last_recognition_time = time.time() - 5
    retry_count = 0
    max_retries = 5
    
    while True:
        try:
            if cam is None or not cam.isOpened():
                # Try to reconnect
                if retry_count < max_retries:
                    retry_count += 1
                    print(f"Camera disconnected, trying to reconnect (attempt {retry_count}/{max_retries})")
                    time.sleep(1)  # Wait before retry
                    cam = get_camera()
                    if cam is None:
                        error_frame = create_error_frame("Reconnecting to camera...")
                        _, buffer = cv2.imencode('.jpg', error_frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        continue
                else:
                    error_frame = create_error_frame("Failed to reconnect to camera")
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    break
            
            success, frame = cam.read()
            
            if not success:
                print("Failed to read frame from camera")
                retry_count += 1
                if retry_count >= max_retries:
                    error_frame = create_error_frame("Failed to read from camera")
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    break
                
                # Try to reconnect
                print(f"Trying to reconnect to camera (attempt {retry_count}/{max_retries})")
                time.sleep(1)
                cam = get_camera()
                continue
            
            # Reset retry count on successful frame read
            retry_count = 0
            
            # Store last successful frame
            last_frame = frame.copy()
            
            # Create a copy for display
            display_frame = frame.copy()
            current_time = time.time()
            
            # Process only every few frames to improve performance
            if frame_count % recognition_frequency == 0 and current_time - last_recognition_time > 2:
                last_recognition_time = current_time
                
                # Convert to RGB (MTCNN expects RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb_frame)
                
                # Detect faces with MTCNN
                try:
                    boxes, _ = detector.detect(img_pil)
                    
                    if boxes is not None:
                        for box in boxes:
                            # Get coordinates
                            x1, y1, x2, y2 = [int(i) for i in box]
                            
                            # Ensure coordinates are within frame boundaries
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            
                            if x2 > x1 and y2 > y1:  # Valid box dimensions
                                # Draw rectangle around face
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Extract and process the face
                                try:
                                    # Extract face manually
                                    face_tensor = extract_face(img_pil, (x1, y1, x2, y2))
                                    
                                    # Identify face
                                    identity = identify_face(face_tensor)
                                    name, user_id = identity.split('_')[0], identity.split('_')[1]
                                    
                                    # Draw name text
                                    cv2.putText(display_frame, f'{name}', (x1, y1 - 10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                except Exception as e:
                                    print(f"Error processing detected face: {e}")
                                    cv2.putText(display_frame, "Error", (x1, y1 - 10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception as e:
                    print(f"Error in face detection: {e}")
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                continue
                
            # Return the frame as bytes in multipart/x-mixed-replace format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
            frame_count += 1
            
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            traceback.print_exc()
            
            # Return error frame on exception
            error_frame = create_error_frame(f"Camera error: {str(e)}")
            _, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Wait a bit before continuing
            time.sleep(1)

# Routes
@app.route('/')
def home():
    # Redirect to the attendance page by default
    return redirect(url_for('attendance_page'))

@app.route('/attendance')
def attendance_page():
    names, rolls, times, l = extract_attendance()    
    return render_template('attendance.html', 
                            names=names, 
                            rolls=rolls, 
                            times=times, 
                            l=l, 
                            # totalreg=get_total_users(), # Not needed for this page directly
                            datetoday2=datetoday2,
                            mess=request.args.get('mess')) # For flash messages if any

@app.route('/user-management')
def user_management_page():
    return render_template('user_management.html', 
                            totalreg=get_total_users(), 
                            datetoday2=datetoday2,
                            mess=request.args.get('mess'))


# def index():
#     names, rolls, times, l = extract_attendance()    
#     return render_template('index.html', 
#                            names=names, 
#                            rolls=rolls, 
#                            times=times, 
#                            l=l, 
#                            totalreg=get_total_users(), 
#                            datetoday2=datetoday2)


@app.route('/start_attendance')
def start_attendance():
    """Now just a stub that returns success"""
    return jsonify({'success': True, 'message': 'Client-side camera mode active'})

@app.route('/stop_attendance')
def stop_attendance():
    """Now just a stub that returns success"""
    return jsonify({'success': True, 'message': 'Client-side camera mode stopped'})

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    """Process a frame from client-side camera and mark attendance if a face is recognized"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        try:
            # Decode base64 image
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL Image
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Add some debugging output (optional)
            print(f"Received image of size: {img.width}x{img.height}")
        except Exception as e:
            print(f"Error decoding image: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Error decoding image: {str(e)}'})
        
        # Detect faces
        try:
            boxes, _ = detector.detect(img)
            
            if boxes is None or len(boxes) == 0:
                return jsonify({'success': False, 'message': 'No face detected in image'})
            
            # Use the first detected face
            box = boxes[0]
            x1, y1, x2, y2 = [int(i) for i in box]
            
            # Extract face
            face_tensor = extract_face(img, (x1, y1, x2, y2))
            
            # Identify face
            identity = identify_face(face_tensor)
            name, user_id = identity.split('_')[0], identity.split('_')[1]
            
            if name == "Unknown":
                return jsonify({'success': False, 'message': 'Unknown person detected'})
            
            # Add attendance
            marked, time_marked = add_attendance(name, user_id)
            
            if marked:
                return jsonify({
                    'success': True, 
                    'message': f'Attendance marked for {name}',
                    'name': name,
                    'user_id': user_id,
                    'time': time_marked
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': f'Attendance already marked for {name} today'
                })
        except Exception as e:
            print(f"Error in face detection/recognition: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Error in processing: {str(e)}'})
    
    except Exception as e:
        print(f"Error marking attendance: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/get_attendance')
def get_attendance():
    """API endpoint to get attendance for real-time updates"""
    names, rolls, times, l = extract_attendance()
    attendance_data = []
    for i in range(l):
        attendance_data.append({
            'name': names[i],
            'roll': rolls[i],
            'time': times[i]
        })
    return jsonify(attendance_data)

@app.route('/get_all_users')
def get_users():
    """API endpoint to get all registered users"""
    users = get_all_users()
    return jsonify({'success': True, 'users': users})

@app.route('/register_user', methods=['POST'])
def register_user():
    """Register a new user (first step - just create the DB entry)"""
    try:
        data = request.json
        username = data.get('username')
        userid = data.get('userid')
        
        if not username or not userid:
            return jsonify({'success': False, 'message': 'Username and user ID are required'})
        
        # Add user to database
        success = add_user_to_db(username, userid)
        
        if not success:
            return jsonify({'success': False, 'message': 'User ID already exists'})
        
        # Create folder for user images and temporary embeddings
        userdir = f'static/faces/{username}_{userid}'
        temp_embedding_dir = f'static/temp_embeddings/{username}_{userid}'
        os.makedirs(userdir, exist_ok=True)
        os.makedirs(temp_embedding_dir, exist_ok=True)
        
        return jsonify({'success': True, 'message': 'User registered'})
    
    except Exception as e:
        print(f"Error registering user: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/delete_user', methods=['POST'])
def delete_user():
    """Delete a user and all associated data"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'success': False, 'message': 'User ID is required'})
        
        # Delete user from database
        success = delete_user_from_db(user_id)
        
        if success:
            return jsonify({'success': True, 'message': 'User deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'User not found or could not be deleted'})
    
    except Exception as e:
        print(f"Error deleting user: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Save a captured image (from webcam or file upload)"""
    try:
        if 'face_image' not in request.files:
            return jsonify({'success': False, 'message': 'No image file'})
        
        username = request.form.get('username')
        userid = request.form.get('userid')
        image_count = int(request.form.get('image_count', '0'))
        source = request.form.get('source', 'unknown')
        
        if not username or not userid:
            return jsonify({'success': False, 'message': 'Username and user ID are required'})
        
        # Save the image
        image_file = request.files['face_image']
        userdir = f'static/faces/{username}_{userid}'
        image_path = os.path.join(userdir, f'{username}_{image_count}.jpg')
        
        os.makedirs(userdir, exist_ok=True)
        image_file.save(image_path)
        
        # Process the image to get face tensor
        face_tensor = process_face_image(image_path)
        
        if face_tensor is None:
            # If no face detected, remove the saved image
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({'success': False, 'message': 'No face detected in the image'})
        
        # Get embedding
        embedding = resnet(face_tensor).detach()
        
        # Store the embedding in a temporary location
        temp_embedding_dir = f'static/temp_embeddings/{username}_{userid}'
        os.makedirs(temp_embedding_dir, exist_ok=True)
        
        embedding_path = os.path.join(temp_embedding_dir, f'embedding_{image_count}.pt')
        torch.save(embedding, embedding_path)
        
        return jsonify({'success': True, 'message': f'Image saved from {source}'})
    
    except Exception as e:
        print(f"Error capturing image: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})
    


@app.route('/export_attendance')
def export_attendance():
    """Export attendance for the current day as CSV"""
    try:
        names, rolls, times, l = extract_attendance()
        
        if l == 0:
            return jsonify({'success': False, 'message': 'No attendance records for today'})
        
        # Create a CSV in memory (in binary mode)
        output = BytesIO()
        
        # Write the CSV header and data as bytes
        output.write('Name,ID,Time\n'.encode('utf-8'))
        
        for i in range(l):
            row = f'{names[i]},{rolls[i]},{times[i]}\n'
            output.write(row.encode('utf-8'))
        
        # Prepare the CSV for download
        output.seek(0)
        
        # Set filename with date
        filename = f"attendance_{datetoday}.csv"
        
        # Return as downloadable file
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"Error exporting attendance: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/get_attendance_history', methods=['POST'])
def get_attendance_history():
    """Get attendance history for a specific user"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'success': False, 'message': 'User ID is required'})
        
        # Get user attendance history
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, time FROM attendance 
                WHERE user_id = ? 
                ORDER BY date DESC, time DESC
                LIMIT 30
            """, (user_id,))
            
            history = [{'date': row[0], 'time': row[1]} for row in cursor.fetchall()]
            
            return jsonify({
                'success': True, 
                'history': history
            })
    
    except Exception as e:
        print(f"Error getting attendance history: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/complete_registration', methods=['POST'])
def complete_registration():
    """Process all temporary embeddings and finalize registration"""
    try:
        data = request.json
        username = data.get('username')
        userid = data.get('userid')
        
        if not username or not userid:
            return jsonify({'success': False, 'message': 'Username and user ID are required'})
        
        # Check if we have enough embeddings
        temp_embedding_dir = f'static/temp_embeddings/{username}_{userid}'
        if not os.path.exists(temp_embedding_dir):
            return jsonify({'success': False, 'message': 'No embeddings found for this user'})
        
        # Load all embeddings
        all_embeddings = []
        embedding_files = [f for f in os.listdir(temp_embedding_dir) if f.endswith('.pt')]
        
        if len(embedding_files) < 1:
            return jsonify({'success': False, 'message': 'No valid embeddings found'})
        
        for file in embedding_files:
            file_path = os.path.join(temp_embedding_dir, file)
            emb = torch.load(file_path)
            all_embeddings.append(emb)
        
        # Average embeddings
        stacked_embeddings = torch.cat(all_embeddings, dim=0)
        avg_embedding = torch.mean(stacked_embeddings, dim=0, keepdim=True)
        
        # Save to database
        if save_embedding(userid, avg_embedding):
            print(f"Successfully saved face embedding for user {username}_{userid}")
            
            # Optional: Clean up temporary files
            try:
                for file in embedding_files:
                    os.remove(os.path.join(temp_embedding_dir, file))
                os.rmdir(temp_embedding_dir)
            except:
                print("Warning: Could not clean up temporary embedding files")
            
            return jsonify({'success': True, 'message': 'Registration completed successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to save embeddings to database'})
        
    except Exception as e:
        print(f"Error completing registration: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

# Cleanup function to release camera when app stops
@app.teardown_appcontext
def release_camera(exception):
    global camera
    
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
                print("Camera released on app shutdown")
            except:
                pass
            camera = None

# Run the Flask app
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000, ssl_context=('/home/ayyappa/Documents/FaceRec-attendance/ssl/cert.pem', '/home/ayyappa/Documents/FaceRec-attendance/ssl/key.pem'))