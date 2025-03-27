import os
import cv2
import numpy as np
import pandas as pd
import time
import json
import shutil
from datetime import date, datetime
import sqlite3
import torch
import traceback
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify, Response
from PIL import Image
from contextlib import contextmanager
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Date formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

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
            registration_date = date.today().strftime("%Y-%m-%d")
            
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
            
            current_date = date.today().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Check if attendance already marked for today
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE user_id = ? AND date = ?
            """, (user_id, current_date))
            
            if cursor.fetchone() is None:
                cursor.execute("""
                    INSERT INTO attendance (user_id, name, date, time) 
                    VALUES (?, ?, ?, ?)
                """, (user_id, name, current_date, current_time))
                
                # Also update CSV for backward compatibility
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{name},{user_id},{current_time}')
                
                print(f"Added attendance for {name} (ID: {user_id}) at {current_time}.")
                return True
            else:
                print(f"Attendance already marked for {name} (ID: {user_id}) today.")
                return False
    
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
            return True
        except:
            print("Failed to add attendance to CSV.")
            return False

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
    """Extract face from image given bounding box coordinates"""
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

# Process face from uploaded image
def process_face_image(image_path):
    """Extract face from uploaded image file"""
    try:
        # Open image
        img = Image.open(image_path).convert('RGB')
        
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
def identify_face(face_tensor, threshold=0.6):  # Lowered threshold for easier matching
    try:
        # Get embeddings database
        embeddings_dict = get_all_embeddings()
        
        if not embeddings_dict:
            print("No embeddings in database. Returning Unknown.")
            return "Unknown_0"
        
        # Get embedding for current face
        embedding = resnet(face_tensor).detach()
        
        # Find closest match
        best_match = None
        best_score = 0
        
        for name_id, ref_embedding in embeddings_dict.items():
            # Calculate cosine similarity (higher is better)
            similarity = torch.nn.functional.cosine_similarity(embedding, ref_embedding)
            score = similarity.item()
            
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

# Routes
@app.route('/')
def index():
    names, rolls, times, l = extract_attendance()    
    return render_template('index.html', 
                           names=names, 
                           rolls=rolls, 
                           times=times, 
                           l=l, 
                           totalreg=get_total_users(), 
                           datetoday2=datetoday2)

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

@app.route('/start', methods=['GET'])
def start():
    if get_total_users() == 0:
        return render_template('index.html', 
                               totalreg=get_total_users(), 
                               datetoday2=datetoday2, 
                               mess='No users in the database. Please add a new user to continue.')

    # Start camera in a separate process
    try:
        # Try different camera indices if the first one fails
        cap = None
        for camera_idx in [0, 1]:
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                print(f"Successfully opened camera with index {camera_idx}")
                break
        
        if not cap or not cap.isOpened():
            print("Failed to open any camera")
            return render_template('index.html', 
                                  totalreg=get_total_users(), 
                                  datetoday2=datetoday2, 
                                  mess='Unable to access camera. Please check your camera connection.')
        
        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        recognition_frequency = 3  # Only process every 3rd frame for better performance
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Process only every few frames to improve performance
            if frame_count % recognition_frequency == 0:
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
                                    
                                    # Add attendance
                                    if name != "Unknown":
                                        add_attendance(name, user_id)
                                    
                                    # Draw name text
                                    cv2.putText(display_frame, f'{name}', (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                except Exception as e:
                                    print(f"Error processing detected face: {e}")
                                    cv2.putText(display_frame, "Error processing face", (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception as e:
                    print(f"Error in face detection: {e}")
            
            # Display instructions
            cv2.putText(display_frame, "Press ESC or Q to quit", (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Attendance System', display_frame)
            
            # Check for quit key (ESC or 'q')
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Extract updated attendance info and redirect to index with a flag for auto-refresh
        return redirect(url_for('index', attendance_mode='true'))
    
    except Exception as e:
        print(f"Error in start route: {e}")
        traceback.print_exc()
        return render_template('index.html', 
                              totalreg=get_total_users(), 
                              datetoday2=datetoday2, 
                              mess=f'An error occurred: {str(e)}')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=1000)