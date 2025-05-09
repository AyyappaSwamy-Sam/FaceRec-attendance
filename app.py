import os
import cv2
import numpy as np
import pandas as pd
import time
import json
import shutil
from datetime import date, datetime # Ensure datetime is imported from datetime
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
os.makedirs('static/images', exist_ok=True) # Ensure this exists for spicy.png
os.makedirs('static/temp_embeddings', exist_ok=True)

# Ensure today's attendance file exists
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n') # Added newline for header

# Database connection manager to prevent locking issues
@contextmanager
def get_db_connection():
    """Context manager for database connections to ensure proper cleanup"""
    conn = None
    try:
        conn = sqlite3.connect('face_attendance.db', timeout=30.0)
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
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                user_id TEXT NOT NULL UNIQUE,
                registration_date TEXT NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE 
            )
            ''') # Added ON DELETE CASCADE
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            ''') # Added ON DELETE CASCADE
        
        print("Database initialized successfully.")
        
    try:
        execute_with_retry(_init)
    except Exception as e:
        print(f"Error initializing database: {e}")
        traceback.print_exc()

init_db()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

detector = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=20,
    thresholds=[0.5, 0.6, 0.6],  
    factor=0.709,
    post_process=True,
    device=device,
    select_largest=False, 
    keep_all=True 
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Face recognition models loaded successfully.")

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

def get_all_users_from_db(): # Renamed to avoid conflict with route
    def _get_users():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, user_id, registration_date FROM users ORDER BY name")
            result = cursor.fetchall()
            users = [{'name': row[0], 'user_id': row[1], 'registration_date': row[2]} for row in result]
            return users
    try:
        return execute_with_retry(_get_users)
    except Exception as e:
        print(f"Error getting users: {e}")
        traceback.print_exc()
        return []

def add_user_to_db(name, user_id):
    def _add_user():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            registration_date = datetime.now(ist).strftime("%Y-%m-%d")
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

def delete_user_from_db(user_id):
    def _delete_user():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            if not result:
                print(f"User ID {user_id} not found.")
                return False, None
            user_name = result[0]
            # ON DELETE CASCADE should handle embeddings and attendance
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            deleted = cursor.rowcount > 0
            print(f"Deleted user with ID {user_id} from database: {deleted}")
            return deleted, user_name
    try:
        deleted, user_name = execute_with_retry(_delete_user)
        if deleted and user_name:
            user_dir = f'static/faces/{user_name}_{user_id}'
            temp_embedding_dir = f'static/temp_embeddings/{user_name}_{user_id}'
            if os.path.exists(user_dir): shutil.rmtree(user_dir, ignore_errors=True)
            if os.path.exists(temp_embedding_dir): shutil.rmtree(temp_embedding_dir, ignore_errors=True)
            # CSV update (optional, as DB is primary)
            try:
                for attendance_file in [f for f in os.listdir('Attendance') if f.endswith('.csv')]:
                    file_path = os.path.join('Attendance', attendance_file)
                    df = pd.read_csv(file_path)
                    if 'Roll' in df.columns and user_id in df['Roll'].astype(str).values: # Ensure Roll is compared as string if needed
                        df = df[df['Roll'].astype(str) != user_id]
                        df.to_csv(file_path, index=False)
            except Exception as e: print(f"Error updating CSVs: {e}")
        return deleted
    except Exception as e:
        print(f"Error deleting user: {e}")
        traceback.print_exc()
        return False

def save_embedding(user_id, embedding):
    def _save_embedding():
        with get_db_connection() as conn:
            cursor = conn.cursor()
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

def get_all_embeddings():
    def _get_embeddings():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT users.name, users.user_id, embeddings.embedding 
                FROM users JOIN embeddings ON users.user_id = embeddings.user_id
            """)
            result = cursor.fetchall()
            embeddings_dict = {}
            for name, user_id, embedding_bytes in result:
                try:
                    embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(1, -1)
                    embedding_tensor = torch.from_numpy(embedding_array).to(device)
                    embeddings_dict[f"{name}_{user_id}"] = embedding_tensor
                except Exception as e:
                    print(f"Error processing embedding for user {name}_{user_id}: {e}")
            print(f"Loaded {len(embeddings_dict)} embeddings from database.")
            return embeddings_dict
    try:
        return execute_with_retry(_get_embeddings)
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        traceback.print_exc()
        return {}

def add_attendance(name, user_id):
    def _add_attendance():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_time_obj = datetime.now(ist)
            current_date_ist_db = current_time_obj.strftime("%Y-%m-%d")
            current_time_ist_db = current_time_obj.strftime("%H:%M:%S")
            
            cursor.execute("SELECT id FROM attendance WHERE user_id = ? AND date = ?", (user_id, current_date_ist_db))
            if cursor.fetchone() is None:
                cursor.execute("INSERT INTO attendance (user_id, name, date, time) VALUES (?, ?, ?, ?)", 
                               (user_id, name, current_date_ist_db, current_time_ist_db))
                # CSV update (datetoday is already IST based)
                attendance_csv_path = f'Attendance/Attendance-{datetoday}.csv'
                if not os.path.exists(attendance_csv_path):
                    with open(attendance_csv_path, 'w') as f_csv: f_csv.write('Name,Roll,Time\n')
                with open(attendance_csv_path, 'a') as f_csv:
                    f_csv.write(f'{name},{user_id},{current_time_ist_db}\n')
                print(f"Added attendance for {name} (ID: {user_id}) at {current_time_ist_db} (IST).")
                return True, current_time_ist_db
            else:
                print(f"Attendance already marked for {name} (ID: {user_id}) today (IST).")
                return False, None
    try:
        return execute_with_retry(_add_attendance)
    except Exception as e:
        print(f"Error adding attendance: {e}")
        traceback.print_exc()
        return False, None

def extract_attendance():
    def _extract_attendance():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            current_date_ist_db = datetime.now(ist).strftime("%Y-%m-%d")
            cursor.execute("SELECT name, user_id, time FROM attendance WHERE date = ? ORDER BY time", (current_date_ist_db,))
            result = cursor.fetchall()
            names = [row[0] for row in result]
            rolls = [row[1] for row in result]
            times = [row[2] for row in result]
            l = len(result)
            print(f"Retrieved {l} attendance records for today from DB.")
            return names, rolls, times, l
    try:
        return execute_with_retry(_extract_attendance)
    except Exception as e:
        print(f"Error extracting attendance from DB: {e}")
        traceback.print_exc()
        try: # Fallback to CSV
            df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
            names, rolls, times, l = df['Name'].tolist(), df['Roll'].tolist(), df['Time'].tolist(), len(df)
            print(f"Retrieved {l} attendance records from CSV.")
            return names, rolls, times, l
        except Exception as csv_e:
            print(f"Failed to read attendance from CSV: {csv_e}")
            return [], [], [], 0

def extract_face(img_pil, box, image_size=160, margin=20):
    try:
        x1, y1, x2, y2 = box
        size_bb = max(x2-x1, y2-y1)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        size = int(size_bb * 1.0 + margin * 2) # Ensure size is int
        
        x1_new = max(int(center_x - size // 2), 0)
        y1_new = max(int(center_y - size // 2), 0)
        x2_new = min(int(center_x + size // 2), img_pil.width)
        y2_new = min(int(center_y + size // 2), img_pil.height)
        
        face = img_pil.crop((x1_new, y1_new, x2_new, y2_new))
        face = face.resize((image_size, image_size), Image.Resampling.BILINEAR) # Updated Resampling
        
        face_np = np.array(face, dtype=np.float32) / 255.0
        face_tensor = torch.from_numpy(face_np).permute(2, 0, 1)
        return face_tensor.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error extracting face: {e}")
        traceback.print_exc()
        return None

def process_face_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            img = Image.fromarray(img_np)
        
        boxes, _ = detector.detect(img)
        if boxes is None or len(boxes) == 0:
            print(f"No face detected in {image_path}")
            return None
        box = boxes[0]
        return extract_face(img, [int(i) for i in box])
    except Exception as e:
        print(f"Error processing face image {image_path}: {e}")
        traceback.print_exc()
        return None

def identify_face(face_tensor, threshold=0.6):
    try:
        embeddings_dict = get_all_embeddings()
        if not embeddings_dict: return "Unknown_0"
        
        embedding = resnet(face_tensor).detach() # Embedding is already on device
        
        best_match, best_score = "Unknown_0", 0.0
        for name_id, ref_embedding in embeddings_dict.items():
            similarity = torch.nn.functional.cosine_similarity(embedding, ref_embedding).item()
            if similarity > best_score:
                best_score, best_match = similarity, name_id
        
        print(f"Best match: {best_match} with score {best_score:.4f}")
        return best_match if best_score >= threshold else "Unknown_0"
    except Exception as e:
        print(f"Error identifying face: {e}")
        traceback.print_exc()
        return "Unknown_0"

# Routes
@app.route('/')
def home():
    return redirect(url_for('attendance_page'))

@app.route('/attendance')
def attendance_page():
    names, rolls, times, l = extract_attendance()    
    return render_template('attendance.html', names=names, rolls=rolls, times=times, l=l, datetoday2=datetoday2, mess=request.args.get('mess'))

@app.route('/user-management')
def user_management_page():
    return render_template('user_management.html', totalreg=get_total_users(), datetoday2=datetoday2, mess=request.args.get('mess'))

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_route(): # Renamed to avoid conflict
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data: return jsonify({'success': False, 'message': 'No image data provided'})
        
        try:
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            img = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error decoding image: {str(e)}'})
        
        try:
            boxes, _ = detector.detect(img)
            if boxes is None or len(boxes) == 0:
                return jsonify({'success': False, 'message': 'No face detected in image'})
            
            box = boxes[0]
            face_tensor = extract_face(img, [int(i) for i in box])
            if face_tensor is None:
                 return jsonify({'success': False, 'message': 'Could not extract face'})

            identity = identify_face(face_tensor)
            name, user_id_str = identity.split('_')[0], identity.split('_')[1]
            
            if name == "Unknown":
                return jsonify({'success': False, 'message': 'Unknown person detected'})
            
            marked, time_marked = add_attendance(name, user_id_str)
            if marked:
                return jsonify({'success': True, 'message': f'Attendance marked for {name}', 'name': name, 'user_id': user_id_str, 'time': time_marked})
            else:
                return jsonify({'success': False, 'message': f'Attendance already marked for {name} today'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error in processing: {str(e)}'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/get_attendance')
def get_attendance_route(): # Renamed
    names, rolls, times, l = extract_attendance()
    return jsonify([{'name': names[i], 'roll': rolls[i], 'time': times[i]} for i in range(l)])

@app.route('/get_all_users')
def get_users_route(): # Renamed
    return jsonify({'success': True, 'users': get_all_users_from_db()})

@app.route('/register_user', methods=['POST'])
def register_user_route(): # Renamed
    try:
        data = request.json
        username, userid = data.get('username'), data.get('userid')
        if not username or not userid:
            return jsonify({'success': False, 'message': 'Username and user ID are required'})
        if not add_user_to_db(username, userid):
            return jsonify({'success': False, 'message': 'User ID already exists'})
        
        os.makedirs(f'static/faces/{username}_{userid}', exist_ok=True)
        os.makedirs(f'static/temp_embeddings/{username}_{userid}', exist_ok=True)
        return jsonify({'success': True, 'message': 'User registered'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/delete_user', methods=['POST'])
def delete_user_route(): # Renamed
    try:
        user_id = request.json.get('user_id')
        if not user_id: return jsonify({'success': False, 'message': 'User ID is required'})
        if delete_user_from_db(user_id):
            return jsonify({'success': True, 'message': 'User deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'User not found or could not be deleted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/capture_image', methods=['POST'])
def capture_image_route(): # Renamed
    try:
        if 'face_image' not in request.files:
            return jsonify({'success': False, 'message': 'No image file'})
        
        username, userid = request.form.get('username'), request.form.get('userid')
        image_count = int(request.form.get('image_count', '0'))
        
        if not username or not userid:
            return jsonify({'success': False, 'message': 'Username and user ID are required'})
        
        image_file = request.files['face_image']
        userdir = f'static/faces/{username}_{userid}'
        os.makedirs(userdir, exist_ok=True)
        image_path = os.path.join(userdir, f'{username}_{image_count}.jpg')
        image_file.save(image_path)
        
        face_tensor = process_face_image(image_path)
        if face_tensor is None:
            if os.path.exists(image_path): os.remove(image_path)
            return jsonify({'success': False, 'message': 'No face detected in the image'})
        
        embedding = resnet(face_tensor).detach() # Embedding is already on device
        
        temp_embedding_dir = f'static/temp_embeddings/{username}_{userid}'
        os.makedirs(temp_embedding_dir, exist_ok=True)
        torch.save(embedding, os.path.join(temp_embedding_dir, f'embedding_{image_count}.pt'))
        
        return jsonify({'success': True, 'message': 'Image saved'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/export_attendance')
def export_attendance_route(): # Renamed
    try:
        names, rolls, times, l = extract_attendance()
        if l == 0: return jsonify({'success': False, 'message': 'No attendance records for today'})
        
        output = BytesIO()
        output.write('Name,ID,Time\n'.encode('utf-8'))
        for i in range(l):
            output.write(f'{names[i]},{rolls[i]},{times[i]}\n'.encode('utf-8'))
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name=f"attendance_{datetoday}.csv")
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    
@app.route('/get_attendance_history', methods=['POST'])
def get_attendance_history_route(): # Renamed
    try:
        user_id = request.json.get('user_id')
        if not user_id: return jsonify({'success': False, 'message': 'User ID is required'})
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT date, time FROM attendance WHERE user_id = ? ORDER BY date DESC, time DESC LIMIT 30", (user_id,))
            history = [{'date': row[0], 'time': row[1]} for row in cursor.fetchall()]
            return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/complete_registration', methods=['POST'])
def complete_registration_route(): # Renamed
    try:
        username, userid = request.json.get('username'), request.json.get('userid')
        if not username or not userid:
            return jsonify({'success': False, 'message': 'Username and user ID are required'})
        
        temp_embedding_dir = f'static/temp_embeddings/{username}_{userid}'
        if not os.path.exists(temp_embedding_dir):
            return jsonify({'success': False, 'message': 'No embeddings found'})
        
        embedding_files = [f for f in os.listdir(temp_embedding_dir) if f.endswith('.pt')]
        if not embedding_files: return jsonify({'success': False, 'message': 'No valid embeddings found'})
        
        all_embeddings = [torch.load(os.path.join(temp_embedding_dir, f)).to(device) for f in embedding_files] # Move to device
        
        if not all_embeddings: return jsonify({'success': False, 'message': 'Failed to load embeddings'})

        stacked_embeddings = torch.cat(all_embeddings, dim=0)
        avg_embedding = torch.mean(stacked_embeddings, dim=0, keepdim=True)
        
        if save_embedding(userid, avg_embedding):
            print(f"Successfully saved face embedding for user {username}_{userid}")
            shutil.rmtree(temp_embedding_dir, ignore_errors=True)
            return jsonify({'success': True, 'message': 'Registration completed successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to save embeddings to database'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    # The SSL context path should be correct for your environment.
    # If local_ssl_setup.py was run in the project root, it would create 'ssl/cert.pem' and 'ssl/key.pem'
    # Example for relative paths: ssl_context=('ssl/cert.pem', 'ssl/key.pem')
    # Using the user-provided absolute path:
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('/home/ayyappa/Documents/FaceRec-attendance/ssl/cert.pem', '/home/ayyappa/Documents/FaceRec-attendance/ssl/key.pem'))