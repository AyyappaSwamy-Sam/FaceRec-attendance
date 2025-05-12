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
datetoday = now_ist.strftime("%m_%d_%y") # Used for CSV filename (today's CSV)
datetoday2_default = now_ist.strftime("%d-%B-%Y") # Default display date (today)

# Create required directories
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
os.makedirs('static/embeddings', exist_ok=True)
os.makedirs('static/images', exist_ok=True) 
os.makedirs('static/temp_embeddings', exist_ok=True)

# Ensure today's attendance file exists (for CSV fallback)
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect('face_attendance.db', timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    except Exception as e:
        print(f"Database connection error: {e}")
        traceback.print_exc()
        if conn: conn.rollback()
        raise
    finally:
        if conn:
            conn.commit()
            conn.close()

def execute_with_retry(operation, max_retries=5, retry_delay=1.0):
    retries = 0
    while retries < max_retries:
        try:
            return operation()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and retries < max_retries - 1:
                retries += 1
                print(f"Database locked, retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else: raise
        except Exception as e:
            print(f"Database operation error: {e}")
            traceback.print_exc()
            raise

def init_db():
    def _init():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, user_id TEXT NOT NULL UNIQUE, registration_date TEXT NOT NULL)')
            cursor.execute('CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, embedding BLOB NOT NULL, FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE)')
            cursor.execute('CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, date TEXT NOT NULL, time TEXT NOT NULL, FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE)')
        print("Database initialized successfully.")
    try:
        execute_with_retry(_init)
    except Exception as e:
        print(f"Error initializing database: {e}")
        traceback.print_exc()

init_db()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

detector = MTCNN(image_size=160, margin=20, min_face_size=20, thresholds=[0.5, 0.6, 0.6], factor=0.709, post_process=True, device=device, select_largest=False, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Face recognition models loaded successfully.")

def format_time_to_12hr(time_str_24hr):
    """Converts HH:MM:SS to HH:MM:SS AM/PM."""
    if not time_str_24hr:
        return ""
    try:
        dt_obj = datetime.strptime(time_str_24hr, "%H:%M:%S")
        return dt_obj.strftime("%I:%M:%S %p")
    except ValueError:
        # If already in 12hr or other format, return as is (or handle error)
        return time_str_24hr

def extract_attendance(target_date_str=None):
    """
    Extracts attendance records for a given date.
    If target_date_str is None, fetches for today's date (IST).
    Times are formatted to 12-hour AM/PM.
    """
    def _extract_attendance_for_date(date_to_query_str_db):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, user_id, time FROM attendance WHERE date = ? ORDER BY time",
                (date_to_query_str_db,)
            )
            result = cursor.fetchall()
            names = [row[0] for row in result]
            rolls = [row[1] for row in result]
            times = [format_time_to_12hr(row[2]) for row in result] # Format time here
            l = len(result)
            print(f"Retrieved {l} attendance records for {date_to_query_str_db} from DB.")
            return names, rolls, times, l

    if target_date_str:
        try:
            # Validate and use the provided date string (expected YYYY-MM-DD)
            datetime.strptime(target_date_str, "%Y-%m-%d") 
            date_to_query_db = target_date_str
        except ValueError:
            print(f"Invalid date format for target_date_str: {target_date_str}. Defaulting to today.")
            date_to_query_db = datetime.now(ist).strftime("%Y-%m-%d")
    else:
        # Default to today's date (IST) for database query
        date_to_query_db = datetime.now(ist).strftime("%Y-%m-%d")

    try:
        return execute_with_retry(lambda: _extract_attendance_for_date(date_to_query_db))
    except Exception as e:
        print(f"Error extracting attendance for {date_to_query_db} from DB: {e}")
        # Fallback to CSV only if querying for today and DB fails
        if date_to_query_db == datetime.now(ist).strftime("%Y-%m-%d"): # datetoday is IST based for CSV
            try:
                df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
                names = df['Name'].tolist()
                rolls = df['Roll'].tolist()
                times = [format_time_to_12hr(t) for t in df['Time'].tolist()]
                l = len(df)
                print(f"Retrieved {l} attendance records for today from CSV as fallback.")
                return names, rolls, times, l
            except Exception as csv_e:
                print(f"Failed to read attendance from CSV: {csv_e}")
        return [], [], [], 0


def get_total_users():
    def _get_count():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            return cursor.fetchone()[0]
    try: return execute_with_retry(_get_count)
    except Exception as e: print(f"Error getting user count: {e}"); return 0

def get_all_users_from_db():
    def _get_users():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, user_id, registration_date FROM users ORDER BY name")
            return [{'name': r[0], 'user_id': r[1], 'registration_date': r[2]} for r in cursor.fetchall()]
    try: return execute_with_retry(_get_users)
    except Exception as e: print(f"Error getting users: {e}"); traceback.print_exc(); return []

def add_user_to_db(name, user_id):
    def _add():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            reg_date = datetime.now(ist).strftime("%Y-%m-%d")
            cursor.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
            if cursor.fetchone(): return False # User exists
            cursor.execute("INSERT INTO users (name, user_id, registration_date) VALUES (?, ?, ?)", (name, user_id, reg_date))
            return True
    try: return execute_with_retry(_add)
    except sqlite3.IntegrityError: return False
    except Exception as e: print(f"Error adding user: {e}"); traceback.print_exc(); return False

def delete_user_from_db(user_id):
    def _delete():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users WHERE user_id = ?", (user_id,))
            res = cursor.fetchone()
            if not res: return False, None
            user_name = res[0]
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,)) # Cascade delete handles others
            return cursor.rowcount > 0, user_name
    try:
        deleted, user_name = execute_with_retry(_delete)
        if deleted and user_name:
            for d in [f'static/faces/{user_name}_{user_id}', f'static/temp_embeddings/{user_name}_{user_id}']:
                if os.path.exists(d): shutil.rmtree(d, ignore_errors=True)
            # Optional: Update CSVs (DB is primary source of truth now)
        return deleted
    except Exception as e: print(f"Error deleting user: {e}"); traceback.print_exc(); return False

def save_embedding(user_id, embedding):
    def _save():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            emb_bytes = embedding.cpu().numpy().tobytes()
            cursor.execute("INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)", (user_id, emb_bytes))
            return True
    try: return execute_with_retry(_save)
    except Exception as e: print(f"Error saving embedding: {e}"); traceback.print_exc(); return False

def get_all_embeddings():
    def _get():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT u.name, u.user_id, e.embedding FROM users u JOIN embeddings e ON u.user_id = e.user_id")
            emb_dict = {}
            for name, uid, emb_bytes in cursor.fetchall():
                try:
                    arr = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, -1)
                    emb_dict[f"{name}_{uid}"] = torch.from_numpy(arr).to(device)
                except Exception as e: print(f"Error processing embedding for {name}_{uid}: {e}")
            return emb_dict
    try: return execute_with_retry(_get)
    except Exception as e: print(f"Error getting all embeddings: {e}"); traceback.print_exc(); return {}

def add_attendance(name, user_id):
    def _add():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(ist)
            date_db = now.strftime("%Y-%m-%d")
            time_db_24hr = now.strftime("%H:%M:%S")
            
            cursor.execute("SELECT 1 FROM attendance WHERE user_id = ? AND date = ?", (user_id, date_db))
            if cursor.fetchone(): return False, None # Already marked
            
            cursor.execute("INSERT INTO attendance (user_id, name, date, time) VALUES (?, ?, ?, ?)", (user_id, name, date_db, time_db_24hr))
            
            # CSV (datetoday is IST based for filename)
            csv_path = f'Attendance/Attendance-{datetoday}.csv'
            if not os.path.exists(csv_path):
                with open(csv_path, 'w') as f: f.write('Name,Roll,Time\n')
            with open(csv_path, 'a') as f: f.write(f'{name},{user_id},{time_db_24hr}\n')
            
            return True, format_time_to_12hr(time_db_24hr) # Return formatted time
    try: return execute_with_retry(_add)
    except Exception as e: print(f"Error adding attendance: {e}"); traceback.print_exc(); return False, None

def extract_face(img_pil, box, image_size=160, margin=20):
    try:
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        size_bb = max(x2 - x1, y2 - y1)
        size = int(size_bb * 1.0 + margin * 2)
        
        x1_new = max(int(center_x - size // 2), 0)
        y1_new = max(int(center_y - size // 2), 0)
        x2_new = min(int(center_x + size // 2), img_pil.width)
        y2_new = min(int(center_y + size // 2), img_pil.height)
        
        face = img_pil.crop((x1_new, y1_new, x2_new, y2_new)).resize((image_size, image_size), Image.Resampling.BILINEAR)
        face_np = np.array(face, dtype=np.float32) / 255.0
        return torch.from_numpy(face_np).permute(2, 0, 1).unsqueeze(0).to(device)
    except Exception as e: print(f"Error in extract_face: {e}"); traceback.print_exc(); return None

def process_face_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        # Optional: Histogram equalization (can sometimes help, sometimes not)
        # img_np = np.array(img)
        # if img_np.shape[2] == 3:
        #     img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        #     img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        #     img = Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB))
        
        boxes, _ = detector.detect(img) # boxes is None or a numpy array (num_faces, 4)
        
        # --- START OF CORRECTION for process_face_image ---
        if boxes is None or (isinstance(boxes, np.ndarray) and boxes.shape[0] == 0):
            print(f"No face detected by MTCNN in {image_path}")
            return None
        # --- END OF CORRECTION for process_face_image ---
        
        # If we reach here, 'boxes' is a non-empty NumPy array.
        # Process the first detected face
        box = boxes[0] 
        return extract_face(img, [int(c) for c in box]) # Pass the PIL image 'img'
    except Exception as e: 
        print(f"Error processing image {image_path}: {e}")
        traceback.print_exc()
        return None

def identify_face(face_tensor, threshold=0.6):
    try:
        all_embs = get_all_embeddings()
        if not all_embs: return "Unknown_0"
        
        current_emb = resnet(face_tensor).detach()
        best_match, best_score = "Unknown_0", 0.0
        
        for name_id, ref_emb in all_embs.items():
            sim = torch.nn.functional.cosine_similarity(current_emb, ref_emb).item()
            if sim > best_score:
                best_score, best_match = sim, name_id
        
        return best_match if best_score >= threshold else "Unknown_0"
    except Exception as e: print(f"Error identifying face: {e}"); traceback.print_exc(); return "Unknown_0"

# --- Flask Routes ---
@app.route('/')
def home():
    return redirect(url_for('attendance_page'))

@app.route('/attendance')
def attendance_page():
    selected_date_str = request.args.get('date') # YYYY-MM-DD from date input
    
    names, rolls, times, l = extract_attendance(selected_date_str)
    
    # Determine display_date for the title/header and date picker value
    if selected_date_str:
        try:
            current_display_date_obj = datetime.strptime(selected_date_str, "%Y-%m-%d")
            display_date_formatted = current_display_date_obj.strftime("%d-%B-%Y")
            date_for_picker = selected_date_str
        except ValueError: # Invalid date in query
            display_date_formatted = datetoday2_default # Fallback to today
            date_for_picker = datetime.now(ist).strftime("%Y-%m-%d")
    else: # No date in query, default to today
        display_date_formatted = datetoday2_default
        date_for_picker = datetime.now(ist).strftime("%Y-%m-%d")

    return render_template('attendance.html',
                           names=names, rolls=rolls, times=times, l=l,
                           datetoday2_display=display_date_formatted, # For panel header
                           selected_date_for_picker=date_for_picker, # For date input value
                           mess=request.args.get('mess'))

@app.route('/user-management')
def user_management_page():
    return render_template('user_management.html', totalreg=get_total_users(), datetoday2=datetoday2_default, mess=request.args.get('mess'))

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_route(): # Renamed to avoid conflict
    try:
        data = request.json
        img_data_uri = data.get('image')
        if not img_data_uri: 
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        try:
            # Decode base64 image: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
            header, encoded = img_data_uri.split(",", 1)
            img = Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB')
        except Exception as e:
            print(f"Error decoding image: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Error decoding image: {str(e)}'})
        
        try:
            boxes, _ = detector.detect(img) # boxes is None or a numpy array (num_faces, 4)
            
            if boxes is None or (isinstance(boxes, np.ndarray) and boxes.shape[0] == 0):
                print("No face detected by MTCNN.")
                return jsonify({'success': False, 'message': 'No face detected in image'})
            
            box = boxes[0] 
            face_tensor = extract_face(img, [int(c) for c in box]) # Pass the PIL image 'img'
            
            if face_tensor is None:
                 print("Could not extract face features from detected box.")
                 return jsonify({'success': False, 'message': 'Could not extract face features'})

            identity = identify_face(face_tensor)
            name, user_id_str = identity.split('_')[0], identity.split('_')[1] 
            
            if name == "Unknown":
                print(f"Unknown person detected. User ID part: {user_id_str}")
                return jsonify({'success': False, 'message': 'Unknown person detected'})
            
            marked, time_marked_12hr = add_attendance(name, user_id_str) 
            if marked:
                return jsonify({'success': True, 'message': f'Attendance marked for {name}', 'name': name, 'user_id': user_id_str, 'time': time_marked_12hr})
            else:
                return jsonify({'success': False, 'message': f'Attendance already marked for {name} today'})
        except Exception as e:
            print(f"Error during face detection/recognition or attendance marking: {e}")
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Error in processing: {str(e)}'})
            
    except Exception as e:
        print(f"Overall error in /mark_attendance: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'An unexpected error occurred: {str(e)}'})

@app.route('/get_attendance')
def get_attendance_route():
    # This live update will always fetch for *today's* date.
    # The main table display can show historical data, but live pings are for current day.
    names, rolls, times, l = extract_attendance() # Default to today
    return jsonify([{'name': n, 'roll': r, 'time': t} for n, r, t in zip(names, rolls, times)])

@app.route('/get_all_users')
def get_users_route():
    return jsonify({'success': True, 'users': get_all_users_from_db()})

@app.route('/register_user', methods=['POST'])
def register_user_route():
    try:
        data = request.json
        username, userid = data.get('username'), data.get('userid')
        if not (username and userid): return jsonify({'success': False, 'message': 'Name and ID required'})
        if not add_user_to_db(username, userid): return jsonify({'success': False, 'message': 'User ID exists'})
        
        os.makedirs(f'static/faces/{username}_{userid}', exist_ok=True)
        os.makedirs(f'static/temp_embeddings/{username}_{userid}', exist_ok=True)
        return jsonify({'success': True, 'message': 'User registered'})
    except Exception as e: print(f"Error in /register_user: {e}"); return jsonify({'success': False, 'message': str(e)})

@app.route('/delete_user', methods=['POST'])
def delete_user_route():
    try:
        user_id = request.json.get('user_id')
        if not user_id: return jsonify({'success': False, 'message': 'User ID required'})
        if delete_user_from_db(user_id):
            return jsonify({'success': True, 'message': 'User deleted'})
        return jsonify({'success': False, 'message': 'User not found or delete failed'})
    except Exception as e: print(f"Error in /delete_user: {e}"); return jsonify({'success': False, 'message': str(e)})

@app.route('/capture_image', methods=['POST'])
def capture_image_route(): # Renamed
    try:
        if 'face_image' not in request.files: 
            return jsonify({'success': False, 'message': 'No image file'})
        
        username, userid = request.form.get('username'), request.form.get('userid')
        img_count = int(request.form.get('image_count', '0')) # Renamed from image_count to img_count
        
        if not (username and userid): 
            return jsonify({'success': False, 'message': 'Username and user ID are required'})
        
        img_file = request.files['face_image'] # Renamed from image_file to img_file
        user_dir = f'static/faces/{username}_{userid}' # Renamed from userdir to user_dir
        os.makedirs(user_dir, exist_ok=True)
        img_path = os.path.join(user_dir, f'{username}_{img_count}.jpg') # Renamed from image_path to img_path
        img_file.save(img_path)
        
        face_tensor = process_face_image(img_path) # This function now has the fix
        if face_tensor is None:
            if os.path.exists(img_path): os.remove(img_path)
            return jsonify({'success': False, 'message': 'No face detected in the image'})
        
        emb = resnet(face_tensor).detach() # Renamed from embedding to emb
        
        temp_emb_dir = f'static/temp_embeddings/{username}_{userid}' # Renamed from temp_embedding_dir to temp_emb_dir
        os.makedirs(temp_emb_dir, exist_ok=True)
        torch.save(emb, os.path.join(temp_emb_dir, f'embedding_{img_count}.pt'))
        
        return jsonify({'success': True, 'message': 'Image saved'})
    except Exception as e: 
        print(f"Error in /capture_image: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})
    
    
@app.route('/export_attendance')
def export_attendance_route():
    try:
        export_date_str = request.args.get('date') # YYYY-MM-DD
        
        # Determine filename based on whether a specific date is requested or today's
        filename_date_part = export_date_str if export_date_str else datetime.now(ist).strftime("%Y-%m-%d")
        
        names, rolls, times, l = extract_attendance(export_date_str) # times are already 12hr formatted
        
        if l == 0:
            flash_message = f"No attendance records found for {datetime.strptime(filename_date_part, '%Y-%m-%d').strftime('%d-%B-%Y')}."
            return redirect(url_for('attendance_page', mess=flash_message, date=export_date_str if export_date_str else ''))

        output = BytesIO()
        # Use consistent timezone info in CSV header
        output.write('Name,ID,Time (IST)\n'.encode('utf-8'))
        for i in range(l):
            output.write(f'{names[i]},{rolls[i]},{times[i]}\n'.encode('utf-8'))
        output.seek(0)
        
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name=f"attendance_{filename_date_part}.csv")
    except Exception as e: print(f"Error in /export_attendance: {e}"); traceback.print_exc(); return jsonify({'success': False, 'message': str(e)})
    
@app.route('/get_attendance_history', methods=['POST'])
def get_attendance_history_route():
    try:
        user_id = request.json.get('user_id')
        if not user_id: return jsonify({'success': False, 'message': 'User ID required'})
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Fetch raw time, format later
            cursor.execute("SELECT date, time FROM attendance WHERE user_id = ? ORDER BY date DESC, time DESC LIMIT 30", (user_id,))
            history = [{'date': r[0], 'time': format_time_to_12hr(r[1])} for r in cursor.fetchall()]
            return jsonify({'success': True, 'history': history})
    except Exception as e: print(f"Error in /get_attendance_history: {e}"); return jsonify({'success': False, 'message': str(e)})

@app.route('/complete_registration', methods=['POST'])
def complete_registration_route():
    try:
        username, userid = request.json.get('username'), request.json.get('userid')
        if not (username and userid): return jsonify({'success': False, 'message': 'Name and ID required'})
        
        temp_emb_dir = f'static/temp_embeddings/{username}_{userid}'
        if not os.path.exists(temp_emb_dir): return jsonify({'success': False, 'message': 'No embeddings found'})
        
        emb_files = [f for f in os.listdir(temp_emb_dir) if f.endswith('.pt')]
        if not emb_files: return jsonify({'success': False, 'message': 'No valid embeddings'})
            
        all_embs = [torch.load(os.path.join(temp_emb_dir, f)).to(device) for f in emb_files]
        if not all_embs: return jsonify({'success': False, 'message': 'Failed to load embeddings'})

        avg_emb = torch.mean(torch.cat(all_embs, dim=0), dim=0, keepdim=True)
        
        if save_embedding(userid, avg_emb):
            shutil.rmtree(temp_emb_dir, ignore_errors=True)
            return jsonify({'success': True, 'message': 'Registration completed'})
        return jsonify({'success': False, 'message': 'Failed to save embeddings'})
    except Exception as e: print(f"Error in /complete_registration: {e}"); traceback.print_exc(); return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('/home/ayyappa/Documents/FaceRec-attendance/ssl/cert.pem', '/home/ayyappa/Documents/FaceRec-attendance/ssl/key.pem'))