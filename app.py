import os
import cv2
import numpy as np
import pandas as pd
import time
import json
import shutil
from datetime import date, datetime, timedelta # Added timedelta
import sqlite3 # Will be replaced by mysql.connector
import mysql.connector # Import MySQL connector
from mysql.connector import Error as MySQLError # For MySQL specific errors
from mysql.connector import IntegrityError as MySQLIntegrityError # For MySQL integrity errors

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
app.secret_key = 'your_secret_key'

# --- MySQL Database Configuration ---
DB_CONFIG = {
    'host': '157.51.243.100',
    'user': 'myclassboardai_faceuser',
    'password': 'M3KlLdFC7;Zj',
    'database': 'myclassboardai_facedb',
    'port': 3306 
}
# --- End MySQL Database Configuration ---

try:
    ist = pytz.timezone('Asia/Kolkata')
except pytz.exceptions.UnknownTimeZoneError:
    print("Error: Timezone 'Asia/Kolkata' not found. Ensure pytz is installed correctly.")
    ist = pytz.utc

now_ist = datetime.now(ist)
datetoday = now_ist.strftime("%m_%d_%y")
datetoday2_default = now_ist.strftime("%d-%B-%Y")

os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
os.makedirs('static/embeddings', exist_ok=True)
os.makedirs('static/images', exist_ok=True) 
os.makedirs('static/temp_embeddings', exist_ok=True)

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')

@contextmanager
def get_db_connection():
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        # MySQL enables foreign keys by default if defined at table creation.
        # No PRAGMA needed like in SQLite.
        # autocommit can be set to True if you prefer, or manage commits manually.
        # conn.autocommit = True 
        cursor = conn.cursor()
        yield conn, cursor # Yield both connection and cursor
    except MySQLError as e:
        print(f"MySQL Database connection error: {e}")
        traceback.print_exc()
        # No rollback needed here if autocommit is False and no operations were done before error
        raise
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            # conn.commit() # Commit if autocommit is False and operations were successful
            conn.close()
            # print("MySQL connection closed.")


def execute_with_retry(operation, max_retries=3, retry_delay=2.0): # Reduced retries for network DB
    retries = 0
    while retries < max_retries:
        try:
            return operation()
        # Catching generic MySQLError. Specific lock errors are harder to pinpoint across versions/configs.
        # MySQL's InnoDB usually handles row-level locking well.
        except MySQLError as e:
            # Common MySQL errors that might warrant a retry:
            # 1205: Lock wait timeout exceeded
            # 1213: Deadlock found when trying to get lock
            # 2006: MySQL server has gone away (might be a temporary network issue)
            # 2013: Lost connection to MySQL server during query
            if e.errno in [1205, 1213, 2006, 2013] and retries < max_retries - 1:
                retries += 1
                print(f"MySQL Error ({e.errno}): {e}. Retrying in {retry_delay}s... (Attempt {retries}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 1.5 
            else:
                print(f"MySQL Database operation error (final or non-retryable): {e}")
                traceback.print_exc()
                raise # Re-raise the exception if it's not a retryable one or max retries reached
        except Exception as e: # Catch other potential errors
            print(f"General Database operation error: {e}")
            traceback.print_exc()
            raise


def init_db():
    def _init():
        with get_db_connection() as (conn, cursor):
            # MySQL uses %s as placeholder, not ?
            # AUTO_INCREMENT is MySQL's equivalent of SQLite's INTEGER PRIMARY KEY auto-increment
            # VARCHAR for strings, DATE for dates, TIME for times, BLOB for binary data
            # Ensure user_id in users is sufficiently long if it's a string.
            # Using TEXT for user_id for flexibility, or VARCHAR(255)
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL UNIQUE,
                registration_date DATE NOT NULL
            ) ENGINE=InnoDB; 
            ''') # Specify InnoDB for foreign key support
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                name VARCHAR(255) NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            ''')
            conn.commit() # Explicit commit after DDL statements
        print("MySQL Database tables checked/created successfully.")
    try:
        execute_with_retry(_init)
    except Exception as e:
        print(f"Error initializing MySQL database: {e}")

init_db()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

detector = MTCNN(image_size=160, margin=20, min_face_size=20, thresholds=[0.5, 0.6, 0.6], factor=0.709, post_process=True, device=device, select_largest=False, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Face recognition models loaded successfully.")

def format_time_to_12hr(time_obj_or_str):
    if not time_obj_or_str: return ""
    if isinstance(time_obj_or_str, str): # If it's a string like "HH:MM:SS"
        try:
            time_obj = datetime.strptime(time_obj_or_str, "%H:%M:%S").time()
        except ValueError:
            return time_obj_or_str # Return as is if parsing fails
    elif isinstance(time_obj_or_str, timedelta): # MySQL TIME type is often read as timedelta
         # Convert timedelta to total seconds, then to HH:MM:SS string
        total_seconds = int(time_obj_or_str.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_obj = datetime.strptime(f"{hours:02}:{minutes:02}:{seconds:02}", "%H:%M:%S").time()
    elif hasattr(time_obj_or_str, 'strftime'): # If it's already a time or datetime object
        time_obj = time_obj_or_str
    else:
        return str(time_obj_or_str) # Fallback

    return time_obj.strftime("%I:%M:%S %p")


def extract_attendance(target_date_str=None):
    def _extract(date_to_query_db):
        with get_db_connection() as (conn, cursor):
            # Use %s for MySQL placeholders
            query = "SELECT name, user_id, time FROM attendance WHERE date = %s ORDER BY time"
            cursor.execute(query, (date_to_query_db,))
            res = cursor.fetchall()
            # MySQL TIME type might be returned as datetime.timedelta by mysql-connector
            return [r[0] for r in res], [r[1] for r in res], [format_time_to_12hr(r[2]) for r in res], len(res)

    date_to_query = target_date_str or datetime.now(ist).strftime("%Y-%m-%d")
    if target_date_str:
        try: datetime.strptime(target_date_str, "%Y-%m-%d")
        except ValueError: date_to_query = datetime.now(ist).strftime("%Y-%m-%d")
    
    try:
        names, rolls, times, l = execute_with_retry(lambda: _extract(date_to_query))
        print(f"Retrieved {l} records for {date_to_query} from MySQL DB.")
        return names, rolls, times, l
    except Exception as e:
        print(f"Error extracting for {date_to_query} from MySQL DB: {e}")
        if date_to_query == datetime.now(ist).strftime("%Y-%m-%d"):
            try:
                df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
                print(f"Retrieved {len(df)} records for today from CSV fallback.")
                return df['Name'].tolist(), df['Roll'].tolist(), [format_time_to_12hr(t) for t in df['Time'].tolist()], len(df)
            except Exception as csv_e: print(f"CSV fallback failed: {csv_e}")
        return [], [], [], 0

def get_total_users():
    def _get():
        with get_db_connection() as (conn, cursor):
            cursor.execute("SELECT COUNT(*) FROM users")
            return cursor.fetchone()[0]
    try: return execute_with_retry(_get)
    except: return 0

def get_all_users_from_db():
    def _get_users():
        with get_db_connection() as (conn, cursor):
            cursor.execute("SELECT name, user_id, registration_date FROM users ORDER BY name")
            users_data = []
            for row in cursor.fetchall():
                name, user_id, reg_date_obj = row[0], row[1], row[2]
                reg_date_str = reg_date_obj.strftime("%Y-%m-%d") if isinstance(reg_date_obj, date) else str(reg_date_obj)
                
                first_image_filename = f"{name}_0.jpg" # Assuming simple name
                first_image_path_relative = f"static/faces/{name}_{user_id}/{first_image_filename}"
                image_url = url_for('static', filename=f'faces/{name}_{user_id}/{first_image_filename}') if os.path.exists(first_image_path_relative) else None
                
                users_data.append({
                    'name': name, 'user_id': user_id, 
                    'registration_date': reg_date_str, 'image_url': image_url 
                })
            return users_data
    try: return execute_with_retry(_get_users)
    except Exception as e: print(f"Error getting users: {e}"); traceback.print_exc(); return []

def add_user_to_db(name, user_id):
    def _add():
        with get_db_connection() as (conn, cursor):
            reg_date = datetime.now(ist).strftime("%Y-%m-%d")
            cursor.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,))
            if cursor.fetchone(): return False
            cursor.execute("INSERT INTO users (name, user_id, registration_date) VALUES (%s, %s, %s)", 
                           (name, user_id, reg_date))
            conn.commit()
            return True
    try: return execute_with_retry(_add)
    except MySQLIntegrityError: # Catch MySQL specific integrity error
        print(f"MySQLIntegrityError: User ID {user_id} likely already exists.")
        return False
    except Exception as e: print(f"Error adding user: {e}"); traceback.print_exc(); return False

def delete_user_from_db(user_id):
    def _delete():
        with get_db_connection() as (conn, cursor):
            cursor.execute("SELECT name FROM users WHERE user_id = %s", (user_id,))
            res = cursor.fetchone()
            if not res: return False, None
            user_name = res[0]
            # ON DELETE CASCADE should handle related records in InnoDB tables
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count > 0, user_name
    try:
        deleted, user_name_for_cleanup = execute_with_retry(_delete)
        if deleted and user_name_for_cleanup:
            for d in [f'static/faces/{user_name_for_cleanup}_{user_id}', f'static/temp_embeddings/{user_name_for_cleanup}_{user_id}']:
                if os.path.exists(d): shutil.rmtree(d, ignore_errors=True)
        return deleted
    except Exception as e:
        print(f"Error deleting user {user_id}: {e}")
        traceback.print_exc()
        return False

def save_embedding(user_id, embedding):
    def _save():
        with get_db_connection() as (conn, cursor):
            emb_bytes = embedding.cpu().numpy().tobytes()
            cursor.execute("INSERT INTO embeddings (user_id, embedding) VALUES (%s, %s)", (user_id, emb_bytes))
            conn.commit()
            return True
    try: return execute_with_retry(_save)
    except Exception as e: print(f"Error saving embedding: {e}"); traceback.print_exc(); return False

def get_all_embeddings():
    def _get():
        with get_db_connection() as (conn, cursor):
            emb_dict = {}
            cursor.execute("SELECT u.name, u.user_id, e.embedding FROM users u JOIN embeddings e ON u.user_id = e.user_id")
            for name, uid, b in cursor.fetchall():
                try: emb_dict[f"{name}_{uid}"] = torch.from_numpy(np.frombuffer(b, dtype=np.float32).reshape(1, -1)).to(device)
                except: pass
            return emb_dict
    try: return execute_with_retry(_get)
    except: return {}

def add_attendance(name, user_id):
    def _add():
        with get_db_connection() as (conn, cursor):
            now = datetime.now(ist)
            date_db = now.strftime("%Y-%m-%d")
            time_db_24hr = now.strftime("%H:%M:%S")

            cursor.execute(
                "SELECT time FROM attendance WHERE user_id = %s AND date = %s ORDER BY time DESC LIMIT 1",
                (user_id, date_db)
            )
            last_entry = cursor.fetchone()
            if last_entry:
                last_time_val = last_entry[0] # This might be timedelta or string
                if isinstance(last_time_val, timedelta):
                    last_time_str = (datetime.min + last_time_val).time().strftime("%H:%M:%S")
                else: # Assuming it's already a string
                    last_time_str = str(last_time_val)
                
                try:
                    last_time_obj = datetime.strptime(last_time_str, "%H:%M:%S").time()
                    current_time_obj_for_compare = datetime.strptime(time_db_24hr, "%H:%M:%S").time()
                    dummy_date = date(2000,1,1)
                    last_dt = datetime.combine(dummy_date, last_time_obj)
                    current_dt = datetime.combine(dummy_date, current_time_obj_for_compare)
                    if 0 <= (current_dt - last_dt).total_seconds() < 120:
                        return False, "COOLDOWN"
                except ValueError as ve: print(f"Time parsing error in cooldown: {ve}")

            cursor.execute("INSERT INTO attendance (user_id, name, date, time) VALUES (%s, %s, %s, %s)", 
                           (user_id, name, date_db, time_db_24hr))
            conn.commit()
            
            csv_path = f'Attendance/Attendance-{datetoday}.csv'
            if not os.path.exists(csv_path):
                with open(csv_path, 'w') as f: f.write('Name,Roll,Time\n')
            with open(csv_path, 'a') as f: f.write(f'{name},{user_id},{time_db_24hr}\n')
            
            return True, format_time_to_12hr(time_db_24hr)
    try: return execute_with_retry(_add)
    except Exception as e: print(f"Error adding attendance: {e}"); traceback.print_exc(); return False, None

def extract_face(img_pil, box, image_size=160, margin=20):
    try:
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        size_bb = max(x2 - x1, y2 - y1)
        size = int(size_bb * 1.0 + margin * 2)
        x1_new = max(int(center_x - size // 2), 0); y1_new = max(int(center_y - size // 2), 0)
        x2_new = min(int(center_x + size // 2), img_pil.width); y2_new = min(int(center_y + size // 2), img_pil.height)
        face = img_pil.crop((x1_new, y1_new, x2_new, y2_new)).resize((image_size, image_size), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.array(face, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    except: return None

def process_face_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        boxes, _ = detector.detect(img)
        if boxes is None or (isinstance(boxes, np.ndarray) and boxes.shape[0] == 0): return None
        return extract_face(img, [int(c) for c in boxes[0]])
    except: return None

def identify_face(face_tensor, threshold=0.6):
    try:
        all_embs = get_all_embeddings()
        if not all_embs: return "Unknown_0"
        current_emb = resnet(face_tensor).detach()
        best_match, best_score = "Unknown_0", 0.0
        for name_id, ref_emb in all_embs.items():
            sim = torch.nn.functional.cosine_similarity(current_emb, ref_emb).item()
            if sim > best_score: best_score, best_match = sim, name_id
        return best_match if best_score >= threshold else "Unknown_0"
    except: return "Unknown_0"

@app.route('/')
def home(): return redirect(url_for('attendance_page'))

@app.route('/attendance')
def attendance_page():
    selected_date = request.args.get('date')
    names, rolls, times, l = extract_attendance(selected_date)
    display_date = datetoday2_default
    picker_date = datetime.now(ist).strftime("%Y-%m-%d")
    if selected_date:
        try:
            dt_obj = datetime.strptime(selected_date, "%Y-%m-%d")
            display_date = dt_obj.strftime("%d-%B-%Y")
            picker_date = selected_date
        except ValueError: pass 
    today_for_check_yyyymmdd = datetime.now(ist).strftime("%Y-%m-%d")
    return render_template('attendance.html', names=names, rolls=rolls, times=times, l=l,
                           datetoday2_display=display_date, selected_date_for_picker=picker_date,
                           today_date_for_check=today_for_check_yyyymmdd, 
                           mess=request.args.get('mess'))

@app.route('/user-management')
def user_management_page():
    return render_template('user_management.html', totalreg=get_total_users(), datetoday2=datetoday2_default, mess=request.args.get('mess'))

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_route():
    try:
        data = request.json; img_data_uri = data.get('image')
        if not img_data_uri: return jsonify({'success': False, 'message': 'No image data'})
        try:
            header, encoded = img_data_uri.split(",", 1)
            img = Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB')
        except Exception as e: return jsonify({'success': False, 'message': f'Error decoding image: {e}'})
        
        boxes, _ = detector.detect(img)
        if boxes is None or (isinstance(boxes, np.ndarray) and boxes.shape[0] == 0):
            return jsonify({'success': False, 'message': 'No face detected'})
            
        face_tensor = extract_face(img, [int(c) for c in boxes[0]])
        if face_tensor is None: return jsonify({'success': False, 'message': 'Could not extract face'})

        identity = identify_face(face_tensor)
        name, user_id_str = identity.split('_')[0], identity.split('_')[1]
        if name == "Unknown": return jsonify({'success': False, 'message': 'Unknown person'})
        
        logged, status_or_time = add_attendance(name, user_id_str) 
        if logged:
            return jsonify({'success': True, 'message': f'Logged for {name} at {status_or_time}', 'name': name, 'user_id': user_id_str, 'time': status_or_time})
        elif status_or_time == "COOLDOWN":
            return jsonify({'success': False, 'message': 'COOLDOWN'})
        return jsonify({'success': False, 'message': f'Failed to log for {name}'})
    except Exception as e: print(f"Mark attendance error: {e}"); traceback.print_exc(); return jsonify({'success': False, 'message': str(e)})

@app.route('/get_attendance')
def get_attendance_route():
    names, rolls, times, l = extract_attendance()
    return jsonify([{'name': n, 'roll': r, 'time': t} for n,r,t in zip(names,rolls,times)])

@app.route('/get_all_users')
def get_users_route(): return jsonify({'success': True, 'users': get_all_users_from_db()})

@app.route('/register_user', methods=['POST'])
def register_user_route():
    try:
        data = request.json; u, uid = data.get('username'), data.get('userid')
        if not (u and uid): return jsonify({'success': False, 'message': 'Name/ID required'})
        if not add_user_to_db(u, uid): return jsonify({'success': False, 'message': 'User ID exists'})
        os.makedirs(f'static/faces/{u}_{uid}', exist_ok=True)
        os.makedirs(f'static/temp_embeddings/{u}_{uid}', exist_ok=True)
        return jsonify({'success': True, 'message': 'User registered'})
    except: return jsonify({'success': False, 'message': 'Registration error'})

@app.route('/delete_user', methods=['POST'])
def delete_user_route():
    try:
        uid = request.json.get('user_id')
        if not uid: return jsonify({'success': False, 'message': 'User ID required'})
        if delete_user_from_db(uid): return jsonify({'success': True, 'message': 'User deleted'})
        return jsonify({'success': False, 'message': 'Delete failed'})
    except: return jsonify({'success': False, 'message': 'Delete error'})

@app.route('/capture_image', methods=['POST'])
def capture_image_route():
    try:
        if 'face_image' not in request.files: return jsonify({'success': False, 'message': 'No image'})
        u, uid, count = request.form.get('username'), request.form.get('userid'), int(request.form.get('image_count','0'))
        if not (u and uid): return jsonify({'success': False, 'message': 'Name/ID required'})
        
        img_file = request.files['face_image']
        img_path = os.path.join(f'static/faces/{u}_{uid}', f'{u}_{count}.jpg')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img_file.save(img_path)
        
        tensor = process_face_image(img_path)
        if tensor is None:
            if os.path.exists(img_path): os.remove(img_path)
            return jsonify({'success': False, 'message': 'No face detected'})
        
        emb_path = os.path.join(f'static/temp_embeddings/{u}_{uid}', f'embedding_{count}.pt')
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        torch.save(resnet(tensor).detach(), emb_path)
        return jsonify({'success': True, 'message': 'Image saved'})
    except Exception as e: print(f"Capture error: {e}"); traceback.print_exc(); return jsonify({'success': False, 'message': str(e)})
    
@app.route('/export_attendance')
def export_attendance_route():
    try:
        date_str = request.args.get('date')
        fname_date = date_str or datetime.now(ist).strftime("%Y-%m-%d")
        names, rolls, times, l = extract_attendance(date_str)
        if l == 0:
            return redirect(url_for('attendance_page', mess=f"No records for {datetime.strptime(fname_date, '%Y-%m-%d').strftime('%d-%B-%Y')}.", date=date_str or ''))
        
        out = BytesIO()
        out.write('Name,ID,Time (IST)\n'.encode())
        for i in range(l): out.write(f'{names[i]},{rolls[i]},{times[i]}\n'.encode())
        out.seek(0)
        return send_file(out, mimetype='text/csv', as_attachment=True, download_name=f"attendance_{fname_date}.csv")
    except: return jsonify({'success': False, 'message': 'Export error'})
    
@app.route('/get_attendance_history', methods=['POST'])
def get_attendance_history_route():
    try:
        uid = request.json.get('user_id')
        if not uid: return jsonify({'success': False, 'message': 'User ID required'})
        with get_db_connection() as (conn, cursor):
            cursor.execute("SELECT date, time FROM attendance WHERE user_id = %s ORDER BY date DESC, time DESC LIMIT 30", (uid,))
            hist = [{'date': r[0].strftime('%Y-%m-%d') if isinstance(r[0], date) else str(r[0]), 
                     'time': format_time_to_12hr(r[1])} for r in cursor.fetchall()]
            return jsonify({'success': True, 'history': hist})
    except Exception as e: print(f"History error: {e}"); traceback.print_exc(); return jsonify({'success': False, 'message': 'History error'})

@app.route('/complete_registration', methods=['POST'])
def complete_registration_route():
    try:
        u, uid = request.json.get('username'), request.json.get('userid')
        if not (u and uid): return jsonify({'success': False, 'message': 'Name/ID required'})
        
        temp_dir = f'static/temp_embeddings/{u}_{uid}'
        if not os.path.exists(temp_dir): return jsonify({'success': False, 'message': 'No embeddings'})
        
        files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
        if not files: return jsonify({'success': False, 'message': 'No valid embeddings'})
            
        embs = [torch.load(os.path.join(temp_dir, f)).to(device) for f in files]
        if not embs: return jsonify({'success': False, 'message': 'Load embeddings failed'})

        avg_emb = torch.mean(torch.cat(embs, dim=0), dim=0, keepdim=True)
        if save_embedding(uid, avg_emb):
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'success': True, 'message': 'Registration completed'})
        return jsonify({'success': False, 'message': 'Save embeddings failed'})
    except: return jsonify({'success': False, 'message': 'Completion error'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('/home/ayyappa/Documents/FaceRec-attendance/ssl/cert.pem', '/home/ayyappa/Documents/FaceRec-attendance/ssl/key.pem'))