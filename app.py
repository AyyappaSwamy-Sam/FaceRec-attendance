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

if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect('face_attendance.db', timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON;") # ENABLE FOREIGN KEY ENFORCEMENT
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
    if not time_str_24hr: return ""
    try:
        return datetime.strptime(time_str_24hr, "%H:%M:%S").strftime("%I:%M:%S %p")
    except ValueError: return time_str_24hr

def extract_attendance(target_date_str=None):
    def _extract(date_to_query_db):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, user_id, time FROM attendance WHERE date = ? ORDER BY time", (date_to_query_db,))
            res = cursor.fetchall()
            return [r[0] for r in res], [r[1] for r in res], [format_time_to_12hr(r[2]) for r in res], len(res)

    date_to_query = target_date_str or datetime.now(ist).strftime("%Y-%m-%d")
    if target_date_str:
        try: datetime.strptime(target_date_str, "%Y-%m-%d")
        except ValueError: date_to_query = datetime.now(ist).strftime("%Y-%m-%d")
    
    try:
        names, rolls, times, l = execute_with_retry(lambda: _extract(date_to_query))
        print(f"Retrieved {l} records for {date_to_query} from DB.")
        return names, rolls, times, l
    except Exception as e:
        print(f"Error extracting for {date_to_query} from DB: {e}")
        if date_to_query == datetime.now(ist).strftime("%Y-%m-%d"):
            try:
                df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
                print(f"Retrieved {len(df)} records for today from CSV fallback.")
                return df['Name'].tolist(), df['Roll'].tolist(), [format_time_to_12hr(t) for t in df['Time'].tolist()], len(df)
            except Exception as csv_e: print(f"CSV fallback failed: {csv_e}")
        return [], [], [], 0

def get_total_users():
    def _get():
        with get_db_connection() as conn: return conn.cursor().execute("SELECT COUNT(*) FROM users").fetchone()[0]
    try: return execute_with_retry(_get)
    except: return 0

# --- MODIFIED get_all_users_from_db ---
def get_all_users_from_db():
    def _get_users():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, user_id, registration_date FROM users ORDER BY name")
            users_data = []
            for row in cursor.fetchall():
                name, user_id, reg_date = row[0], row[1], row[2]
                # Construct the path to the first registration image (USERNAME_0.jpg)
                # Ensure USERNAME part of path is safe (e.g. no slashes if username could contain them)
                # For simplicity, assuming username is simple.
                first_image_filename = f"{name}_0.jpg"
                first_image_path_relative = f"static/faces/{name}_{user_id}/{first_image_filename}"
                
                # Check if the image actually exists
                if os.path.exists(first_image_path_relative):
                    # Pass the URL path for the template
                    image_url = url_for('static', filename=f'faces/{name}_{user_id}/{first_image_filename}')
                else:
                    image_url = None # Or a placeholder image URL

                users_data.append({
                    'name': name,
                    'user_id': user_id,
                    'registration_date': reg_date,
                    'image_url': image_url 
                })
            return users_data
    try: 
        return execute_with_retry(_get_users)
    except Exception as e: 
        print(f"Error getting users: {e}")
        traceback.print_exc()
        return []
# --- END OF MODIFIED get_all_users_from_db ---

def add_user_to_db(name, user_id):
    def _add():
        with get_db_connection() as conn:
            cur = conn.cursor()
            if cur.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone(): return False
            cur.execute("INSERT INTO users (name, user_id, registration_date) VALUES (?, ?, ?)", 
                        (name, user_id, datetime.now(ist).strftime("%Y-%m-%d")))
            return True
    try: return execute_with_retry(_add)
    except: return False

def delete_user_from_db(user_id):
    def _delete_with_manual_cascade_fallback():
        with get_db_connection() as conn: # PRAGMA foreign_keys = ON is set here
            cursor = conn.cursor()
            
            # Get user name first (for directory deletion)
            cursor.execute("SELECT name FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            
            if not result:
                print(f"User ID {user_id} not found for deletion.")
                return False, None
            user_name = result[0]

            try:
                # Attempt to delete the user. If PRAGMA foreign_keys=ON is working,
                # ON DELETE CASCADE should handle related records.
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    print(f"Successfully deleted user {user_name} (ID: {user_id}) and cascaded related records.")
                    return True, user_name
                else:
                    # This case should ideally not be hit if user existed,
                    # but as a safeguard if PRAGMA isn't effective.
                    print(f"User {user_name} (ID: {user_id}) found but not deleted by direct query. Attempting manual cascade.")
            
            except sqlite3.IntegrityError as e:
                # This block will be hit if PRAGMA foreign_keys=ON is NOT effectively enabling CASCADE
                # or if there's some other constraint issue not covered by CASCADE (less likely here).
                print(f"FOREIGN KEY constraint failed for user {user_id} (Error: {e}). Attempting manual deletion of dependent records.")
                conn.rollback() # Rollback the failed delete attempt on users table

                # Manually delete dependent records
                print(f"Manually deleting attendance records for user_id: {user_id}")
                cursor.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
                print(f"Manually deleting embeddings for user_id: {user_id}")
                cursor.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
                
                # Now try deleting the user again
                print(f"Retrying deletion of user_id: {user_id} from users table.")
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    print(f"Successfully deleted user {user_name} (ID: {user_id}) after manual cascade.")
                    return True, user_name
                else:
                    print(f"Failed to delete user {user_name} (ID: {user_id}) even after manual cascade.")
                    return False, user_name # Return user_name for potential directory cleanup

            # If we reach here, it means the user was found but not deleted by the first attempt,
            # and no IntegrityError was caught (which is unusual if PRAGMA isn't working).
            # This is a fallback for an unexpected state.
            print(f"User {user_name} (ID: {user_id}) was found but not deleted, and no IntegrityError was caught. This is unexpected.")
            return False, user_name


    try:
        # Use the new internal function name for execute_with_retry
        deleted, user_name_for_cleanup = execute_with_retry(_delete_with_manual_cascade_fallback)
        
        if deleted and user_name_for_cleanup:
            # Directory cleanup logic
            user_faces_dir = f'static/faces/{user_name_for_cleanup}_{user_id}'
            user_temp_embeddings_dir = f'static/temp_embeddings/{user_name_for_cleanup}_{user_id}'
            
            if os.path.exists(user_faces_dir):
                shutil.rmtree(user_faces_dir, ignore_errors=True)
                print(f"Cleaned up directory: {user_faces_dir}")
            if os.path.exists(user_temp_embeddings_dir):
                shutil.rmtree(user_temp_embeddings_dir, ignore_errors=True)
                print(f"Cleaned up directory: {user_temp_embeddings_dir}")
            
            # Optional: Update CSVs (DB is primary source of truth now)
            # This part can be kept or removed based on how critical CSVs are
            try:
                for attendance_file in [f for f in os.listdir('Attendance') if f.endswith('.csv')]:
                    file_path = os.path.join('Attendance', attendance_file)
                    df = pd.read_csv(file_path)
                    # Ensure comparison is robust, e.g., by converting both to string or appropriate type
                    if 'Roll' in df.columns and str(user_id) in df['Roll'].astype(str).values:
                        df = df[df['Roll'].astype(str) != str(user_id)]
                        df.to_csv(file_path, index=False)
                        print(f"Updated CSV: {file_path}")
            except Exception as e:
                print(f"Error updating CSVs during user deletion: {e}")
                
        return deleted
    except Exception as e:
        print(f"Outer error in delete_user_from_db for user_id {user_id}: {e}")
        traceback.print_exc()
        return False


def save_embedding(user_id, embedding):
    def _save():
        with get_db_connection() as conn:
            conn.cursor().execute("INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)", 
                                  (user_id, embedding.cpu().numpy().tobytes()))
            return True
    try: return execute_with_retry(_save)
    except: return False

def get_all_embeddings():
    def _get():
        with get_db_connection() as conn:
            emb_dict = {}
            for name, uid, b in conn.cursor().execute("SELECT u.name, u.user_id, e.embedding FROM users u JOIN embeddings e ON u.user_id = e.user_id").fetchall():
                try: emb_dict[f"{name}_{uid}"] = torch.from_numpy(np.frombuffer(b, dtype=np.float32).reshape(1, -1)).to(device)
                except: pass
            return emb_dict
    try: return execute_with_retry(_get)
    except: return {}

def add_attendance(name, user_id):
    def _add():
        with get_db_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(ist)
            date_db = now.strftime("%Y-%m-%d")
            time_db_24hr = now.strftime("%H:%M:%S")

            # --- COOLDOWN CHECK (2 minutes = 120 seconds) ---
            cursor.execute(
                "SELECT time FROM attendance WHERE user_id = ? AND date = ? ORDER BY time DESC LIMIT 1",
                (user_id, date_db)
            )
            last_entry = cursor.fetchone()
            if last_entry:
                last_time_str = last_entry[0]
                try:
                    last_time_obj = datetime.strptime(last_time_str, "%H:%M:%S").time()
                    current_time_obj_for_compare = datetime.strptime(time_db_24hr, "%H:%M:%S").time()
                    
                    dummy_date = date(2000, 1, 1) # Use a common dummy date for comparison
                    last_datetime_combined = datetime.combine(dummy_date, last_time_obj)
                    current_datetime_combined = datetime.combine(dummy_date, current_time_obj_for_compare)
                    
                    time_difference_seconds = (current_datetime_combined - last_datetime_combined).total_seconds()
                    
                    # Check if current time is after last time and within cooldown
                    if 0 <= time_difference_seconds < 120: 
                        print(f"Cooldown: User {name} (ID: {user_id}) logged {time_difference_seconds:.0f}s ago. Not logging again yet.")
                        return False, "COOLDOWN" 
                except ValueError as ve:
                    print(f"Error parsing time for cooldown check: {ve}")
            # --- END COOLDOWN CHECK ---
            
            cursor.execute("INSERT INTO attendance (user_id, name, date, time) VALUES (?, ?, ?, ?)", 
                           (user_id, name, date_db, time_db_24hr))
            
            csv_path = f'Attendance/Attendance-{datetoday}.csv'
            if not os.path.exists(csv_path):
                with open(csv_path, 'w') as f: f.write('Name,Roll,Time\n')
            with open(csv_path, 'a') as f: f.write(f'{name},{user_id},{time_db_24hr}\n')
            
            print(f"Logged attendance for {name} (ID: {user_id}) at {time_db_24hr} (IST).")
            return True, format_time_to_12hr(time_db_24hr)
    try: 
        return execute_with_retry(_add)
    except Exception as e: 
        print(f"Error adding attendance: {e}")
        traceback.print_exc()
        return False, None

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
        data = request.json
        img_data_uri = data.get('image')
        if not img_data_uri: 
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        try:
            header, encoded = img_data_uri.split(",", 1)
            img = Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB')
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error decoding image: {str(e)}'})
        
        boxes, _ = detector.detect(img)
        if boxes is None or (isinstance(boxes, np.ndarray) and boxes.shape[0] == 0):
            return jsonify({'success': False, 'message': 'No face detected in image'})
            
        box = boxes[0] 
        face_tensor = extract_face(img, [int(c) for c in box])
        if face_tensor is None:
             return jsonify({'success': False, 'message': 'Could not extract face features'})

        identity = identify_face(face_tensor)
        name, user_id_str = identity.split('_')[0], identity.split('_')[1]
        
        if name == "Unknown":
            return jsonify({'success': False, 'message': 'Unknown person detected'})
        
        logged, status_or_time = add_attendance(name, user_id_str) 
        
        if logged:
            return jsonify({'success': True, 'message': f'Attendance logged for {name} at {status_or_time}', 'name': name, 'user_id': user_id_str, 'time': status_or_time})
        elif status_or_time == "COOLDOWN":
            return jsonify({'success': False, 'message': 'COOLDOWN'})
        else:
            return jsonify({'success': False, 'message': f'Failed to log attendance for {name}'})
            
    except Exception as e:
        print(f"Error in /mark_attendance: {e}"); traceback.print_exc()
        return jsonify({'success': False, 'message': f'An unexpected error occurred: {str(e)}'})

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
        with get_db_connection() as conn:
            hist = [{'date': r[0], 'time': format_time_to_12hr(r[1])} for r in 
                    conn.cursor().execute("SELECT date, time FROM attendance WHERE user_id = ? ORDER BY date DESC, time DESC LIMIT 30", (uid,)).fetchall()]
            return jsonify({'success': True, 'history': hist})
    except: return jsonify({'success': False, 'message': 'History error'})

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