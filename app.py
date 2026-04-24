from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sys, os, base64, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'face_recognition_project'))

from utils.db_helper import insert_person, get_all_persons, person_count, attendance_today_count, get_attendance_log, log_attendance
from utils.setup import ensure_directories, initialize_database
from utils.trainer import train_model, load_encodings
import config

app = Flask(__name__)
app.secret_key = 'face_recognition_secret_key'

ensure_directories()
initialize_database()

# ── Login ──────────────────────────────────────────────
@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            error = 'Wrong username or password!'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── Dashboard ──────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    stats = {
        'persons': person_count(),
        'today': attendance_today_count(),
        'recent': get_attendance_log(limit=8)
    }
    return render_template('dashboard.html', stats=stats)

# ── Persons ────────────────────────────────────────────
@app.route('/persons')
def persons():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    all_persons = get_all_persons()
    return render_template('persons.html', persons=all_persons)

# ── Attendance ─────────────────────────────────────────
@app.route('/attendance')
def attendance():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    logs = get_attendance_log(limit=100)
    return render_template('attendance.html', logs=logs)

# ── Register ───────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    success = None
    error = None
    if request.method == 'POST':
        try:
            name        = request.form.get('name', '').strip()
            father_name = request.form.get('father_name', '').strip()
            cnic        = request.form.get('cnic', '').strip()
            department  = request.form.get('department', '').strip()
            semester    = request.form.get('semester', '').strip()
            phone       = request.form.get('phone', '').strip()
            address     = request.form.get('address', '').strip()
            photos_json = request.form.get('photos', '[]')
            photos      = json.loads(photos_json)

            person_id = insert_person(
                name=name, father_name=father_name, cnic=cnic,
                department=department, semester=semester,
                phone=phone, address=address, image_dir=""
            )

            # Save photos
            safe_name = name.replace(" ", "_")
            folder = os.path.join(config.DATASET_DIR, f"{person_id}_{safe_name}")
            os.makedirs(folder, exist_ok=True)

            for i, photo_data in enumerate(photos):
                img_data = base64.b64decode(photo_data.split(',')[1])
                with open(os.path.join(folder, f"img_{i+1:03d}.jpg"), 'wb') as f:
                    f.write(img_data)

            success = f"{name} registered successfully with {len(photos)} photos!"
        except Exception as e:
            error = str(e)

    return render_template('register.html', success=success, error=error)

# ── Train ──────────────────────────────────────────────
@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    result = None
    if request.method == 'POST':
        success, message = train_model()
        result = {'success': success, 'message': message}
    return render_template('train.html', result=result)

# ── Camera ─────────────────────────────────────────────
@app.route('/camera')
def camera():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('camera.html')

# ── Recognize API ──────────────────────────────────────
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'logged_in' not in session:
        return jsonify({'error': 'Not logged in'})
    try:
        import face_recognition
        import cv2

        data = request.json
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        encodings_data = load_encodings()
        if not encodings_data:
            return jsonify({'name': 'Unknown Person', 'confidence': 0})

        face_locs = face_recognition.face_locations(rgb)
        face_encs = face_recognition.face_encodings(rgb, face_locs)

        for face_enc in face_encs:
            distances = face_recognition.face_distance(encodings_data['encodings'], face_enc)
            if len(distances) == 0:
                continue
            best_idx = int(np.argmin(distances))
            best_dist = distances[best_idx]
            confidence = round((1 - best_dist) * 100, 1)

            if best_dist < config.RECOGNITION_TOLERANCE:
                person_id = encodings_data['ids'][best_idx]
                name = encodings_data['names'][best_idx]

                all_persons = get_all_persons()
                person_info = next((p for p in all_persons if p['id'] == person_id), {})

                log_attendance(person_id=person_id, confidence=confidence)

                return jsonify({
                    'name': name,
                    'department': person_info.get('department', ''),
                    'semester': person_info.get('semester', ''),
                    'confidence': confidence
                })

        return jsonify({'name': 'Unknown Person', 'confidence': 0})
    except Exception as e:
        return jsonify({'error': str(e), 'name': 'Unknown Person', 'confidence': 0})

if __name__ == '__main__':
    app.run(debug=True, port=5000)