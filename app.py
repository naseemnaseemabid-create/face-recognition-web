# app.py - Complete Face Recognition Attendance System
# Fresh Build - All features working

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3
import os
import hashlib
from datetime import datetime, date, timedelta
import json
import base64
import cv2
import numpy as np
import face_recognition
from io import BytesIO
import pandas as pd
import pickle

# =============================================
# APP INITIALIZATION
# =============================================

app = Flask(__name__)
app.secret_key = 'face-recognition-secret-key-2024'
app.config['SESSION_TYPE'] = 'filesystem'

# Setup directories
os.makedirs('database', exist_ok=True)
os.makedirs('dataset', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('exports', exist_ok=True)

DB_PATH = 'database/attendance.db'

# =============================================
# DATABASE SETUP
# =============================================

def init_db():
    """Initialize database with all tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Teachers Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id VARCHAR(20) UNIQUE NOT NULL,
            username VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100),
            password_hash VARCHAR(255) NOT NULL,
            phone VARCHAR(15),
            department VARCHAR(100),
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Students Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id VARCHAR(20) UNIQUE NOT NULL,
            roll_number VARCHAR(20) NOT NULL,
            name VARCHAR(100) NOT NULL,
            class_name VARCHAR(50) NOT NULL,
            section VARCHAR(10),
            email VARCHAR(100),
            phone VARCHAR(15),
            father_name VARCHAR(100),
            registration_date DATE DEFAULT CURRENT_DATE,
            is_active BOOLEAN DEFAULT 1,
            face_encoding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(class_name, roll_number)
        )
    ''')
    
    # Subjects Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_code VARCHAR(20) UNIQUE NOT NULL,
            subject_name VARCHAR(100) NOT NULL,
            teacher_id INTEGER NOT NULL,
            class_name VARCHAR(50) NOT NULL,
            semester VARCHAR(20),
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES teachers(id)
        )
    ''')
    
    # Attendance Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            teacher_id INTEGER NOT NULL,
            attendance_date DATE NOT NULL,
            status VARCHAR(20) DEFAULT 'present',
            marked_time TIME DEFAULT CURRENT_TIME,
            recognition_confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (subject_id) REFERENCES subjects(id),
            FOREIGN KEY (teacher_id) REFERENCES teachers(id),
            UNIQUE(student_id, subject_id, attendance_date)
        )
    ''')
    
    # Insert default admin teacher
    admin_pass = hashlib.sha256('admin123'.encode()).hexdigest()
    cursor.execute('''
        INSERT OR IGNORE INTO teachers (teacher_id, username, name, email, password_hash, department)
        VALUES ('TCH001', 'admin', 'Admin Teacher', 'admin@attendance.com', ?, 'Administration')
    ''', (admin_pass,))
    
    # Get admin ID
    cursor.execute('SELECT id FROM teachers WHERE username = "admin"')
    admin_result = cursor.fetchone()
    
    if admin_result:
        admin_id = admin_result[0]
        
        # Insert default subjects
        default_subjects = [
            ('MATH101', 'Mathematics', admin_id, '10th', 'Fall 2024'),
            ('PHY101', 'Physics', admin_id, '10th', 'Fall 2024'),
            ('CS101', 'Computer Science', admin_id, 'BCS 1st', 'Fall 2024'),
            ('ENG101', 'English', admin_id, '10th', 'Fall 2024'),
            ('URD101', 'Urdu', admin_id, '10th', 'Fall 2024'),
        ]
        
        for sub in default_subjects:
            cursor.execute('''
                INSERT OR IGNORE INTO subjects (subject_code, subject_name, teacher_id, class_name, semester)
                VALUES (?, ?, ?, ?, ?)
            ''', sub)
    
    conn.commit()
    conn.close()
    print("[Database] Initialized successfully!")

# =============================================
# FLASK-LOGIN SETUP
# =============================================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, teacher_id, username, name, department):
        self.id = id
        self.teacher_id = teacher_id
        self.username = username
        self.name = name
        self.department = department

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, teacher_id, username, name, department FROM teachers WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1], user[2], user[3], user[4])
    return None

# =============================================
# FACE RECOGNITION SETUP
# =============================================

face_encodings = []
face_names = []
face_ids = []

def load_face_encodings():
    """Load trained face encodings"""
    global face_encodings, face_names, face_ids
    model_path = 'models/face_encodings.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                face_encodings = data.get('encodings', [])
                face_names = data.get('names', [])
                face_ids = data.get('ids', [])
            print(f"[FaceRec] Loaded {len(face_encodings)} faces")
            return True
        except Exception as e:
            print(f"[FaceRec] Error: {e}")
    return False

# =============================================
# ROUTES - AUTHENTICATION
# =============================================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, teacher_id, username, name, department 
            FROM teachers 
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            user_obj = User(user[0], user[1], user[2], user[3], user[4])
            login_user(user_obj)
            flash(f'Welcome back, {user[3]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get statistics
    cursor.execute('SELECT COUNT(*) FROM students WHERE is_active = 1')
    total_students = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM subjects WHERE teacher_id = ?', (current_user.id,))
    my_subjects = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT COUNT(DISTINCT student_id) FROM attendance 
        WHERE teacher_id = ? AND attendance_date = ?
    ''', (current_user.id, date.today()))
    today_attendance = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT COUNT(*) FROM attendance 
        WHERE teacher_id = ? AND attendance_date = ?
    ''', (current_user.id, date.today()))
    marked_today = cursor.fetchone()[0]
    
    # Get recent attendance
    cursor.execute('''
        SELECT a.attendance_date, COUNT(DISTINCT a.student_id) as count, s.subject_name
        FROM attendance a
        JOIN subjects s ON a.subject_id = s.id
        WHERE a.teacher_id = ?
        GROUP BY a.attendance_date, a.subject_id
        ORDER BY a.attendance_date DESC
        LIMIT 5
    ''', (current_user.id,))
    recent = cursor.fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', 
                         user=current_user,
                         total_students=total_students,
                         my_subjects=my_subjects,
                         today_attendance=today_attendance,
                         marked_today=marked_today,
                         recent_attendance=recent)

# =============================================
# ROUTES - TEACHER MANAGEMENT
# =============================================

@app.route('/teachers')
@login_required
def manage_teachers():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, teacher_id, username, name, email, department, phone, is_active FROM teachers')
    teachers = cursor.fetchall()
    conn.close()
    return render_template('teachers.html', teachers=teachers)

@app.route('/teacher/register', methods=['GET', 'POST'])
@login_required
def register_teacher():
    if request.method == 'POST':
        username = request.form['username']
        name = request.form['name']
        password = request.form['password']
        department = request.form.get('department', '')
        email = request.form.get('email', '')
        phone = request.form.get('phone', '')
        
        teacher_id = f"TCH{datetime.now().strftime('%Y%m%d%H%M%S')}"
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO teachers (teacher_id, username, name, email, password_hash, department, phone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (teacher_id, username, name, email, password_hash, department, phone))
            conn.commit()
            flash(f'Teacher {name} registered successfully!', 'success')
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'danger')
        finally:
            conn.close()
        
        return redirect(url_for('manage_teachers'))
    
    return render_template('register_teacher.html')

# =============================================
# ROUTES - STUDENT MANAGEMENT
# =============================================

@app.route('/students')
@login_required
def manage_students():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, student_id, roll_number, name, class_name, section, email, phone, is_active 
        FROM students WHERE is_active = 1
        ORDER BY class_name, roll_number
    ''')
    students = cursor.fetchall()
    conn.close()
    return render_template('students.html', students=students)

@app.route('/student/register', methods=['GET', 'POST'])
@login_required
def register_student():
    if request.method == 'POST':
        name = request.form['name']
        roll_number = request.form['roll_number']
        class_name = request.form['class_name']
        section = request.form.get('section', '')
        email = request.form.get('email', '')
        phone = request.form.get('phone', '')
        father_name = request.form.get('father_name', '')
        
        student_id = f"STU{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO students (student_id, roll_number, name, class_name, section, email, phone, father_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (student_id, roll_number, name, class_name, section, email, phone, father_name))
            conn.commit()
            
            student_db_id = cursor.lastrowid
            os.makedirs(f'dataset/{student_db_id}', exist_ok=True)
            
            flash(f'Student {name} registered successfully! Now capture face images.', 'success')
            return redirect(url_for('capture_faces', student_id=student_db_id))
        except sqlite3.IntegrityError:
            flash('Roll number already exists in this class!', 'danger')
        finally:
            conn.close()
    
    return render_template('register_student.html')

@app.route('/student/capture/<int:student_id>')
@login_required
def capture_faces(student_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, roll_number FROM students WHERE id = ?', (student_id,))
    student = cursor.fetchone()
    conn.close()
    
    if not student:
        flash('Student not found!', 'danger')
        return redirect(url_for('manage_students'))
    
    return render_template('capture_faces.html', 
                         student_id=student_id, 
                         student_name=student[0], 
                         roll_number=student[1])

@app.route('/api/capture-face', methods=['POST'])
@login_required
def api_capture_face():
    """API to capture face images from webcam"""
    try:
        data = request.json
        student_id = data.get('student_id')
        image_data = data.get('image')
        
        if not student_id or not image_data:
            return jsonify({'success': False, 'message': 'Missing data'})
        
        # Decode image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Detect face
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Save image
        student_dir = f'dataset/{student_id}'
        os.makedirs(student_dir, exist_ok=True)
        
        existing_files = len([f for f in os.listdir(student_dir) if f.endswith('.jpg')])
        
        if existing_files >= 20:
            return jsonify({'success': False, 'message': 'Maximum 20 images captured!'})
        
        image_path = f'{student_dir}/face_{existing_files + 1}.jpg'
        cv2.imwrite(image_path, frame)
        
        # Get face encoding
        face_encodings_detected = face_recognition.face_encodings(rgb_frame, face_locations)
        if face_encodings_detected:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('UPDATE students SET face_encoding = ? WHERE id = ?', 
                          (pickle.dumps(face_encodings_detected[0]), student_id))
            conn.commit()
            conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'Face captured! ({existing_files + 1}/20)',
            'count': existing_files + 1
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# =============================================
# ROUTES - SUBJECT MANAGEMENT
# =============================================

@app.route('/subjects')
@login_required
def manage_subjects():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.id, s.subject_code, s.subject_name, s.class_name, s.semester, 
               t.name as teacher_name
        FROM subjects s
        JOIN teachers t ON s.teacher_id = t.id
        WHERE s.is_active = 1
        ORDER BY s.class_name, s.subject_name
    ''')
    subjects = cursor.fetchall()
    conn.close()
    return render_template('subjects.html', subjects=subjects)

@app.route('/subject/add', methods=['GET', 'POST'])
@login_required
def add_subject():
    if request.method == 'POST':
        subject_code = request.form['subject_code']
        subject_name = request.form['subject_name']
        class_name = request.form['class_name']
        semester = request.form.get('semester', '')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO subjects (subject_code, subject_name, teacher_id, class_name, semester)
                VALUES (?, ?, ?, ?, ?)
            ''', (subject_code, subject_name, current_user.id, class_name, semester))
            conn.commit()
            flash(f'Subject {subject_name} added successfully!', 'success')
        except sqlite3.IntegrityError:
            flash('Subject code already exists!', 'danger')
        finally:
            conn.close()
        
        return redirect(url_for('manage_subjects'))
    
    return render_template('add_subject.html')

# =============================================
# ROUTES - ATTENDANCE SYSTEM
# =============================================

@app.route('/attendance/mark')
@login_required
def mark_attendance():
    """Page for marking attendance"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, subject_code, subject_name, class_name 
        FROM subjects 
        WHERE teacher_id = ? AND is_active = 1
    ''', (current_user.id,))
    subjects = cursor.fetchall()
    
    conn.close()
    return render_template('attendance_mark.html', subjects=subjects)

@app.route('/attendance/live')
@login_required
def live_attendance():
    """Live face recognition attendance"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, subject_code, subject_name, class_name 
        FROM subjects 
        WHERE teacher_id = ? AND is_active = 1
    ''', (current_user.id,))
    subjects = cursor.fetchall()
    
    conn.close()
    return render_template('live_attendance.html', subjects=subjects, user=current_user)

@app.route('/api/recognize-attendance', methods=['POST'])
@login_required
def api_recognize_attendance():
    """API endpoint for face recognition attendance marking"""
    try:
        data = request.json
        image_data = data.get('image')
        subject_id = data.get('subject_id')
        
        if not image_data or not subject_id:
            return jsonify({'success': False, 'message': 'Missing data'})
        
        # Load face encodings if not loaded
        if not face_encodings:
            load_face_encodings()
            if not face_encodings:
                return jsonify({'success': False, 'message': 'Model not trained. Please train first.'})
        
        # Decode image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Get face encodings
        face_encodings_detected = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings_detected:
            return jsonify({'success': False, 'message': 'Could not encode face'})
        
        # Compare with known faces
        for face_encoding in face_encodings_detected:
            distances = face_recognition.face_distance(face_encodings, face_encoding)
            best_match_idx = np.argmin(distances)
            confidence = (1 - distances[best_match_idx]) * 100
            
            if confidence > 40:  # Threshold
                student_id = face_ids[best_match_idx]
                student_name = face_names[best_match_idx]
                
                # Check if already marked today
                today = date.today()
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM attendance 
                    WHERE student_id = ? AND subject_id = ? AND attendance_date = ?
                ''', (student_id, subject_id, today))
                
                if cursor.fetchone():
                    conn.close()
                    return jsonify({
                        'success': False,
                        'message': f'{student_name} already marked today'
                    })
                
                # Mark attendance
                cursor.execute('''
                    INSERT INTO attendance (student_id, subject_id, teacher_id, attendance_date, recognition_confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (student_id, subject_id, current_user.id, today, confidence))
                
                conn.commit()
                
                # Get student details
                cursor.execute('SELECT roll_number, class_name FROM students WHERE id = ?', (student_id,))
                student_info = cursor.fetchone()
                conn.close()
                
                return jsonify({
                    'success': True,
                    'student_name': student_name,
                    'roll_number': student_info[0] if student_info else '',
                    'class': student_info[1] if student_info else '',
                    'confidence': confidence,
                    'message': f'✓ {student_name} - Attendance marked!'
                })
        
        return jsonify({'success': False, 'message': 'Face not recognized'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# =============================================
# ROUTES - ATTENDANCE REPORTS
# =============================================

@app.route('/attendance/report')
@login_required
def attendance_report():
    """View attendance reports with filters"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get subjects for filter
    cursor.execute('SELECT id, subject_name FROM subjects WHERE teacher_id = ?', (current_user.id,))
    subjects = cursor.fetchall()
    
    # Get classes for filter
    cursor.execute('SELECT DISTINCT class_name FROM students WHERE is_active = 1')
    classes = cursor.fetchall()
    
    conn.close()
    return render_template('attendance_report.html', subjects=subjects, classes=classes)

@app.route('/api/attendance-data')
@login_required
def api_attendance_data():
    """API to get attendance data for reports"""
    subject_id = request.args.get('subject_id')
    class_name = request.args.get('class_name')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = '''
        SELECT 
            s.name as student_name,
            s.roll_number,
            s.class_name,
            sub.subject_name,
            a.attendance_date,
            a.status,
            a.marked_time,
            a.recognition_confidence
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        JOIN subjects sub ON a.subject_id = sub.id
        WHERE 1=1
    '''
    params = []
    
    if subject_id and subject_id != 'all':
        query += " AND a.subject_id = ?"
        params.append(subject_id)
    
    if class_name and class_name != 'all':
        query += " AND s.class_name = ?"
        params.append(class_name)
    
    if start_date:
        query += " AND a.attendance_date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND a.attendance_date <= ?"
        params.append(end_date)
    
    query += " ORDER BY a.attendance_date DESC, s.class_name, s.roll_number"
    
    cursor.execute(query, params)
    records = cursor.fetchall()
    conn.close()
    
    return jsonify({
        'records': records,
        'count': len(records)
    })

@app.route('/attendance/export')
@login_required
def export_attendance():
    """Export attendance to Excel"""
    subject_id = request.args.get('subject_id')
    class_name = request.args.get('class_name')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    conn = sqlite3.connect(DB_PATH)
    
    query = '''
        SELECT 
            s.roll_number,
            s.name as student_name,
            s.class_name,
            sub.subject_name,
            a.attendance_date,
            a.status,
            a.marked_time,
            a.recognition_confidence
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        JOIN subjects sub ON a.subject_id = sub.id
        WHERE 1=1
    '''
    params = []
    
    if subject_id and subject_id != 'all':
        query += " AND a.subject_id = ?"
        params.append(subject_id)
    
    if class_name and class_name != 'all':
        query += " AND s.class_name = ?"
        params.append(class_name)
    
    if start_date:
        query += " AND a.attendance_date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND a.attendance_date <= ?"
        params.append(end_date)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        flash('No data to export!', 'warning')
        return redirect(url_for('attendance_report'))
    
    # Create Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Attendance Report', index=False)
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'attendance_report_{date.today()}.xlsx'
    )

# =============================================
# ROUTES - TRAIN MODEL
# =============================================

@app.route('/train', methods=['GET', 'POST'])
@login_required
def train_model():
    """Train the face recognition model"""
    if request.method == 'POST':
        from pathlib import Path
        
        dataset_path = Path('dataset')
        known_encodings = []
        known_names = []
        known_ids = []
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for person_dir in dataset_path.iterdir():
            if person_dir.is_dir():
                try:
                    person_id = int(person_dir.name)
                    cursor.execute('SELECT name FROM students WHERE id = ?', (person_id,))
                    student = cursor.fetchone()
                    if not student:
                        continue
                    
                    student_name = student[0]
                    encodings = []
                    
                    for image_path in person_dir.glob('*.jpg'):
                        try:
                            image = face_recognition.load_image_file(str(image_path))
                            face_locations = face_recognition.face_locations(image)
                            
                            if face_locations:
                                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                                encodings.append(face_encoding)
                        except Exception as e:
                            print(f"Error: {e}")
                    
                    if encodings:
                        avg_encoding = np.mean(encodings, axis=0)
                        known_encodings.append(avg_encoding)
                        known_names.append(student_name)
                        known_ids.append(person_id)
                        print(f"Trained: {student_name} - {len(encodings)} images")
                except ValueError:
                    continue
        
        conn.close()
        
        if known_encodings:
            model_data = {
                'encodings': known_encodings,
                'names': known_names,
                'ids': known_ids
            }
            with open('models/face_encodings.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            global face_encodings, face_names, face_ids
            face_encodings = known_encodings
            face_names = known_names
            face_ids = known_ids
            
            flash(f'🎯 Model trained successfully with {len(known_encodings)} faces!', 'success')
        else:
            flash('No faces found to train! Please capture face images first.', 'danger')
        
        return redirect(url_for('dashboard'))
    
    return render_template('train.html')

# =============================================
# ROUTES - PERSONS (Legacy Support)
# =============================================

@app.route('/persons')
@login_required
def view_persons():
    """View all registered persons"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, student_id, name, roll_number, class_name, email, phone FROM students WHERE is_active = 1')
    persons = cursor.fetchall()
    conn.close()
    return render_template('persons.html', persons=persons)

# =============================================
# MAIN
# =============================================

if __name__ == '__main__':
    init_db()
    load_face_encodings()
    print("\n" + "="*50)
    print("🎯 Face Recognition Attendance System")
    print("="*50)
    print("📍 Server Running at: http://localhost:5000")
    print("🔐 Username: admin")
    print("🔐 Password: admin123")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, host='0.0.0.0')