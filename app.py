# app.py - Complete Face Recognition Attendance System
# With Batches, Semesters, Promotions, Certificates, Student Portal, Parent Portal, Leave System

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, session
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
import qrcode
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from functools import wraps

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
# ACCESS CONTROL DECORATORS
# =============================================

def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('student_logged_in'):
            flash('Please login as student first!', 'danger')
            return redirect(url_for('student_login'))
        return f(*args, **kwargs)
    return decorated_function

def parent_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('parent_logged_in'):
            flash('Please login as parent first!', 'danger')
            return redirect(url_for('parent_login'))
        return f(*args, **kwargs)
    return decorated_function

# =============================================
# DATABASE SETUP
# =============================================

def init_db():
    """Initialize database with all tables including new features"""
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
    
    # Batches Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_name VARCHAR(50) NOT NULL,
            academic_year VARCHAR(20),
            start_date DATE,
            end_date DATE,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Semesters Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS semesters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER,
            semester_number INTEGER,
            semester_name VARCHAR(50),
            start_date DATE,
            end_date DATE,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (batch_id) REFERENCES batches(id)
        )
    ''')
    
    # Student Academic Records
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_academic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            batch_id INTEGER,
            current_semester INTEGER DEFAULT 1,
            enrollment_date DATE,
            graduation_date DATE,
            is_graduated BOOLEAN DEFAULT 0,
            cgpa REAL DEFAULT 0,
            total_credits INTEGER DEFAULT 0,
            earned_credits INTEGER DEFAULT 0,
            status VARCHAR(20) DEFAULT 'active',
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (batch_id) REFERENCES batches(id)
        )
    ''')
    
    # Leave Applications Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leave_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            teacher_id INTEGER,
            subject VARCHAR(200),
            reason TEXT,
            start_date DATE,
            end_date DATE,
            status VARCHAR(20) DEFAULT 'pending',
            applied_date DATE DEFAULT CURRENT_DATE,
            approved_date DATE,
            remarks TEXT,
            attachment_path VARCHAR(255),
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (teacher_id) REFERENCES teachers(id)
        )
    ''')
    
    # Attendance Certificates Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_certificates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            certificate_number VARCHAR(50) UNIQUE,
            semester_id INTEGER,
            attendance_percentage REAL,
            issue_date DATE DEFAULT CURRENT_DATE,
            certificate_path VARCHAR(255),
            is_downloaded BOOLEAN DEFAULT 0,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')
    
    # Parents Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id VARCHAR(20) UNIQUE,
            name VARCHAR(100),
            email VARCHAR(100) UNIQUE,
            phone VARCHAR(15),
            password_hash VARCHAR(255),
            cnic VARCHAR(13),
            address TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Student-Parent Relationship
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_parents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            parent_id INTEGER,
            relationship VARCHAR(20),
            is_primary BOOLEAN DEFAULT 0,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (parent_id) REFERENCES parents(id)
        )
    ''')
    
    # Student Login Accounts
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER UNIQUE,
            username VARCHAR(50) UNIQUE,
            password_hash VARCHAR(255),
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')
    
    # Promotions History
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            from_semester INTEGER,
            to_semester INTEGER,
            promotion_date DATE DEFAULT CURRENT_DATE,
            promoted_by INTEGER,
            remarks TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (promoted_by) REFERENCES teachers(id)
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
    print("[Database] Initialized successfully with all tables!")

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
    return render_template('index.html')

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
    
    cursor.execute('SELECT COUNT(*) FROM leave_applications WHERE status = "pending"')
    pending_leaves = cursor.fetchone()[0]
    
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
                         pending_leaves=pending_leaves,
                         recent_attendance=recent)

# =============================================
# ROUTES - TEACHER MANAGEMENT (Protected)
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
            
            student_username = f"student_{roll_number}"
            student_pass = hashlib.sha256(f"pass{roll_number}".encode()).hexdigest()
            cursor.execute('''
                INSERT INTO student_accounts (student_id, username, password_hash)
                VALUES (?, ?, ?)
            ''', (student_db_id, student_username, student_pass))
            
            cursor.execute('''
                INSERT INTO student_academic (student_id, current_semester, status)
                VALUES (?, 1, 'active')
            ''', (student_db_id,))
            
            conn.commit()
            flash(f'Student {name} registered successfully! Username: {student_username}, Password: pass{roll_number}', 'success')
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
    try:
        data = request.json
        student_id = data.get('student_id')
        image_data = data.get('image')
        
        if not student_id or not image_data:
            return jsonify({'success': False, 'message': 'Missing data'})
        
        image_bytes = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        student_dir = f'dataset/{student_id}'
        os.makedirs(student_dir, exist_ok=True)
        
        existing_files = len([f for f in os.listdir(student_dir) if f.endswith('.jpg')])
        
        if existing_files >= 20:
            return jsonify({'success': False, 'message': 'Maximum 20 images captured!'})
        
        image_path = f'{student_dir}/face_{existing_files + 1}.jpg'
        cv2.imwrite(image_path, frame)
        
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
# ROUTES - BATCH & SEMESTER MANAGEMENT
# =============================================

@app.route('/batches')
@login_required
def manage_batches():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM batches ORDER BY academic_year DESC')
    batches = cursor.fetchall()
    conn.close()
    return render_template('batches.html', batches=batches)

@app.route('/batch/create', methods=['GET', 'POST'])
@login_required
def create_batch():
    if request.method == 'POST':
        batch_name = request.form['batch_name']
        academic_year = request.form['academic_year']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO batches (batch_name, academic_year, start_date, end_date)
            VALUES (?, ?, ?, ?)
        ''', (batch_name, academic_year, start_date, end_date))
        conn.commit()
        batch_id = cursor.lastrowid
        
        for sem in range(1, 9):
            cursor.execute('''
                INSERT INTO semesters (batch_id, semester_number, semester_name, is_active)
                VALUES (?, ?, ?, 1)
            ''', (batch_id, sem, f'Semester {sem}'))
        
        conn.commit()
        conn.close()
        flash('Batch created successfully!', 'success')
        return redirect(url_for('manage_batches'))
    
    return render_template('create_batch.html')

@app.route('/semesters/<int:batch_id>')
@login_required
def view_semesters(batch_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM semesters WHERE batch_id = ? ORDER BY semester_number', (batch_id,))
    semesters = cursor.fetchall()
    conn.close()
    return render_template('semesters.html', semesters=semesters, batch_id=batch_id)

# =============================================
# ROUTES - PROMOTION SYSTEM
# =============================================

@app.route('/promotions')
@login_required
def manage_promotions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.id, s.name, s.roll_number, s.class_name,
               COALESCE(sa.current_semester, 1) as semester,
               COALESCE(sa.is_graduated, 0) as graduated
        FROM students s
        LEFT JOIN student_academic sa ON s.id = sa.student_id
        WHERE s.is_active = 1
    ''')
    students = cursor.fetchall()
    conn.close()
    
    return render_template('promotions.html', students=students)

@app.route('/api/promote-student', methods=['POST'])
@login_required
def promote_student():
    data = request.json
    student_id = data.get('student_id')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT current_semester, is_graduated FROM student_academic WHERE student_id = ?', (student_id,))
    current = cursor.fetchone()
    
    if current:
        current_semester = current[0]
        is_graduated = current[1]
        
        if is_graduated:
            conn.close()
            return jsonify({'success': False, 'message': 'Student already graduated!'})
        
        if current_semester >= 8:
            cursor.execute('''
                UPDATE student_academic 
                SET is_graduated = 1, status = 'graduated', graduation_date = CURRENT_DATE
                WHERE student_id = ?
            ''', (student_id,))
            message = 'Student graduated successfully!'
        else:
            next_semester = current_semester + 1
            cursor.execute('''
                UPDATE student_academic 
                SET current_semester = ?
                WHERE student_id = ?
            ''', (next_semester, student_id))
            
            cursor.execute('''
                INSERT INTO promotions (student_id, from_semester, to_semester, promoted_by)
                VALUES (?, ?, ?, ?)
            ''', (student_id, current_semester, next_semester, current_user.id))
            
            message = f'Student promoted to Semester {next_semester}'
        
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': message})
    
    conn.close()
    return jsonify({'success': False, 'message': 'Student not found'})

@app.route('/api/bulk-promote', methods=['POST'])
@login_required
def bulk_promote():
    data = request.json
    semester = data.get('semester')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT student_id FROM student_academic 
        WHERE current_semester = ? AND is_graduated = 0
    ''', (semester,))
    students = cursor.fetchall()
    
    promoted_count = 0
    graduated_count = 0
    
    for student in students:
        student_id = student[0]
        
        cursor.execute('SELECT current_semester FROM student_academic WHERE student_id = ?', (student_id,))
        current = cursor.fetchone()
        
        if current and current[0] >= 8:
            cursor.execute('UPDATE student_academic SET is_graduated = 1 WHERE student_id = ?', (student_id,))
            graduated_count += 1
        else:
            cursor.execute('''
                UPDATE student_academic SET current_semester = current_semester + 1
                WHERE student_id = ?
            ''', (student_id,))
            promoted_count += 1
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': f'{promoted_count} promoted, {graduated_count} graduated!'})

# =============================================
# ROUTES - STUDENT PORTAL (Limited Access)
# =============================================

@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sa.id, s.name, s.roll_number, s.class_name
            FROM student_accounts sa
            JOIN students s ON sa.student_id = s.id
            WHERE sa.username = ? AND sa.password_hash = ?
        ''', (username, password_hash))
        student = cursor.fetchone()
        conn.close()
        
        if student:
            session['student_id'] = student[0]
            session['student_name'] = student[1]
            session['student_roll'] = student[2]
            session['student_class'] = student[3]
            session['student_logged_in'] = True
            flash(f'Welcome {student[1]}!', 'success')
            return redirect(url_for('student_dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('student_login.html')

@app.route('/student/dashboard')
@student_required
def student_dashboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM students WHERE name = ?', (session['student_name'],))
    student = cursor.fetchone()
    student_id = student[0] if student else None
    
    if student_id:
        cursor.execute('''
            SELECT 
                COUNT(CASE WHEN status = 'present' THEN 1 END) as present,
                COUNT(*) as total
            FROM attendance WHERE student_id = ?
        ''', (student_id,))
        attendance = cursor.fetchone()
        
        cursor.execute('''
            SELECT sub.subject_name, COUNT(*) as total,
                   COUNT(CASE WHEN a.status = 'present' THEN 1 END) as present
            FROM attendance a JOIN subjects sub ON a.subject_id = sub.id
            WHERE a.student_id = ?
            GROUP BY sub.subject_name
        ''', (student_id,))
        subject_attendance = cursor.fetchall()
    else:
        attendance = (0, 0)
        subject_attendance = []
    
    conn.close()
    
    return render_template('student_dashboard.html',
                         student_name=session['student_name'],
                         student_roll=session['student_roll'],
                         student_class=session['student_class'],
                         attendance=attendance,
                         subject_attendance=subject_attendance)

@app.route('/student/attendance')
@student_required
def student_attendance_view():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM students WHERE name = ?', (session['student_name'],))
    student = cursor.fetchone()
    student_id = student[0] if student else None
    
    if student_id:
        cursor.execute('''
            SELECT a.attendance_date, sub.subject_name, a.status, a.marked_time
            FROM attendance a JOIN subjects sub ON a.subject_id = sub.id
            WHERE a.student_id = ?
            ORDER BY a.attendance_date DESC LIMIT 30
        ''', (student_id,))
        records = cursor.fetchall()
    else:
        records = []
    
    conn.close()
    return render_template('student_attendance.html', records=records)

@app.route('/student/logout')
def student_logout():
    session.clear()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('student_login'))

# =============================================
# ROUTES - PARENT PORTAL
# =============================================

@app.route('/parent/register', methods=['GET', 'POST'])
@login_required
def register_parent():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        student_id = request.form['student_id']
        
        parent_id = f"PAR{datetime.now().strftime('%Y%m%d%H%M%S')}"
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO parents (parent_id, name, email, phone, password_hash)
            VALUES (?, ?, ?, ?, ?)
        ''', (parent_id, name, email, phone, password_hash))
        parent_db_id = cursor.lastrowid
        
        cursor.execute('''
            INSERT INTO student_parents (student_id, parent_id, relationship, is_primary)
            VALUES (?, ?, 'parent', 1)
        ''', (student_id, parent_db_id))
        
        conn.commit()
        conn.close()
        flash('Parent registered successfully!', 'success')
        return redirect(url_for('manage_parents'))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, roll_number FROM students WHERE is_active = 1')
    students = cursor.fetchall()
    conn.close()
    return render_template('register_parent.html', students=students)

@app.route('/parents')
@login_required
def manage_parents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.id, p.parent_id, p.name, p.email, p.phone, s.name as student_name
        FROM parents p
        JOIN student_parents sp ON p.id = sp.parent_id
        JOIN students s ON sp.student_id = s.id
    ''')
    parents = cursor.fetchall()
    conn.close()
    return render_template('parents.html', parents=parents)

@app.route('/parent/login', methods=['GET', 'POST'])
def parent_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.id, p.name, s.name as student_name, s.roll_number, s.class_name
            FROM parents p
            JOIN student_parents sp ON p.id = sp.parent_id
            JOIN students s ON sp.student_id = s.id
            WHERE p.email = ? AND p.password_hash = ?
        ''', (email, password_hash))
        parent = cursor.fetchone()
        conn.close()
        
        if parent:
            session['parent_id'] = parent[0]
            session['parent_name'] = parent[1]
            session['child_name'] = parent[2]
            session['child_roll'] = parent[3]
            session['child_class'] = parent[4]
            session['parent_logged_in'] = True
            flash(f'Welcome {parent[1]}!', 'success')
            return redirect(url_for('parent_dashboard'))
        else:
            flash('Invalid email or password!', 'danger')
    
    return render_template('parent_login.html')

@app.route('/parent/dashboard')
@parent_required
def parent_dashboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM students WHERE name = ?', (session['child_name'],))
    student = cursor.fetchone()
    student_id = student[0] if student else None
    
    if student_id:
        cursor.execute('''
            SELECT 
                COUNT(CASE WHEN status = 'present' THEN 1 END) as present,
                COUNT(*) as total
            FROM attendance WHERE student_id = ?
        ''', (student_id,))
        attendance = cursor.fetchone()
        
        cursor.execute('''
            SELECT a.attendance_date, sub.subject_name, a.status
            FROM attendance a JOIN subjects sub ON a.subject_id = sub.id
            WHERE a.student_id = ?
            ORDER BY a.attendance_date DESC LIMIT 10
        ''', (student_id,))
        recent = cursor.fetchall()
    else:
        attendance = (0, 0)
        recent = []
    
    conn.close()
    
    return render_template('parent_dashboard.html',
                         parent_name=session['parent_name'],
                         child_name=session['child_name'],
                         child_roll=session['child_roll'],
                         child_class=session['child_class'],
                         attendance=attendance,
                         recent=recent)

@app.route('/parent/logout')
def parent_logout():
    session.clear()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('parent_login'))

# =============================================
# ROUTES - LEAVE APPLICATION SYSTEM
# =============================================

@app.route('/leave/apply', methods=['GET', 'POST'])
@student_required
def apply_leave():
    if request.method == 'POST':
        subject = request.form['subject']
        reason = request.form['reason']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM students WHERE name = ?', (session['student_name'],))
        student = cursor.fetchone()
        
        if student:
            cursor.execute('''
                INSERT INTO leave_applications (student_id, subject, reason, start_date, end_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (student[0], subject, reason, start_date, end_date))
            conn.commit()
            flash('Leave application submitted successfully!', 'success')
        else:
            flash('Error submitting application!', 'danger')
        
        conn.close()
        return redirect(url_for('student_dashboard'))
    
    return render_template('apply_leave.html')

@app.route('/leaves')
@login_required
def manage_leaves():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT l.id, s.name, s.roll_number, s.class_name,
               l.subject, l.reason, l.start_date, l.end_date, l.status, l.applied_date
        FROM leave_applications l
        JOIN students s ON l.student_id = s.id
        ORDER BY l.applied_date DESC
    ''')
    leaves = cursor.fetchall()
    conn.close()
    return render_template('manage_leaves.html', leaves=leaves)

@app.route('/leave/update/<int:leave_id>/<status>')
@login_required
def update_leave(leave_id, status):
    if status not in ['approved', 'rejected']:
        flash('Invalid status!', 'danger')
        return redirect(url_for('manage_leaves'))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE leave_applications 
        SET status = ?, approved_date = CURRENT_DATE, remarks = ?
        WHERE id = ?
    ''', (status, f'Application {status} by {current_user.name}', leave_id))
    conn.commit()
    conn.close()
    
    flash(f'Leave application {status}!', 'success')
    return redirect(url_for('manage_leaves'))

# =============================================
# ROUTES - CERTIFICATE GENERATION
# =============================================

@app.route('/certificate/attendance')
@student_required
def download_attendance_certificate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM students WHERE name = ?', (session['student_name'],))
    student = cursor.fetchone()
    student_id = student[0] if student else None
    
    if not student_id:
        flash('Student not found!', 'danger')
        return redirect(url_for('student_dashboard'))
    
    cursor.execute('''
        SELECT 
            COUNT(CASE WHEN status = 'present' THEN 1 END) as present,
            COUNT(*) as total
        FROM attendance WHERE student_id = ?
    ''', (student_id,))
    result = cursor.fetchone()
    
    present = result[0] or 0
    total = result[1] or 0
    percentage = (present / total * 100) if total > 0 else 0
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    c.setFillColorRGB(0.1, 0.1, 0.2)
    c.rect(0, 0, width, height, fill=1)
    
    c.setStrokeColorRGB(0, 0.83, 1)
    c.setLineWidth(2)
    c.rect(40, 40, width-80, height-80)
    
    c.setFillColorRGB(0, 0.83, 1)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height-100, "ATTENDANCE CERTIFICATE")
    
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica", 14)
    c.drawCentredString(width/2, height-160, "This is to certify that")
    
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-200, session['student_name'])
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height-250, f"Roll Number: {session.get('student_roll', 'N/A')}")
    c.drawCentredString(width/2, height-280, f"Class: {session.get('student_class', 'N/A')}")
    c.drawCentredString(width/2, height-320, f"has maintained")
    c.drawCentredString(width/2, height-350, f"{round(percentage, 2)}% attendance")
    
    c.drawCentredString(width/2, height-400, f"Present: {present} days out of {total} days")
    
    c.setFont("Helvetica", 10)
    c.drawCentredString(width/2, height-500, f"Issue Date: {datetime.now().strftime('%B %d, %Y')}")
    
    qr = qrcode.make(f"http://localhost:5000/verify/{student_id}")
    qr_path = BytesIO()
    qr.save(qr_path, 'PNG')
    qr_path.seek(0)
    qr_img = ImageReader(qr_path)
    c.drawImage(qr_img, width-100, 60, width=50, height=50)
    
    c.save()
    buffer.seek(0)
    
    cert_number = f"ATT{student_id}{datetime.now().strftime('%Y%m%d')}"
    cursor.execute('''
        INSERT INTO attendance_certificates (student_id, certificate_number, attendance_percentage)
        VALUES (?, ?, ?)
    ''', (student_id, cert_number, percentage))
    conn.commit()
    conn.close()
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'attendance_certificate_{session["student_name"]}.pdf',
        mimetype='application/pdf'
    )

# =============================================
# ROUTES - ATTENDANCE SYSTEM
# =============================================

@app.route('/attendance/mark')
@login_required
def mark_attendance():
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
    try:
        data = request.json
        image_data = data.get('image')
        subject_id = data.get('subject_id')
        
        if not image_data or not subject_id:
            return jsonify({'success': False, 'message': 'Missing data'})
        
        if not face_encodings:
            load_face_encodings()
            if not face_encodings:
                return jsonify({'success': False, 'message': 'Model not trained. Please train first.'})
        
        image_bytes = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        face_encodings_detected = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings_detected:
            return jsonify({'success': False, 'message': 'Could not encode face'})
        
        for face_encoding in face_encodings_detected:
            distances = face_recognition.face_distance(face_encodings, face_encoding)
            best_match_idx = np.argmin(distances)
            confidence = (1 - distances[best_match_idx]) * 100
            
            if confidence > 40:
                student_id = face_ids[best_match_idx]
                student_name = face_names[best_match_idx]
                
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
                
                cursor.execute('''
                    INSERT INTO attendance (student_id, subject_id, teacher_id, attendance_date, recognition_confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (student_id, subject_id, current_user.id, today, confidence))
                
                conn.commit()
                
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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, subject_name FROM subjects WHERE teacher_id = ?', (current_user.id,))
    subjects = cursor.fetchall()
    
    cursor.execute('SELECT DISTINCT class_name FROM students WHERE is_active = 1')
    classes = cursor.fetchall()
    
    conn.close()
    return render_template('attendance_report.html', subjects=subjects, classes=classes)

@app.route('/api/attendance-data')
@login_required
def api_attendance_data():
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
# ROUTES - PERSONS
# =============================================

@app.route('/persons')
@login_required
def view_persons():
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
    print("\n" + "="*60)
    print("🎯 Face Recognition Attendance System - COMPLETE EDITION")
    print("="*60)
    print("📍 Teacher Portal: http://localhost:5000")
    print("📍 Student Portal: http://localhost:5000/student/login")
    print("📍 Parent Portal: http://localhost:5000/parent/login")
    print("\n🔐 Teacher Login: admin / admin123")
    print("🔐 Student Login: student_{roll_number} / pass{roll_number}")
    print("🔐 Parent Login: (Registered email) / (password)")
    print("="*60 + "\n")
    app.run(debug=True, port=5000, host='0.0.0.0')