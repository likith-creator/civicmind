from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

import sqlite3
import os
from difflib import SequenceMatcher
import re
import math
import uuid
import smtplib
import mimetypes

from datetime import datetime
from email.message import EmailMessage

from dotenv import load_dotenv

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from itsdangerous import URLSafeTimedSerializer

from ultralytics import YOLO


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_RECEIVER = os.getenv("MAIL_RECEIVER")

CIVICMIND_BASE_URL = os.getenv(
    "CIVICMIND_BASE_URL"
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "civicmind-development-key"
)

email_token_serializer = URLSafeTimedSerializer(
    app.secret_key
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE = os.path.join(
    BASE_DIR,
    "civicmind.db"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

DETECTION_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "detections"
)

POTHOLE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "civicmind_pothole.pt"
)

GARBAGE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "civicmind_garbage.pt"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    DETECTION_FOLDER,
    exist_ok=True
)


# ============================================================
# ALLOWED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}
def normalize_text(text):
    """Convert text into a consistent format for comparison."""
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_similarity(text1, text2):
    """Return similarity between two pieces of text from 0 to 1."""
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(None, text1, text2).ratio()

def location_similarity(location1, location2):
    """
    Compare two locations.
    Returns a similarity score between 0 and 1.
    """

    location1 = location1 or ""
    location2 = location2 or ""

    # Try to extract latitude and longitude
    pattern = r"(-?\d+(?:\.\d+)?)[,\s]+(-?\d+(?:\.\d+)?)"

    match1 = re.search(pattern, location1)
    match2 = re.search(pattern, location2)

    # If both locations contain coordinates
    if match1 and match2:
        lat1, lon1 = float(match1.group(1)), float(match1.group(2))
        lat2, lon2 = float(match2.group(1)), float(match2.group(2))

        # Haversine formula
        R = 6371000  # Earth radius in meters

        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

        dlat = lat2 - lat1
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1)
            * math.cos(lat2)
            * math.sin(dlon / 2) ** 2
        )

        distance = 2 * R * math.asin(math.sqrt(a))

        # Convert distance into similarity
        if distance <= 50:
            return 1.0
        elif distance <= 100:
            return 0.8
        elif distance <= 250:
            return 0.5
        else:
            return 0.0

    # If locations are normal text addresses
    return text_similarity(location1, location2)

def find_duplicate_complaint(conn, issue, location, description):
    """
    Compare a new complaint with previous complaints.
    Returns:
        (duplicate_report_id, duplicate_score)
    """

    issue = normalize_text(issue)
    location = location or ""
    description = description or ""

    # Get previous complaints
    previous_reports = conn.execute("""
        SELECT id, issue, location, description
        FROM reports
        ORDER BY id DESC
    """).fetchall()

    best_report_id = None
    best_score = 0.0

    for report in previous_reports:

        old_issue = normalize_text(report["issue"])
        old_location = report["location"] or ""
        old_description = report["description"] or ""

        # Different issue → don't consider it a duplicate
        if issue != old_issue:
            continue

        location_score = location_similarity(
            location,
            old_location
        )

        description_score = text_similarity(
            description,
            old_description
        )

        # Location is slightly more important for civic complaints
        score = (
            location_score * 0.6
            + description_score * 0.4
        )

        if score > best_score:
            best_score = score
            best_report_id = report["id"]

    # Duplicate threshold = 70%
    if best_score >= 0.70:
        return best_report_id, best_score

    return None, None

def count_area_reports(reports):
    """
    Count how many complaints are from the same/similar area
    for each report.
    """

    area_counts = {}

    for report in reports:

        count = 0

        for other_report in reports:

            similarity = location_similarity(
                report["location"],
                other_report["location"]
            )

            # 0.5 means the locations are considered
            # close enough to be in the same area.
            if similarity >= 0.5:
                count += 1

        area_counts[report["id"]] = count

    return area_counts
def calculate_severity_score(
    confidence,
    issue,
    nearby_count=0,
    is_duplicate=False
):
    if confidence is None:
        confidence_points = 0.0
    else:
        confidence = max(0.0, min(float(confidence), 1.0))
        confidence_points = confidence * 5.0

    issue_name = (issue or "").strip().lower()

    if issue_name == "pothole":
        impact_points = 3.0
    elif issue_name == "garbage":
        impact_points = 2.0
    else:
        impact_points = 0.0

    nearby_count = max(0, int(nearby_count or 0))
    nearby_points = min(nearby_count * 0.25, 1.0)

    duplicate_points = 1.0 if is_duplicate else 0.0

    severity_score = (
        confidence_points
        + impact_points
        + nearby_points
        + duplicate_points
    )

    severity_score = min(severity_score, 10.0)

    return round(severity_score, 1)



    
def get_severity_level(severity_score):
    """
    Convert severity score into a severity level.
    """

    if severity_score >= 8:
        return "Critical"

    elif severity_score >= 6:
        return "High"

    elif severity_score >= 3:
        return "Medium"

    else:
        return "Low"
    
def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO models...")

pothole_model = YOLO(POTHOLE_MODEL_PATH)

print("Pothole YOLO model loaded successfully!")

garbage_model = YOLO(GARBAGE_MODEL_PATH)

print("Garbage YOLO model loaded successfully!")

# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():

    conn = get_db()

    # --------------------------------------------------------
    # USERS TABLE
    # --------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
        """
    )

    # --------------------------------------------------------
    # REPORTS TABLE
    # --------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            location TEXT NOT NULL,

            issue TEXT NOT NULL,

            description TEXT NOT NULL,

            image TEXT,

            detected_image TEXT,

            confidence REAL,

            detected_issue TEXT,

            priority TEXT,

            status TEXT DEFAULT 'Pending',

            FOREIGN KEY (user_id)
                REFERENCES users(id)
        )
        """
    )

    # --------------------------------------------------------
    # ADD NEW STATUS TIMESTAMP COLUMNS
    # --------------------------------------------------------

    existing_columns = {
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(reports)"
        ).fetchall()
    }

    if "created_at" not in existing_columns:

        conn.execute(
            """
            ALTER TABLE reports
            ADD COLUMN created_at TEXT
            """
        )

    if "reviewed_at" not in existing_columns:

        conn.execute(
            """
            ALTER TABLE reports
            ADD COLUMN reviewed_at TEXT
            """
        )

    if "work_started_at" not in existing_columns:

        conn.execute(
            """
            ALTER TABLE reports
            ADD COLUMN work_started_at TEXT
            """
        )

    if "completed_at" not in existing_columns:

        conn.execute(
            """
            ALTER TABLE reports
            ADD COLUMN completed_at TEXT
            """
        )
        # Duplicate complaint fields
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(reports)").fetchall()
    }

    if "duplicate_of" not in existing_columns:
        conn.execute("""
            ALTER TABLE reports
            ADD COLUMN duplicate_of INTEGER
        """)

    if "duplicate_score" not in existing_columns:
        conn.execute("""
            ALTER TABLE reports
            ADD COLUMN duplicate_score REAL
     """)
    
    
    if "severity_score" not in existing_columns:
        conn.execute("""
            ALTER TABLE reports
            ADD COLUMN severity_score REAL
        """)
    if "severity_level" not in existing_columns:
        conn.execute("""
            ALTER TABLE reports
            ADD COLUMN severity_level TEXT
        """)

    if "latitude" not in existing_columns:
        conn.execute("""
            ALTER TABLE reports
            ADD COLUMN latitude REAL
        """)

    if "longitude" not in existing_columns:
        conn.execute("""
            ALTER TABLE reports
            ADD COLUMN longitude REAL
        """)
        
    conn.commit()

    conn.close()
# ============================================================
# EMAIL SENDING FUNCTION
# ============================================================

def send_email(
    report_id,
    name,
    email,
    location,
    issue,
    description,
    detected_issue,
    confidence,
    priority,
    duplicate_id=None,
    duplicate_score=None,
    area_count=None,
    image_path=None,
    detection_path=None
):

    if (
        not MAIL_USERNAME
        or not MAIL_PASSWORD
        or not MAIL_RECEIVER
    ):

        print(
            "Email configuration is missing."
        )

        return False

    try:

        msg = EmailMessage()

        # ----------------------------------------------------
        # SUBJECT
        # ----------------------------------------------------

        msg["Subject"] = (
            f"CivicMind AI - "
            f"{priority} Priority Civic Issue "
            f"#{report_id}"
        )

        msg["From"] = MAIL_USERNAME

        msg["To"] = MAIL_RECEIVER

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if confidence is not None:

            confidence_text = (
                f"{confidence * 100:.2f}%"
            )

        else:

            confidence_text = "N/A"

        # ----------------------------------------------------
        # EMAIL BODY
        # ----------------------------------------------------

        if duplicate_id:
            duplicate_section = f"""
        ----------------------------------------
        ⚠️ POSSIBLE DUPLICATE
        ----------------------------------------

        This complaint is similar to Report #{duplicate_id}.

        Similarity:
        {duplicate_score * 100:.1f}%

        Reports in this area:
        {area_count}

        ----------------------------------------
        """
        else:
            duplicate_section = """
        ----------------------------------------
        DUPLICATE CHECK
        ----------------------------------------

        No possible duplicate complaint detected.

        ----------------------------------------
        """
        
        
        body = f"""
CIVICMIND AI
NEW CIVIC ISSUE REPORT

----------------------------------------
REPORT DETAILS
----------------------------------------

Report ID: #{report_id}

Citizen Name: {name}

Citizen Email: {email}

Reported Issue: {issue}

AI Detection: {detected_issue}

AI Confidence: {confidence_text}



Location:
{location}

Description:
{description}

{duplicate_section}

----------------------------------------
STATUS
----------------------------------------

Pending

Please review this report.

CivicMind AI
Municipal Civic Complaint System
"""

        msg.set_content(body)

        # ----------------------------------------------------
        # CREATE SECURE ACTION TOKEN
        # ----------------------------------------------------

        token = email_token_serializer.dumps(
            {
                "report_id": report_id
            }
        )

        if CIVICMIND_BASE_URL:

            base_url = (
                CIVICMIND_BASE_URL.rstrip("/")
            )

        else:

            base_url = request.url_root.rstrip("/")

        review_url = (
            f"{base_url}/email-action/"
            f"{token}/review"
        )

        # ----------------------------------------------------
        # HTML EMAIL
        # ----------------------------------------------------
        
        
        if duplicate_id:

            duplicate_html = f"""
            <div style="
                background:#fff3cd;
                border:1px solid #ffc107;
                padding:15px;
                border-radius:8px;
                margin:20px 0;
            ">

                <h3 style="margin-top:0;">
                    ⚠️ Possible Duplicate
                </h3>

                <p>
                    This complaint is similar to
                    <strong>Report #{duplicate_id}</strong>.
                </p>

                <p>
                    <strong>Similarity:</strong>
                    {duplicate_score * 100:.1f}%
                </p>

                <p>
                    <strong>Reports in this area:</strong>
                    {area_count}
                </p>

            </div>
            """

        else:

            duplicate_html = """
            <div style="
                background:#f1f3f5;
                padding:15px;
                border-radius:8px;
                margin:20px 0;
            ">

                <strong>
                    ✓ No possible duplicate detected
                </strong>

            </div>
            """
        
        
        
        
        
        
        
        
        
        
        html_body = f"""
        <html>

        <body style="
            font-family: Arial, sans-serif;
            line-height: 1.6;
        ">

            <h2>CivicMind AI</h2>

            <h3>
                New Civic Issue Report
            </h3>

            <p>
                <strong>Report ID:</strong>
                #{report_id}
            </p>

            <p>
                <strong>Citizen:</strong>
                {name}
            </p>

            <p>
                <strong>Issue:</strong>
                {issue}
            </p>

            <p>
                <strong>AI Detection:</strong>
                {detected_issue}
            </p>

            <p>
                <strong>AI Confidence:</strong>
                {confidence_text}
            </p>

            

            <p>
                <strong>Location:</strong>
                {location}
            </p>

            <p>
                <strong>Description:</strong>
                {description}
            </p>
            
            {duplicate_html}

            <hr>

            <p>
                <strong>Status:</strong>
                Pending
            </p>

            <br>

            <a href="{review_url}"
               style="
               display:inline-block;
               padding:12px 20px;
               background:#2563eb;
               color:white;
               text-decoration:none;
               border-radius:6px;
               font-weight:bold;
               ">

                START REVIEW

            </a>

            <br><br>

            <p>
                CivicMind AI Municipal Management System
            </p>

        </body>

        </html>
        """

        msg.add_alternative(
            html_body,
            subtype="html"
        )

        # ----------------------------------------------------
        # ATTACH IMAGES
        # ----------------------------------------------------

        attachments = [
            image_path,
            detection_path
        ]

        for file_path in attachments:

            if (
                file_path
                and
                os.path.exists(file_path)
            ):

                mime_type, _ = mimetypes.guess_type(
                    file_path
                )

                if mime_type:

                    maintype, subtype = (
                        mime_type.split("/", 1)
                    )

                else:

                    maintype = "application"
                    subtype = "octet-stream"

                with open(
                    file_path,
                    "rb"
                ) as file:

                    file_data = file.read()

                msg.add_attachment(
                    file_data,
                    maintype=maintype,
                    subtype=subtype,
                    filename=os.path.basename(
                        file_path
                    )
                )

        # ----------------------------------------------------
        # SEND EMAIL
        # ----------------------------------------------------

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                MAIL_USERNAME,
                MAIL_PASSWORD
            )
            print("SMTP login successful.")


            smtp.send_message(
                msg
            )

        print(
            "Municipal notification email sent successfully."
        )

        return True

    except Exception as error:

        print(
            "Email sending failed:",
            error
        )

        return False


# ============================================================
# SEND MUNICIPAL FOLLOW-UP EMAIL
# ============================================================

def send_followup_email(
    report,
    action
):

    if (
        not MAIL_USERNAME
        or not MAIL_PASSWORD
        or not MAIL_RECEIVER
    ):

        return False

    try:

        report_id = report["id"]

        token = email_token_serializer.dumps(
            {
                "report_id": report_id
            }
        )

        if CIVICMIND_BASE_URL:

            base_url = (
                CIVICMIND_BASE_URL.rstrip("/")
            )

        else:

            base_url = request.url_root.rstrip("/")

        action_url = (
            f"{base_url}/email-action/"
            f"{token}/{action}"
        )

        # ----------------------------------------------------
        # ACTION DETAILS
        # ----------------------------------------------------

        if action == "start":

            subject = (
                f"CivicMind AI - "
                f"Report #{report_id} "
                f"Ready to Start Work"
            )

            button_text = "START WORK"

            status_text = "Under Review"

        elif action == "complete":

            subject = (
                f"CivicMind AI - "
                f"Report #{report_id} "
                f"Ready for Completion"
            )

            button_text = (
                "MARK WORK COMPLETED"
            )

            status_text = "In Progress"

        else:

            return False

        # ----------------------------------------------------
        # PLAIN TEXT
        # ----------------------------------------------------

        body = f"""
CIVICMIND AI

Report #{report_id}

Issue:
{report["issue"]}

Location:
{report["location"]}


Current Status:
{status_text}

Please take the next action:

{action_url}
"""

        msg = EmailMessage()

        msg["Subject"] = subject

        msg["From"] = MAIL_USERNAME

        msg["To"] = MAIL_RECEIVER

        msg.set_content(body)

        # ----------------------------------------------------
        # HTML
        # ----------------------------------------------------

        html_body = f"""
        <html>

        <body style="
            font-family: Arial, sans-serif;
        ">

            <h2>CivicMind AI</h2>

            <h3>
                Report #{report_id}
            </h3>

            <p>
                <strong>Issue:</strong>
                {report["issue"]}
            </p>

            <p>
                <strong>Location:</strong>
                {report["location"]}
            </p>


            <p>
                <strong>Current Status:</strong>
                {status_text}
            </p>

            <br>

            <a href="{action_url}"
               style="
               display:inline-block;
               padding:12px 20px;
               background:#2563eb;
               color:white;
               text-decoration:none;
               border-radius:6px;
               font-weight:bold;
               ">

                {button_text}

            </a>

            <br><br>

            <p>
                CivicMind AI Municipal Management System
            </p>

        </body>

        </html>
        """

        msg.add_alternative(
            html_body,
            subtype="html"
        )

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                MAIL_USERNAME,
                MAIL_PASSWORD
            )

            smtp.send_message(
                msg
            )

        print(
            "Municipal follow-up email sent."
        )

        return True

    except Exception as error:

        print(
            "Follow-up email failed:",
            error
        )

        return False


# ============================================================
# SEND CITIZEN COMPLETION EMAIL
# ============================================================

def send_citizen_completion_email(
    report
):

    if (
        not MAIL_USERNAME
        or not MAIL_PASSWORD
        or not report["email"]
    ):

        return False

    try:

        msg = EmailMessage()

        msg["Subject"] = (
            f"CivicMind AI - "
            f"Work Completed for Report "
            f"#{report['id']}"
        )

        msg["From"] = MAIL_USERNAME

        msg["To"] = report["email"]

        body = f"""
Hello {report["name"]},

Your CivicMind AI complaint has been processed.

Report ID:
#{report["id"]}

Issue:
{report["issue"]}

Location:
{report["location"]}

Status:
Work Completed

The municipal department has marked the
reported work as completed.

Thank you for using CivicMind AI.

CivicMind AI
Municipal Civic Complaint System
"""

        msg.set_content(body)

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                MAIL_USERNAME,
                MAIL_PASSWORD
            )

            smtp.send_message(
                msg
            )

        print(
            "Citizen completion email sent."
        )

        return True

    except Exception as error:

        print(
            "Citizen email failed:",
            error
        )

        return False


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if (
            not name
            or not email
            or not password
        ):

            flash(
                "Please fill in all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # ----------------------------------------------------
        # DATABASE
        # ----------------------------------------------------

        conn = get_db()

        existing_user = conn.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        if existing_user:

            conn.close()

            flash(
                "An account with this email already exists.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        # ----------------------------------------------------
        # HASH PASSWORD
        # ----------------------------------------------------

        password_hash = generate_password_hash(
            password
        )

        conn.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                password_hash
            )

            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                password_hash
            )
        )

        conn.commit()

        conn.close()

        flash(
            "Registration successful. Please log in.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        conn = get_db()

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        conn.close()

        if (
            user
            and
            check_password_hash(
                user["password_hash"],
                password
            )
        ):

            session["user_id"] = user["id"]

            session["user_name"] = user["name"]

            session["user_email"] = user["email"]

            return redirect(
                url_for("report")
            )

        flash(
            "Invalid email or password.",
            "error"
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# ============================================================
# REPORT
# ============================================================

@app.route(
    "/report",
    methods=["GET", "POST"]
)
def report():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if "user_id" not in session:

        flash(
            "Please log in before submitting a report.",
            "error"
        )

        return redirect(
            url_for("login")
        )

    # --------------------------------------------------------
    # SHOW REPORT PAGE
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "report.html",
            user_name=session["user_name"],
            user_email=session["user_email"]
        )

    # --------------------------------------------------------
    # FORM DATA
    # --------------------------------------------------------

    issue = request.form.get(
        "issue",
        ""
    ).strip()

    description = request.form.get(
       "description",
        ""
    ).strip()

    location = request.form.get(
        "location",
        ""
    ).strip()

    latitude = request.form.get(
        "latitude",
            ""
    ).strip()

    longitude = request.form.get(
        "longitude",
        ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not issue:

        flash(
            "Please select an issue type.",
            "error"
        )

        return redirect(
            url_for("report")
        )

    if not description:

        flash(
            "Please provide a description.",
            "error"
        )

        return redirect(
            url_for("report")
        )

    if not location:

        flash(
            "Please provide a location.",
            "error"
        )

        return redirect(
            url_for("report")
        )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = request.files.get(
        "image"
    )

    if (
        not image
        or
        image.filename == ""
    ):

        flash(
            "Please upload or capture an image.",
            "error"
        )

        return redirect(
            url_for("report")
        )

    if not allowed_file(
        image.filename
    ):

        flash(
            "Invalid image format.",
            "error"
        )

        return redirect(
            url_for("report")
        )

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    original_filename = secure_filename(
        image.filename
    )

    extension = original_filename.rsplit(
        ".",
        1
    )[1].lower()

    unique_filename = (
        str(uuid.uuid4())
        + "."
        + extension
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )

    image.save(
        image_path
    )

    print(
        f"Image saved: {image_path}"
    )

   

    # --------------------------------------------------------
    # SELECT AI MODEL
    # --------------------------------------------------------

    if issue.lower() == "garbage":

        selected_model = garbage_model

        print(
            "Using Garbage Detection Model"
        )

    else:

        selected_model = pothole_model

        print(
            "Using Pothole Detection Model"
        )

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    print(
        "Running YOLO detection..."
    )

    try:

        results = selected_model.predict(
            source=image_path,
            conf=0.25,
            save=False,
            verbose=False
        )

    except Exception as error:

        print(
            "YOLO error:",
            error
        )

        flash(
            "AI detection failed. Please try another image.",
            "error"
        )

        return redirect(
            url_for("report")
        )
    # --------------------------------------------------------
    # PROCESS DETECTION
    # --------------------------------------------------------

    detected_issue = "No detection"

    confidence = None

    result = results[0]

    if (
        result.boxes is not None
        and
        len(result.boxes) > 0
    ):

        best_index = (
            result.boxes.conf.argmax()
        )

        confidence = float(
            result.boxes.conf[
                best_index
            ].item()
        )

        class_id = int(
            result.boxes.cls[
                best_index
            ].item()
        )

        detected_issue = (
            result.names[class_id]
        )

        confidence = round(
            confidence,
            4
        )

    # --------------------------------------------------------
    # SAVE DETECTION IMAGE
    # --------------------------------------------------------

    detection_filename = (
        str(uuid.uuid4())
        + ".jpg"
    )

    detection_path = os.path.join(
        DETECTION_FOLDER,
        detection_filename
    )

    try:

        result.save(
            filename=detection_path
        )

    except Exception as error:

        print(
            "Could not save detection image:",
            error
        )

        detection_filename = None

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    priority = "Medium"

    if confidence is not None:

        if confidence >= 0.80:

            priority = "High"

        elif confidence >= 0.50:

            priority = "Medium"

        else:

            priority = "Low"

    # --------------------------------------------------------
    # CREATED TIME
    # --------------------------------------------------------

    created_at = datetime.now().isoformat(
        timespec="seconds"
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    conn = get_db()

# --------------------------------------------------------
# DUPLICATE COMPLAINT DETECTION
# --------------------------------------------------------

    duplicate_id, duplicate_score = find_duplicate_complaint(
        conn,
        issue,
        location,
        description
    )

    if duplicate_id:
        print(
            f"Possible duplicate found: "
            f"Report #{duplicate_id} "
            f"({duplicate_score * 100:.1f}%)"
        )
    else:
        print("No duplicate complaint found.")



# --------------------------------------------------------
# COUNT OTHER REPORTS IN THE SAME AREA
# --------------------------------------------------------
# --------------------------------------------------------
# COUNT NEARBY EXISTING REPORTS
# --------------------------------------------------------

 # --------------------------------------------------------
# COUNT OTHER REPORTS IN THE SAME AREA
# --------------------------------------------------------

    all_reports = conn.execute(
        """
        SELECT id, location
        FROM reports
        """
    ).fetchall()

    all_reports = conn.execute(
        """
        SELECT id, issue, location
        FROM reports
        """
    ).fetchall()

    nearby_count = 0

    for existing_report in all_reports:

        existing_issue = (
            existing_report["issue"] or ""
        ).strip().lower()

        current_issue = (
            issue or ""
        ).strip().lower()

        if existing_issue != current_issue:
            continue

        if location_similarity(
            location,
            existing_report["location"]
    ) >= 0.5:

            nearby_count += 1

# --------------------------------------------------------
# CALCULATE SEVERITY SCORE
# --------------------------------------------------------
# --------------------------------------------------------
# CALCULATE SEVERITY SCORE
# --------------------------------------------------------

    severity_score = calculate_severity_score(
        confidence=confidence,
        issue=issue,
        nearby_count=nearby_count,
        is_duplicate=(duplicate_id is not None)
    )


    severity_level = get_severity_level(
        severity_score
    )


    print(
        f"Severity Score: {severity_score}/10"
    )


    print(
        f"Severity Level: {severity_level}"
    )
    


# --------------------------------------------------------
# SAVE REPORT
# --------------------------------------------------------

    cursor = conn.execute(
        """
        INSERT INTO reports
        (
            user_id,
            location,
            latitude,
            longitude,
            issue,
            description,
            image,
            detected_image,
            confidence,
            detected_issue,
            priority,
            status,
            created_at,
            duplicate_of,
            duplicate_score,
            severity_score,
            severity_level
        )

        VALUES
        (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            session["user_id"],
            location,
            latitude,
            longitude,
            issue,
            description,
            unique_filename,
            detection_filename,
            confidence,
            detected_issue,
            priority,
            "Pending",
            created_at,
            duplicate_id,
            duplicate_score,
            severity_score,
            severity_level
        )
    )


    report_id = cursor.lastrowid

    conn.commit()

    conn.close()

    print(
        "Report saved successfully."
    )

    # --------------------------------------------------------
    # GET REPORT
    # --------------------------------------------------------

    conn = get_db()

    report_data = conn.execute(
        """
        SELECT
            reports.*,
            users.name,
            users.email

        FROM reports

        JOIN users
        ON reports.user_id = users.id

        WHERE reports.id = ?
        """,
        (report_id,)
    ).fetchone()

    conn.close()

    # --------------------------------------------------------
    # SEND MUNICIPAL EMAIL
    # --------------------------------------------------------

    email_sent = send_email(
        report_id=report_id,

        name=session.get(
            "user_name",
            "Citizen"
        ),

        email=session.get(
            "user_email",
            ""
        ),

        location=location,

        issue=issue,

        description=description,

        detected_issue=detected_issue,

        confidence=confidence,

        priority=priority,
        
        duplicate_id=duplicate_id,
        
        duplicate_score=duplicate_score,
        
        area_count=nearby_count,

        image_path=image_path,

        detection_path=(
            detection_path
            if detection_filename
            else None
        )
    )

    if email_sent:

        print(
            "Municipal department has been notified."
        )

    else:

        print(
            "Report saved, but email could not be sent."
        )

    flash(
        "Complaint submitted successfully.",
        "success"
    )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        flash(
            "Please log in to view the dashboard.",
            "error"
        )

        return redirect(
            url_for("login")
        )

    conn = get_db()

    reports = conn.execute(
        """
        SELECT
            reports.*,
            users.name,
            users.email

        FROM reports

        JOIN users
        ON reports.user_id = users.id

        ORDER BY reports.id DESC
        """
    ).fetchall()

    conn.close()
    area_counts = count_area_reports(reports)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total = len(reports)

    pending = sum(
        1
        for report in reports
        if report["status"] == "Pending"
    )

    under_review = sum(
        1
        for report in reports
        if report["status"] == "Under Review"
    )

    in_progress = sum(
        1
        for report in reports
        if report["status"] == "In Progress"
    )

    work_completed = sum(
        1
        for report in reports
        if report["status"] == "Work Completed"
    )
    # --------------------------------------------------------
# DUPLICATE STATISTICS
# --------------------------------------------------------

    duplicates = sum(
        1
        for report in reports
        if report["duplicate_of"] is not None
    )

    return render_template(
        "dashboard.html",

        reports=reports,

        total=total,

        pending=pending,

        under_review=under_review,

        in_progress=in_progress,

        work_completed=work_completed,

        duplicates=duplicates,    
        
        area_counts=area_counts   
    )

@app.route("/dashboard/reports")
def dashboard_reports():

    if "user_id" not in session:

        flash(
            "Please log in to view the dashboard.",
            "error"
        )

        return redirect(
            url_for("login")
        )

    conn = get_db()

    reports = conn.execute(
        """
        SELECT
            reports.*,
            users.name,
            users.email

        FROM reports

        JOIN users
        ON reports.user_id = users.id

        ORDER BY reports.id DESC
        """
    ).fetchall()

    conn.close()

    area_counts = count_area_reports(reports)

    return render_template(
        "reports.html",

        reports=reports,

        area_counts=area_counts
    )
    
@app.route("/dashboard/map")
def dashboard_map():

    if "user_id" not in session:
        flash("Please log in to view the dashboard.", "error")
        return redirect(url_for("login"))

    conn = get_db()

    reports = conn.execute(
        """
        SELECT
            reports.*,
            users.name,
            users.email
        FROM reports
        JOIN users
        ON reports.user_id = users.id
        WHERE reports.latitude IS NOT NULL
        AND reports.longitude IS NOT NULL
        ORDER BY reports.id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "map.html",
        reports=reports
    )
    
@app.route("/dashboard/analytics")
def dashboard_analytics():

    if "user_id" not in session:
        flash("Please log in to view the dashboard.", "error")
        return redirect(url_for("login"))

    conn = get_db()

    reports = conn.execute(
        """
        SELECT
            reports.*,
            users.name,
            users.email
        FROM reports
        JOIN users
        ON reports.user_id = users.id
        ORDER BY reports.id DESC
        """
    ).fetchall()

    conn.close()

    # -----------------------------
    # Basic statistics
    # -----------------------------

    total = len(reports)

    potholes = sum(
        1 for report in reports
        if report["issue"] == "Pothole"
    )

    garbage = sum(
        1 for report in reports
        if report["issue"] == "Garbage"
    )

    duplicates = sum(
        1 for report in reports
        if report["duplicate_of"] is not None
    )

    pending = sum(
        1 for report in reports
        if report["status"] == "Pending"
    )

    under_review = sum(
        1 for report in reports
        if report["status"] == "Under Review"
    )

    in_progress = sum(
        1 for report in reports
        if report["status"] == "In Progress"
    )

    work_completed = sum(
        1 for report in reports
        if report["status"] == "Work Completed"
    )

    critical = sum(
        1 for report in reports
        if report["severity_level"] == "Critical"
    )

    high = sum(
        1 for report in reports
        if report["severity_level"] == "High"
    )

    medium = sum(
        1 for report in reports
        if report["severity_level"] == "Medium"
    )

    low = sum(
        1 for report in reports
        if report["severity_level"] == "Low"
    )

    # -----------------------------
    # AI confidence
    # -----------------------------

    confidence_values = [
        report["confidence"]
        for report in reports
        if report["confidence"] is not None
    ]

    if confidence_values:
        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        ) * 100
    else:
        average_confidence = 0

    # -----------------------------
    # Severity score
    # -----------------------------

    severity_values = [
        report["severity_score"]
        for report in reports
        if report["severity_score"] is not None
    ]

    if severity_values:
        average_severity = (
            sum(severity_values)
            / len(severity_values)
        )
    else:
        average_severity = 0

    # -----------------------------
    # Completion percentage
    # -----------------------------

    if total > 0:
        completion_rate = (
            work_completed / total
        ) * 100
    else:
        completion_rate = 0

    return render_template(
        "analytics.html",

        total=total,

        potholes=potholes,
        garbage=garbage,
        duplicates=duplicates,

        pending=pending,
        under_review=under_review,
        in_progress=in_progress,
        work_completed=work_completed,

        critical=critical,
        high=high,
        medium=medium,
        low=low,

        average_confidence=round(
            average_confidence,
            1
        ),

        average_severity=round(
            average_severity,
            1
        ),

        completion_rate=round(
            completion_rate,
            1
        )
    )
# ============================================================
# DASHBOARD - START REVIEW
# ============================================================

@app.route(
    "/report/<int:id>/review"
)
def start_review(id):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    conn = get_db()

    report = conn.execute(
        """
        SELECT *
        FROM reports
        WHERE id = ?
        """,
        (id,)
    ).fetchone()

    if not report:

        conn.close()

        flash(
            "Report not found.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    if report["status"] != "Pending":

        conn.close()

        flash(
            "This report cannot be moved to Under Review.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    reviewed_at = datetime.now().isoformat(
        timespec="seconds"
    )

    conn.execute(
        """
        UPDATE reports

        SET
            status = ?,
            reviewed_at = ?

        WHERE id = ?
        """,
        (
            "Under Review",
            reviewed_at,
            id
        )
    )

    conn.commit()

    updated_report = conn.execute(
        """
        SELECT
            reports.*,
            users.name,
            users.email

        FROM reports

        JOIN users
        ON reports.user_id = users.id

        WHERE reports.id = ?
        """,
        (id,)
    ).fetchone()

    conn.close()

    # --------------------------------------------------------
    # SEND NEXT EMAIL
    # --------------------------------------------------------

    send_followup_email(
        updated_report,
        "start"
    )

    flash(
        "Report moved to Under Review.",
        "success"
    )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# DASHBOARD - START WORK
# ============================================================

@app.route(
    "/report/<int:id>/start"
)
def start_work(id):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    conn = get_db()

    report = conn.execute(
        """
        SELECT *
        FROM reports
        WHERE id = ?
        """,
        (id,)
    ).fetchone()

    if not report:

        conn.close()

        flash(
            "Report not found.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    if report["status"] != "Under Review":

        conn.close()

        flash(
            "The report must be Under Review before work can start.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    work_started_at = datetime.now().isoformat(
        timespec="seconds"
    )

    conn.execute(
        """
        UPDATE reports

        SET
            status = ?,
            work_started_at = ?

        WHERE id = ?
        """,
        (
            "In Progress",
            work_started_at,
            id
        )
    )

    conn.commit()

    updated_report = conn.execute(
        """
        SELECT
            reports.*,
            users.name,
            users.email

        FROM reports

        JOIN users
        ON reports.user_id = users.id

        WHERE reports.id = ?
        """,
        (id,)
    ).fetchone()

    conn.close()

    # --------------------------------------------------------
    # SEND NEXT EMAIL
    # --------------------------------------------------------

    send_followup_email(
        updated_report,
        "complete"
    )

    flash(
        "Work has been started.",
        "success"
    )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# DASHBOARD - COMPLETE WORK
# ============================================================

@app.route(
    "/report/<int:id>/complete"
)
def complete_work(id):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    conn = get_db()

    report = conn.execute(
        """
        SELECT *
        FROM reports
        WHERE id = ?
        """,
        (id,)
    ).fetchone()

    if not report:

        conn.close()

        flash(
            "Report not found.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    if report["status"] != "In Progress":

        conn.close()

        flash(
            "The report must be In Progress before it can be completed.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    completed_at = datetime.now().isoformat(
        timespec="seconds"
    )

    conn.execute(
        """
        UPDATE reports

        SET
            status = ?,
            completed_at = ?

        WHERE id = ?
        """,
        (
            "Work Completed",
            completed_at,
            id
        )
    )

    conn.commit()

    updated_report = conn.execute(
        """
        SELECT
            reports.*,
            users.name,
            users.email

        FROM reports

        JOIN users
        ON reports.user_id = users.id

        WHERE reports.id = ?
        """,
        (id,)
    ).fetchone()

    conn.close()

    # --------------------------------------------------------
    # INFORM CITIZEN
    # --------------------------------------------------------

    send_citizen_completion_email(
        updated_report
    )

    flash(
        "Work marked as completed.",
        "success"
    )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# EMAIL ACTION PAGE
# ============================================================

def email_action_page(
    report,
    message
):

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>CivicMind AI</title>

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
        >

        <style>

            body {{
                font-family: Arial, sans-serif;
                background: #f5f7fb;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
            }}

            .box {{
                background: white;
                padding: 35px;
                border-radius: 12px;
                box-shadow:
                    0 5px 20px
                    rgba(0, 0, 0, 0.10);
                text-align: center;
                max-width: 500px;
                width: 90%;
            }}

            h1 {{
                color: #2563eb;
            }}

            .status {{
                font-size: 20px;
                font-weight: bold;
                margin-top: 20px;
            }}

        </style>

    </head>

    <body>

        <div class="box">

            <h1>CivicMind AI</h1>

            <h2>
                Report #{report["id"]}
            </h2>

            <p>
                {message}
            </p>

            <div class="status">

                Current Status:
                {report["status"]}

            </div>

        </div>

    </body>

    </html>
    """


# ============================================================
# EMAIL ACTION
# ============================================================

@app.route(
    "/email-action/<token>/<action>"
)
def email_action(
    token,
    action
):

    # --------------------------------------------------------
    # CHECK ACTION
    # --------------------------------------------------------

    if action not in {
        "review",
        "start",
        "complete"
    }:

        return (
            "Invalid action.",
            400
        )

    # --------------------------------------------------------
    # VERIFY TOKEN
    # --------------------------------------------------------

    try:

        payload = email_token_serializer.loads(
            token,
            max_age=60 * 60 * 24 * 30
        )

        report_id = int(
            payload["report_id"]
        )

    except Exception:

        return (
            """
            <h2>
                Invalid or expired CivicMind AI action link.
            </h2>
            """,
            403
        )

    # --------------------------------------------------------
    # GET REPORT
    # --------------------------------------------------------

    conn = get_db()

    report = conn.execute(
        """
        SELECT
            reports.*,
            users.name,
            users.email

        FROM reports

        JOIN users
        ON reports.user_id = users.id

        WHERE reports.id = ?
        """,
        (report_id,)
    ).fetchone()

    if not report:

        conn.close()

        return (
            "Report not found.",
            404
        )

    # ========================================================
    # REVIEW
    # ========================================================

    if action == "review":

        if report["status"] != "Pending":

            conn.close()

            return email_action_page(
                report,
                "This report has already been reviewed."
            )

        reviewed_at = datetime.now().isoformat(
            timespec="seconds"
        )

        conn.execute(
            """
            UPDATE reports

            SET
                status = ?,
                reviewed_at = ?

            WHERE id = ?
            """,
            (
                "Under Review",
                reviewed_at,
                report_id
            )
        )

        conn.commit()

        updated_report = conn.execute(
            """
            SELECT
                reports.*,
                users.name,
                users.email

            FROM reports

            JOIN users
            ON reports.user_id = users.id

            WHERE reports.id = ?
            """,
            (report_id,)
        ).fetchone()

        conn.close()

        # Send START WORK email
        send_followup_email(
            updated_report,
            "start"
        )

        return email_action_page(
            updated_report,
            "The report has been moved to Under Review."
        )

    # ========================================================
    # START WORK
    # ========================================================

    if action == "start":

        if report["status"] != "Under Review":

            conn.close()

            return email_action_page(
                report,
                "The report must be Under Review before work can start."
            )

        work_started_at = datetime.now().isoformat(
            timespec="seconds"
        )

        conn.execute(
            """
            UPDATE reports

            SET
                status = ?,
                work_started_at = ?

            WHERE id = ?
            """,
            (
                "In Progress",
                work_started_at,
                report_id
            )
        )

        conn.commit()

        updated_report = conn.execute(
            """
            SELECT
                reports.*,
                users.name,
                users.email

            FROM reports

            JOIN users
            ON reports.user_id = users.id

            WHERE reports.id = ?
            """,
            (report_id,)
        ).fetchone()

        conn.close()

        # Send COMPLETE email
        send_followup_email(
            updated_report,
            "complete"
        )

        return email_action_page(
            updated_report,
            "Work has been started for this report."
        )

    # ========================================================
    # COMPLETE WORK
    # ========================================================

    if action == "complete":

        if report["status"] != "In Progress":

            conn.close()

            return email_action_page(
                report,
                "The report must be In Progress before it can be completed."
            )

        completed_at = datetime.now().isoformat(
            timespec="seconds"
        )

        conn.execute(
            """
            UPDATE reports

            SET
                status = ?,
                completed_at = ?

            WHERE id = ?
            """,
            (
                "Work Completed",
                completed_at,
                report_id
            )
        )

        conn.commit()

        updated_report = conn.execute(
            """
            SELECT
                reports.*,
                users.name,
                users.email

            FROM reports

            JOIN users
            ON reports.user_id = users.id

            WHERE reports.id = ?
            """,
            (report_id,)
        ).fetchone()

        conn.close()

        # Inform citizen
        send_citizen_completion_email(
            updated_report
        )

        return email_action_page(
            updated_report,
            "Work has been marked as completed."
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    init_db()

    app.run(
        debug=True
    )