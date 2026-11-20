import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import sqlite3
import pandas as pd
import os
from datetime import datetime
import requests
from fpdf import FPDF
import streamlit as st
import google.generativeai as genai
from EDA import ExploratoryDataAnalysis

st.set_page_config(page_title="Chicken Disease Detection", layout="wide")


# ================================
# Custom Styling
# ================================
st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: {"#27F2F5"};
            color: {"#000000"};
        }}
        </style>
        """,
        unsafe_allow_html=True
)
st.subheader("AI-Powered Poultry Disease Diagnosis and Veterinary Assistance System")
st.sidebar.subheader("Intelligent Poultry Disease Surveillance")

st.sidebar.image("Image/s1.png")

def home_page():
    st.image("Image/coverpage.png")
    st.markdown("""
    
    Poultry farming plays a crucial role in global food production, yet diseases in chickens continue to cause major economic losses for farmers.  
    Most poultry diseases spread rapidly and require **quick diagnosis** and **immediate treatment** to prevent large-scale mortality.
    
    This web application provides a fast and reliable solution by using **Artificial Intelligence** to analyze chicken images and identify common diseases instantly.  
    Farmers no longer need to depend on laboratory tests, physical inspection delays, or limited expert availability.  
    A single image is enough for the system to detect visual symptoms and provide medical recommendations.
    
    ---
    
    ### Why Early Detection Matters
    - Reduces farm mortality and economic loss  
    - Prevents disease transmission across flocks  
    - Saves time compared to clinical testing  
    - Supports farmers with limited veterinary access  
    
    ---
    
    ### Who Can Benefit From This System?
    - Poultry farm owners  
    - Veterinary clinics  
    - Agricultural universities and researchers  
    - Animal husbandry departments  
    - Livestock consultants and field veterinarians  
    
    ---
    
    ### Our Vision
    To support poultry farmers with **affordable, accessible, and technology-driven disease monitoring**, improving the overall health, safety, and productivity of farms.
    
    Healthy poultry leads to **a healthier food chain and a stronger agricultural economy.**
    """)

st.sidebar.markdown("""
##  Smart Poultry AI System

An intelligent farm management tool powered by **Deep Learning** for fast and reliable chicken disease diagnosis.

---

###  Modules
- User Login / Registration  
- Disease Detection (Upload / Camera)  
- PDF Report Export  
- Telegram Disease Alerts  
- Veterinary Hospital Mapping  
- Admin Dashboard & Analytics

---

###  Secure Access
Only registered users can access:
- Disease Detection
- PDF Report
- Live Alerts

Admin can access:
- Analytics
- User Data Statistics

---
""")

# =====================================================
# INITIAL SESSION STATE SETUP
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "page" not in st.session_state:
    st.session_state.page = "Login"

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "pred_label" not in st.session_state:
    st.session_state.pred_label = None

if "pred_info" not in st.session_state:
    st.session_state.pred_info = None

if "pred_image_path" not in st.session_state:
    st.session_state.pred_image_path = None

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = os.getenv("MODEL_DATA_DIR", "models")
MODEL_PATH = os.path.join(DATA_DIR, "EfficientNetB3_best.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

ADMIN_USER = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "")

# =====================================================
# DATABASE
# =====================================================
def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            name TEXT,
            email TEXT,
            phone TEXT,
            whatsapp TEXT,
            farm TEXT,
            location TEXT,
            district TEXT,
            state TEXT,
            pincode TEXT,
            farm_type TEXT,
            birds TEXT,
            breed TEXT,
            established_year TEXT,
            password TEXT,
            reg_time TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def add_user(name, email, phone, whatsapp, farm, location, district, state,
             pincode, farm_type, birds, breed, established_year, password):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, email, phone, whatsapp, farm, location, district, state,
                 pincode, farm_type, birds, breed, established_year,
                 password, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def authenticate_user(name, password):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))
    data = cur.fetchone()
    conn.close()
    return data

def get_all_users():
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df

# =====================================================
# MODEL + TRANSFORMS
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@st.cache_resource
def load_model():
    model = models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(1536),
        nn.Linear(1536,256),
        nn.ReLU(),
        nn.Dropout(0.45),
        nn.Linear(256,4),
        nn.Softmax(dim=1)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def clean_text(text):
    # Replace characters unsupported by Latin-1 (FPDF1.x)
    replacements = {
        "\u26A0": "WARNING",  # ⚠
        "\u2714": "OK",  # ✓
        "\u279C": "->",  # ➜
        "\u2192": "->",  # →
        "\u2013": "-",  # –
        "\u2014": "-",  # —
        "\u2022": "-",  # •
        "\u00B0": "deg",  # °
        "\u201C": '"',
        "\u201D": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

model = load_model()

CLASS_NAMES = ['Coccidiosis', 'Healthy', 'NewCastle', 'Salmonella']

detailed_disease_info = {

  "Coccidiosis": {
    "Disease_Category": "Parasitic – Intestinal",
    "Reason": "Parasitic infection of the intestinal tract caused by Eimeria species protozoa",
    "Clinical_Symptoms": [
      "Diarrhea (watery, sometimes bloody)",
      "Lethargy and reduced activity",
      "Poor feed conversion and stunted growth",
      "Ruffled feathers and weight loss",
      "Dehydration and weakness"
    ],
    "Risk_Factors": [
      "High moisture and poor litter conditions",
      "Overcrowding and inadequate space",
      "Temperature fluctuations",
      "Young birds between 3 to 6 weeks old",
      "Contaminated feed or water"
    ],
    "Prevention": [
      "Maintain dry litter and clean bedding",
      "Provide good ventilation to reduce humidity",
      "Follow strict sanitation and hygiene protocols",
      "Avoid overstocking",
      "Disinfect poultry houses routinely",
      "Prevent fecal oral transmission",
      "Use coccidiostats in feed for prevention"
    ],
    "Medical_Drugs": {
      "Primary": [
        "Amprolium (75 to 100 mg per kg for 5 to 7 days)",
        "Toltrazuril (25 mg per kg single dose)",
        "Diclazuril (1 mg per kg for 2 days)"
      ],
      "Alternative": [
        "Sulphamethoxazole Trimethoprim combination",
        "Monensin (prophylactic feed additive)"
      ],
      "Administration": "Oral medication via drinking water or feed"
    },
    "Recovery_Time": "7 to 10 days with timely treatment",
    "Mortality_Rate": "Low to moderate if untreated",
    "Prognosis": "Excellent with early diagnosis and management",
    "Monitoring": "Observe for improvement within 48 to 72 hours",
    "Complications": [
      "Stunted growth",
      "Feed conversion inefficiency",
      "Secondary bacterial infections"
    ]
  },


  "Healthy": {
    "Disease_Category": "Normal physiological state",
    "Reason": "No pathogenic infection or disease condition detected",
    "Clinical_Status": [
      "Normal feeding, activity and alertness",
      "Clean and smooth feathers",
      "Clear eyes and nostrils",
      "Normal droppings and body condition",
      "Good weight for age and breed"
    ],
    "Health_Indicators": [
      "Normal respiratory rate",
      "Normal body temperature",
      "Strong posture and muscle tone",
      "No signs of pain or distress"
    ],
    "Prevention": [
      "Provide balanced diet and fresh water",
      "Maintain hygiene and sanitation",
      "Ensure proper ventilation",
      "Follow vaccination schedules",
      "Quarantine new birds before introduction",
      "Keep rodents and pests under control"
    ],
    "Medical_Drugs": "None required",
    "Supplementation": [
      "Vitamin and mineral additives",
      "Probiotics for gut health (optional)",
      "Electrolytes during stress periods"
    ],
    "Monitoring": "Continue routine health checks",
    "Lifespan_Expectancy": "Between 3 to 10 years depending on breed",
    "Production_Metrics": "Normal egg yield and feed conversion"
  },

  "NewCastle": {
    "Disease_Category": "Highly Contagious Viral – Respiratory and Neurological Disorder",
    "Reason": "Severe respiratory and neurological condition caused by Newcastle Disease Virus (NDV), known for rapid flock-to-flock transmission",
    "Clinical_Symptoms": [
      "Respiratory distress and harsh coughing",
      "Nasal discharge with frequent sneezing",
      "Facial and eyelid swelling with watery eyes",
      "Neurological tremors and twisted neck posture (torticollis)",
      "Paralysis of wings and legs in advanced stages",
      "Abrupt decline in egg production",
      "Soft-shelled or misshapen eggs",
      "Watery greenish diarrhea",
      "Sudden mortality in acute velogenic outbreaks"
    ],
    "Risk_Factors": [
      "Flocks without proper vaccination protocol",
      "Direct or indirect contact with infected wild birds or poultry",
      "Low biosecurity and movement of equipment/personnel between farms",
      "High stress situations including extreme temperature variation",
      "Weak or immunocompromised birds"
    ],
    "Prevention": [
      "Compulsory vaccination at 4–6 weeks and booster at 12–16 weeks",
      "Isolate new birds for 14–21 days before mixing with flock",
      "Restrict access of wild birds and farm visitors",
      "Use footbaths, vehicle disinfection, and controlled farm entry",
      "Safe disposal of carcasses and proper litter management"
    ],
    "Medical_Drugs": {
      "Treatment": "There are no direct antiviral medications available for Newcastle Disease",
      "Supportive_Therapy": [
        "Broad-spectrum antibiotics to prevent secondary bacterial infections",
        "Vitamin A, D3, E and selenium for immune and mucosal recovery",
        "Electrolytes and glucose supplementation to correct dehydration",
        "Anti-inflammatory agents for pain and inflammation where permitted",
        "Strict isolation and nursing of symptomatic birds"
      ],
      "Note": "Vaccination is the most effective intervention and should not be delayed in unaffected flocks"
    },
    "Recovery_Time": "Typically 2–3 weeks for mild/mesogenic strains; velogenic strains may result in sudden death",
    "Mortality_Rate": "10–100 percent depending on strain type and vaccination coverage",
    "Virus_Strains": [
      "Velogenic – highly pathogenic with rapid mortality",
      "Mesogenic – moderate respiratory and neurological signs",
      "Lentogenic – mild respiratory signs and low mortality"
    ],
    "Prognosis": "Favorable only in vaccinated flocks; poor prognosis in unvaccinated birds",
    "Transmission": "Aerosol inhalation, fecal shedding, contaminated feed/water, equipment and fomites",
    "Quarantine_Duration": "At least 3–4 weeks from last observed clinical case",
    "Monitoring": "Daily monitoring for respiratory distress, neurological signs and flock mortality trends"
  },

  "Salmonella": {
    "Disease_Category": "Bacterial – Systemic and Enteric",
    "Reason": "Bacterial infection caused by Salmonella species",
    "Clinical_Symptoms": [
      "Watery or greenish diarrhea",
      "Severe lethargy and depression",
      "Ruffled feathers and dehydration",
      "Fever and rapid weight loss",
      "Joint swelling in chronic cases",
      "Eye swelling and discharge",
      "Sudden death in acute phase"
    ],
    "Risk_Factors": [
      "Contaminated feed or drinking water",
      "Overcrowding and poor hygiene",
      "Inadequate pest control",
      "Stressful environmental conditions",
      "Weak or young birds are more susceptible"
    ],
    "Transmission": [
      "Fecal oral transmission",
      "Contaminated feed and water",
      "Vertical transmission from parents to chicks",
      "Human handling cross contamination"
    ],
    "Prevention": [
      "Strict hygiene and sanitation practices",
      "Use clean and reliable feed sources",
      "Ensure clean drinking water supply",
      "Quarantine new birds for 2 to 3 weeks",
      "Implement complete rodent and pest control"
    ],
    "Medical_Drugs": {
      "Primary": [
        "Enrofloxacin (10 mg per kg for 5 to 7 days)",
        "Ciprofloxacin (5 to 10 mg per kg for 5 to 7 days)"
      ],
      "Alternative": [
        "Sulphamethoxazole Trimethoprim (30 mg per kg for 5 to 7 days)",
        "Amoxicillin Clavulanic Acid (25 mg per kg for 5 to 7 days)",
        "Tetracyclines (15 to 20 mg per kg for 5 to 7 days)"
      ],
      "Administration": "Oral route with water or feed, or injectables when recommended",
      "Antibiotic_Sensitivity": "Culture and sensitivity testing before treatment is preferred",
      "Resistance_Note": "Avoid excessive and unnecessary use of antibiotics"
    },
    "Recovery_Time": "10 to 14 days with appropriate therapy",
    "Mortality_Rate": "Moderate in young birds if untreated",
    "Prognosis": "Good when diagnosed early",
    "Complications": [
      "Septicemia",
      "Arthritis",
      "Peritonitis",
      "Egg production failure"
    ],
    "Public_Health": "Zoonotic infection risk to humans",
    "Post_Treatment": "Continue monitoring for relapse and disinfect poultry house thoroughly"
  }

}

def convert_txt_to_pdf(txt_path, image_path, detected_label):
    pdf_path = txt_path.replace(".txt", ".pdf")
    pdf = FPDF()
    pdf.add_page()

    # ---------- STATIC TITLE ON TOP CENTER ----------
    pdf.set_font("Arial", "B", 20)
    pdf.ln(6)  # slight margin
    pdf.cell(0, 12, "POULTRY CLINICAL DIAGNOSTIC REPORT", ln=True, align="C")
    pdf.ln(4)

    # PAGE DIMENSIONS
    page_w = pdf.w
    page_h = pdf.h

    # IMAGE SIZE
    img_w = 130
    img_h = 130

    # CENTER POSITION FOR IMAGE
    x = (page_w - img_w) / 2
    y = (page_h - img_h) / 2 - 25  # moved up slightly because title is added

    # ADD IMAGE CENTERED
    try:
        pdf.image(image_path, x=x, y=y, w=img_w, h=img_h)
    except:
        pass

    # ------------------ DISEASE CLASS DISPLAY (BIG BOLD CENTER TEXT) ------------------
    pdf.set_xy(0, y + img_h + 10)
    pdf.set_font("Arial", "B", 28)

    disease = detected_label
    if disease == "Healthy":
        pdf.set_text_color(0, 150, 0)      # GREEN for Healthy
    else:
        pdf.set_text_color(200, 0, 0)      # RED for Diseases

    pdf.cell(0, 18, disease.upper(), align="C", ln=True)

    # RESET COLOR TO BLACK FOR NEXT PAGES
    pdf.set_text_color(0, 0, 0)

    # ------------------- PAGE-2 : FULL TEXT CONTENT -------------------
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_font("Arial", size=11)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            pdf.multi_cell(0, 6, clean_text(line))

    pdf.output(pdf_path)
    return pdf_path



# =====================================================
# PREDICT FUNCTION
# =====================================================
def predict(img_pil):
    t = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        _, pred = torch.max(out,1)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return CLASS_NAMES[pred.item()], probs


# =====================================================
# TELEGRAM ALERT
# =====================================================
def notify_telegram(path, label, info):
    # Fetch user details from DB
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name=?", (st.session_state.username,))
    user = cur.fetchone()
    conn.close()

    # Correct field mapping
    name = user[0]
    email = user[1]
    phone = user[2]
    whatsapp = user[3]
    farm = user[4]
    location = user[5]
    district = user[6]
    state = user[7]
    birds = user[10]
    breed = user[11]

    # Extract treatment safely
    treatment = info.get("Medical_Drugs", info.get("Medical Drugs", {}))
    if isinstance(treatment, dict):
        primary = treatment.get("Primary", [])
        alternative = treatment.get("Alternative", [])
        supportive = treatment.get("Supportive_Therapy", [])
        admin = treatment.get("Administration", treatment.get("Treatment", "Not specified"))

        treatment_summary = ""
        if primary:
            treatment_summary += "Primary:\n  - " + "\n  - ".join(primary) + "\n\n"
        if alternative:
            treatment_summary += "Alternative:\n  - " + "\n  - ".join(alternative) + "\n\n"
        if supportive:
            treatment_summary += "Supportive Therapy:\n  - " + "\n  - ".join(supportive) + "\n\n"
        treatment_summary += f"Administration: {admin}"
    else:
        treatment_summary = str(treatment)

    # Short caption (sendPhoto)
    short_caption = (
        f"Disease Detected: {label}\n"
        f"Farm: {farm}\n"
        f"Owner: {name}\n"
        f"Location: {location}, {district}, {state}"
    )

    # Full detailed report (sendMessage)
    full_message = (
        f"🔔 Chicken Disease Detection Alert\n"
        f"------------------------------------\n"
        f"Owner Name: {name}\n"
        f"Phone: {phone}\n"
        f"Farm Name: {farm}\n"
        f"Location: {location}, {district}, {state}\n"
        f"Birds Count: {birds}\n"
        f"Breed Type: {breed}\n"
        f"Email: {email}\n\n"
        f"Disease Detected: {label}\n"
        f"Reason: {info.get('Reason','Not Available')}\n\n"
        f"Symptoms:\n  - " + "\n  - ".join(info.get("Clinical_Symptoms", [])) + "\n\n"
        f"Treatment Plan:\n{treatment_summary}\n\n"
        f"Recovery Time: {info.get('Recovery_Time','Not Available')}\n"
        f"Prognosis: {info.get('Prognosis','Not Available')}\n\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Send alerts
    try:
        # Photo with caption
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            data={"chat_id": CHAT_ID, "caption": short_caption},
            files={"photo": open(path, "rb")}
        )

        # Full message
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": full_message}
        )

        st.success("Telegram Alert Sent Successfully")

    except Exception as e:
        st.error(f"Telegram Notification Failed: {e}")

HOSPITALS_DATA = [
    {
        "name": "Khushi Pet Care",
        "phone": "+91 95359 61590",
        "address": "IV Stage 3 Block, Rajarajeshwari Nagar, Bengaluru, Karnataka 560098",
        "rating": 4.5,
        "reviews": 230,
        "open_close": "9:30 AM – 9:00 PM (Daily)",
        "website": "https://khushi-petcare.com",
        "map_url": "https://maps.app.goo.gl/aoE41d9LbrsDoJvm9",
        "image": "https://lh3.googleusercontent.com/p/AF1QipO4Uc2JuVDc-55Pox-aL4zw1jyEwly_vHFD0dQg=w408-h544-k-no"
    },
    {
        "name": "BRIGADE PET CLINIC AND GROOMING CENTER",
        "phone": "+91 78999 45896",
        "address": "BHCS Layout, Subramanyapura, Bengaluru, Karnataka 560061",
        "rating": 3.8,
        "reviews": 24,
        "open_close": "10:00 AM – 9:00 PM (Closed Tuesday)",
        "website": "http://www.brigade.petlifeline.in",
        "map_url": "https://maps.app.goo.gl/NXqQLJqDouJxYdN86",
        "image": "https://lh3.googleusercontent.com/gps-cs-s/AG0ilSw1zhtx8BfGOv5Mb0MmvBRrJ8B3KO0uIGxWE2njzB_p_r-jdfDrYo9E46Xew8sAjQhvR6EPeDdgWVJVwoqK_jIM92L1bG8xtcjtkHpljqPGobXSlupJGwh-0pNVwBy9TFnQTbE-=w408-h306-k-no"
    },
    {
        "name": "PfA Pet Care & Clinic",
        "phone": "+91 95132 68686",
        "address": "Uttarahalli Main Rd, Next to BGS Hospitals, Kengeri, Bengaluru 560060",
        "rating": 4.9,
        "reviews": 131,
        "open_close": "9:00 AM – 7:30 PM (Daily)",
        "website": "https://www.pfapetclinic.com",
        "map_url": "https://maps.app.goo.gl/Yt2WmTCbTEfFYriK7",
        "image": "https://lh3.googleusercontent.com/p/AF1QipNWI4wlujD-nEPaIc_Mqn3Byx9w5wtkaH9gooPo=w408-h725-k-no"
    },
    {
        "name": "Pride Pets Health & Wellness Center",
        "phone": "+91 9731144556",
        "address": "Dr. Vishnu Vardhana Road, Rajarajeshwari Nagar, Bengaluru 560098",
        "rating": 5.0,
        "reviews": 33,
        "open_close": "10:00 AM – 8:30 PM",
        "website": "http://www.japethospital.com",
        "map_url": "https://maps.app.goo.gl/isS4ybn6zF9K8HFV6",
        "image": "https://lh3.googleusercontent.com/p/AF1QipOmI66ujS5EiBmc_Z_Wjz_VskJNuznbaBfrz-3d=w408-h306-k-no"
    },
]

def show_hospitals_static():
    st.subheader("Nearby Veterinary Hospitals")
    for h in HOSPITALS_DATA:
        st.markdown(f"### {h['name']}")
        if h.get("image"):
            st.image(h["image"], width=220)
        st.write(f"📍 Address: {h['address']}")
        st.write(f"📞 Phone: {h['phone']}")
        st.write(f"🕐 Hours: {h['open_close']}")
        st.write(f"⭐ Rating: {h['rating']} ({h['reviews']} reviews)")

        links = []
        if h.get("website"):
            links.append(f"[Website]({h['website']})")
        if h.get("map_url"):
            links.append(f"[Google Maps]({h['map_url']})")
        st.markdown(" • ".join(links))
        st.markdown("---")

# ===============================
#  GEMINI CONFIG (FIXED)
# ===============================
import google.generativeai as genai
import streamlit as st

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",   # ✅ FREE-TIER SUPPORTED
    system_instruction=(
        "You are a poultry livestock AI assistant. "
        "Answer ONLY questions related to chicken diseases, symptoms, prevention or treatment. "
        "If the question is NOT related to chicken disease, refuse politely. "
        "Always answer in 20–30 words only."
    )
)

# ===============================
#  SIDEBAR CHATBOT
# ===============================
st.sidebar.title("🐔 Poultry AI Assistant")
st.sidebar.markdown("Ask any question about **Chicken Diseases**")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.sidebar.text_input("Type your question:")

if st.sidebar.button("Ask"):
    if user_query.strip():
        try:
            if not GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY is not configured")
            response = gemini_model.generate_content(user_query)
            answer = response.text
        except Exception:
            answer = "⚠️ AI service temporarily unavailable. Please try again later."

        # Save conversation
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("AI", answer))

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.sidebar.markdown(f"**👤 You:** {msg}")
    else:
        st.sidebar.markdown(f"**🤖 AI:** {msg}")


# =====================================================
# PAGE HANDLERS
# =====================================================

# -------- LOGIN PAGE --------
def login_page():
    st.header("User Login")
    name = st.text_input("Name")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate_user(name, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = name
            st.session_state.page = "Detection"
        else:
            st.error("Invalid Credentials")


# -------- REGISTER PAGE --------
def register_page():
    st.image("Image/s4.png")
    st.header("Create Account")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    whatsapp = st.text_input("WhatsApp Number")
    farm = st.text_input("Farm Name")
    location = st.text_input("Farm Address / Location")
    district = st.text_input("District")
    state = st.text_input("State")
    pincode = st.text_input("Pincode")
    farm_type = st.selectbox("Farm Type", ["Broiler", "Layer", "Mixed", "Other"])
    birds = st.text_input("Number of Birds")
    breed = st.text_input("Breed Type")
    established_year = st.text_input("Year of Establishment")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        add_user(name, email, phone, whatsapp, farm, location, district, state,
                 pincode, farm_type, birds, breed, established_year, password)
        st.success("Registration Successful")
        st.session_state.page = "Login"


# -------- ADMIN PAGE --------
def admin_page():
    st.header("Admin Login")
    st.image("Image/admin.png")

    u = st.text_input("Admin Username")
    p = st.text_input("Admin Password", type="password")

    if st.button("Login"):
        if u == ADMIN_USER and p == ADMIN_PASS:
            st.session_state.admin_logged_in = True
        else:
            st.error("Invalid Admin Credentials")


def txt_to_pdf(txt_path):
    pdf_path = txt_path.replace(".txt", ".pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_font("Arial", size=12)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            pdf.multi_cell(0, 7, clean_text(line))   # clean_text() already defined
    pdf.output(pdf_path)
    return pdf_path


def export_report(image_path, label, info, user_details, probability):
    report_path = os.path.join(DATA_DIR, f"{label}_clinical_report.txt")

    severity = "Severe" if probability > 0.90 else "Moderate" if probability > 0.75 else "Mild"
    score = round(probability * 100, 2)

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("==============================================================\n")
        f.write("           POULTRY BIO-SECURITY RAPID RESPONSE REPORT\n")
        f.write("==============================================================\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("⚠ INFECTIOUS THREAT DETECTED\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Disease                   : {label}\n")
        f.write(f"Threat Level              : {severity}\n")
        f.write(f"Probability Match         : {score}%\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("FLOCK IDENTIFICATION\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Owner                     : {user_details.get('name', 'N/A')}\n")
        f.write(f"Phone                     : {user_details.get('phone', 'N/A')}\n")
        f.write(f"WhatsApp                  : {user_details.get('whatsapp', 'N/A')}\n")
        f.write(f"Farm                      : {user_details.get('farm', 'N/A')}\n")
        f.write(f"Location                  : {user_details.get('location', 'N/A')}, {user_details.get('district', 'N/A')}, {user_details.get('state', 'N/A')}\n")
        f.write(f"Pincode                   : {user_details.get('pincode', 'N/A')}\n")
        f.write(f"Population (Birds)        : {user_details.get('birds', 'N/A')}\n")
        f.write(f"Breed Type                : {user_details.get('breed', 'N/A')}\n")
        f.write(f"Established Year          : {user_details.get('established_year', 'N/A')}\n")
        f.write(f"Report Generated          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("INFECTION ANALYSIS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write("Cause:\n")
        f.write(f"> {info.get('Reason','Not Available')}\n\n")

        if "Clinical_Symptoms" in info:
            f.write("Observed Field Indicators:\n")
            for s in info["Clinical_Symptoms"]:
                f.write(f"• {s}\n")
            f.write("\n")

        if "Risk_Factors" in info:
            f.write("Environmental & Operational Risk Factors:\n")
            for s in info["Risk_Factors"]:
                f.write(f"• {s}\n")
            f.write("\n")

        f.write("--------------------------------------------------------------------------------\n")
        f.write("RESPONSE PROTOCOL (TREATMENT)\n")
        f.write("--------------------------------------------------------------------------------\n")

        treatment = info.get("Medical_Drugs", info.get("Medical Drugs", {}))
        if isinstance(treatment, dict):
            if "Primary" in treatment:
                f.write("Primary Response Agents:\n")
                for p in treatment["Primary"]:
                    f.write(f"  - {p}\n")
                f.write("\n")

            if "Supportive_Therapy" in treatment:
                f.write("Supportive Measures:\n")
                for p in treatment["Supportive_Therapy"]:
                    f.write(f"  - {p}\n")
                f.write("\n")

            if "Alternative" in treatment:
                f.write("Alternative Agents:\n")
                for p in treatment["Alternative"]:
                    f.write(f"  - {p}\n")
                f.write("\n")

            if "Administration" in treatment:
                f.write(f"Administration:\n  → {treatment['Administration']}\n\n")
            if "Treatment" in treatment:
                f.write(f"Additional Treatment Notes:\n  → {treatment['Treatment']}\n\n")

        f.write(f"Estimated Recovery                : {info.get('Recovery_Time','Not Available')}\n")
        f.write(f"Long-Term Flock Prognosis         : {info.get('Prognosis','Not Available')}\n")
        f.write("--------------------------------------------------------------------------------\n")

        f.write("MANDATORY NOTICE:\n")
        f.write("Immediate isolation advised for the affected flock segment.\n")
        f.write("Report applicable for biosecurity compliance audit.\n")
        f.write("=================================================================================\n")

    return report_path

# -------- DETECTION PAGE --------
def detection_page():
    st.title(f"Welcome, {st.session_state.username}")

    mode = st.radio("Choose Input", ["Upload Image","Use Camera"])

    # -------------------- UPLOAD IMAGE --------------------
    if mode == "Upload Image":
        file = st.file_uploader("Upload Chicken Image", type=["jpg","jpeg","png"])
        if file:
            img = Image.open(file).convert("RGB")
            st.session_state.uploaded_image = img
            st.image(img)

            label, probs = predict(img)
            st.session_state.pred_label = label
            st.session_state.pred_info = detailed_disease_info[label]

            temp_path = os.path.join(DATA_DIR, "uploaded.jpg")
            img.save(temp_path)
            st.session_state.pred_image_path = temp_path

            info = st.session_state.pred_info
            # Colored prediction display
            if label == "Healthy":
                st.markdown(
                    f"<h3 style='color:green; font-weight:bold;'>Detected Class: {label}</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h3 style='color:red; font-weight:bold;'>Detected Class: {label}</h3>",
                    unsafe_allow_html=True
                )

            st.subheader("Detailed Disease Information")

            for key, value in info.items():
                if isinstance(value, list):
                    st.markdown(f"### {key}")
                    for v in value:
                        st.markdown(f"- {v}")

                elif isinstance(value, dict):
                    st.markdown(f"### {key}")
                    for sub_key, sub_val in value.items():
                        st.markdown(f"**{sub_key}:**")
                        if isinstance(sub_val, list):
                            for item in sub_val:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(sub_val)

                else:
                    st.markdown(f"**{key}:** {value}")

            if st.button("Generate Report"):
                # probability score
                probability = max(probs)

                # fetch all registered info of the logged-in user
                conn = sqlite3.connect("users.db")
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE name=?", (st.session_state.username,))
                user = cur.fetchone()
                conn.close()

                # map DB fields to dictionary
                user_details = {
                    "name": user[0],
                    "email": user[1],
                    "phone": user[2],
                    "whatsapp": user[3],
                    "farm": user[4],
                    "location": user[5],
                    "district": user[6],
                    "state": user[7],
                    "pincode": user[8],
                    "farm_type": user[9],
                    "birds": user[10],
                    "breed": user[11],
                    "established_year": user[12]
                }

                # generate TXT then PDF
                txt = export_report(temp_path, label, st.session_state.pred_info, user_details, probability)
                pdf = convert_txt_to_pdf(txt, temp_path, st.session_state.pred_label)

                with open(pdf, "rb") as f:
                    st.download_button("Download Final PDF Report", f, file_name=f"{label}_clinical_report.pdf")

            # Auto Telegram Alert after prediction (only if disease is detected)
            # Manual Telegram Alert Button
            if label != "Healthy":
                if st.button("Send Telegram Alert"):
                    notify_telegram(
                        temp_path,
                        label,
                        st.session_state.pred_info
                    )

            if st.button("Show Nearby Hospitals"):
                show_hospitals_static()

    # -------------------- CAMERA MODE --------------------
    else:
        st.info("Camera preview opens in a CV2 window.")
        if st.button("Open Camera"):
            cap = cv2.VideoCapture(0)
            st.info("Press C to capture")
            while True:
                ret, frame = cap.read()
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    st.session_state.captured_image = frame
                    break
            cap.release()
            cv2.destroyAllWindows()

        if st.session_state.captured_image is not None:
            rgb = cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            st.image(img)

            label, probs = predict(img)
            st.session_state.pred_label = label
            st.session_state.pred_info = detailed_disease_info[label]

            out_path = os.path.join(DATA_DIR,"captured.jpg")
            img.save(out_path)
            st.session_state.pred_image_path = out_path

            # Colored prediction display
            if label == "Healthy":
                st.markdown(
                    f"<h3 style='color:green; font-weight:bold;'>Detected Class: {label}</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<h3 style='color:red; font-weight:bold;'>Detected Class: {label}</h3>",
                    unsafe_allow_html=True
                )

            # st.write(st.session_state.pred_info)

            # ==========================================================
            # SHOW FULL DETAILED DISEASE INFORMATION (EXPANDERS)
            # ==========================================================
            info = st.session_state.pred_info
            st.subheader("Detailed Disease Information")

            for key, value in info.items():
                if isinstance(value, list):
                    st.markdown(f"### {key}")
                    for v in value:
                        st.markdown(f"- {v}")

                elif isinstance(value, dict):
                    st.markdown(f"### {key}")
                    for sub_key, sub_val in value.items():
                        st.markdown(f"**{sub_key}:**")
                        if isinstance(sub_val, list):
                            for item in sub_val:
                                st.markdown(f"- {item}")
                        else:
                            st.markdown(sub_val)

                else:
                    st.markdown(f"**{key}:** {value}")


            if st.button("Generate Report"):
                # probability score
                probability = max(probs)

                # fetch all registered info of the logged-in user
                conn = sqlite3.connect("users.db")
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE name=?", (st.session_state.username,))
                user = cur.fetchone()
                conn.close()

                # map DB fields to dictionary
                user_details = {
                    "name": user[0],
                    "email": user[1],
                    "phone": user[2],
                    "whatsapp": user[3],
                    "farm": user[4],
                    "location": user[5],
                    "district": user[6],
                    "state": user[7],
                    "pincode": user[8],
                    "farm_type": user[9],
                    "birds": user[10],
                    "breed": user[11],
                    "established_year": user[12]
                }

                # generate TXT then PDF
                txt = export_report(out_path, label, st.session_state.pred_info, user_details, probability)
                pdf = convert_txt_to_pdf(txt, out_path, st.session_state.pred_label)

                with open(pdf, "rb") as f:
                    st.download_button("Download Final PDF Report", f, file_name=f"{label}_clinical_report.pdf")

            # Manual Telegram Alert Button for CAMERA MODE
            if label != "Healthy":
                if st.button("Send Telegram Alert", key="tg_camera"):
                    notify_telegram(
                        out_path,
                        label,
                        st.session_state.pred_info
                    )

            # Show Hospitals for CAMERA MODE
            if st.button("Show Nearby Hospitals", key="hospitals_camera"):
                show_hospitals_static()


# =====================================================
# SIDEBAR NAVIGATION WITH HOME PAGE
# =====================================================
if not st.session_state.logged_in and not st.session_state.admin_logged_in:
    # Before login -> user sees home + login + register + admin login
    menu_options = ["Home", "Login", "Register", "Admin Login"]
elif st.session_state.logged_in:
    # Normal user after login -> home + detection + logout
    menu_options = ["Home", "Detection", "Logout"]
elif st.session_state.admin_logged_in:
    # Admin after login -> home + admin dashboard + logout
    menu_options = ["Home", "Admin Dashboard", "Logout"]

menu = st.sidebar.radio("Navigation", menu_options)


# --------- Routing ---------
if menu == "Home":
    home_page()

elif menu == "Login":
    st.image("Image/d6.png")
    login_page()

elif menu == "Register":
    register_page()

elif menu == "Detection":
    if st.session_state.logged_in:
        detection_page()
    else:
        st.warning("Please login first")

elif menu == "Admin Login":
    admin_page()

elif menu == "Admin Dashboard":
    if st.session_state.admin_logged_in:
        st.header("Admin Data Analysis Dashboard")
        st.image("Image/EDA.png")

        data_paths = {
            "Registered Users": ("users.db", "users"),
        }
        eda = ExploratoryDataAnalysis(data_paths)
        eda.run()
    else:
        st.warning("Admin authentication required")


elif menu == "Logout":
    st.session_state.logged_in = False
    st.session_state.admin_logged_in = False
    st.session_state.username = None
    st.success("Logged out successfully")
