# 🐔 AI-Powered Chicken Disease Detection & Veterinary Assistance System

An intelligent poultry health monitoring application that uses **Deep Learning and Computer Vision** to classify chicken diseases from images and support farmers with disease information, clinical reports, veterinary assistance, and farm analytics.

---

## 📌 Project Overview

Poultry diseases can spread rapidly and cause significant economic losses. Early identification and timely veterinary attention are essential for maintaining flock health.

This project provides an AI-powered solution that analyzes chicken images and predicts whether a bird is affected by one of four conditions:

- **Coccidiosis**
- **Newcastle Disease**
- **Salmonella**
- **Healthy**

Built using **Python, PyTorch, EfficientNetB3, and Streamlit**, the application combines image-based disease classification with practical tools to support poultry farmers and veterinary professionals.

## ✨ Key Features

### 🧠 AI-Based Disease Detection
- Deep learning model based on **EfficientNetB3**.
- Upload chicken images for disease prediction.
- Live camera input for image-based detection.
- Classification across four categories.
- Displays prediction results and associated disease information.

### 🩺 Clinical Information & Recommendations
- Disease causes and clinical symptoms.
- Risk factors and prevention measures.
- Medical information and recovery guidance.
- Supports informed discussions with veterinary professionals.

### 📄 Clinical PDF Reports
- Generate downloadable clinical reports.
- Present prediction results and relevant disease information.
- Maintain a report of detection outcomes.

### 📍 Veterinary Assistance
- Veterinary hospital mapping and location assistance.
- Helps users identify veterinary support.

### 🔔 Telegram Notifications
- Telegram-based disease alerts.
- Supports notification delivery through bot integration.

### 📊 Admin Dashboard & Analytics
- User registration and login.
- Admin access and user statistics.
- Exploratory data analysis and farm-related insights.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Web Application | Streamlit |
| Deep Learning Framework | PyTorch |
| Model Architecture | EfficientNetB3 |
| Image Processing | OpenCV, Pillow |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Database | SQLite |
| PDF Generation | FPDF |
| Notifications | Telegram Bot API |
| Additional AI Integration | Google Generative AI |

---

## 🧠 Model Information

The project uses **EfficientNetB3**, a convolutional neural network architecture for image classification.

**Model configuration:**
- Architecture: EfficientNetB3
- Input image size: 224 × 224 pixels
- Number of output classes: 4
- Output categories: Coccidiosis, Healthy, Newcastle, Salmonella
- Framework: PyTorch

The model loads trained weights from a `.pth` file and uses image preprocessing before generating predictions.

---

## 📂 Project Structure

```text
Chicken-Disease-Detection/
│
├── app.py
├── predict.py
├── test.py
├── EDA.py
├── Dataset.ipynb
├── get_location.html
│
├── Image/
│   ├── coverpage.png
│   └── s1.png
│
├── models/
│   └── EfficientNetB3_best.pth
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

*Note: The structure above is illustrative. Adjust folder and file names to match the files actually included in your repository.*

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the project directory

```bash
cd YOUR-REPOSITORY
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file using `.env.example` as a template.

Add your own API credentials and configuration values locally. Never upload your actual `.env` file to GitHub.

### 6. Add the trained model

Place the trained EfficientNetB3 weights in the model directory and update the model path in the application configuration if required.

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔐 Security & Configuration

Sensitive credentials should never be committed to a public repository.

- Store API keys and bot tokens in environment variables.
- Keep `.env` in `.gitignore`.
- Upload `.env.example` with placeholder values only.
- Do not upload personal user databases or private clinical reports.
- Use secure password hashing for user authentication.
- Configure admin credentials securely rather than hardcoding them.

---

## ⚠️ Important Disclaimer

This application is an AI-based decision-support and educational tool. Its predictions are not a substitute for laboratory testing, professional veterinary diagnosis, or treatment advice.

Always consult a qualified veterinarian before administering medication or making treatment decisions.

---

## 🚀 Future Improvements

- Improve classification performance with larger and more diverse datasets.
- Add model evaluation metrics and confusion matrix visualizations.
- Improve user authentication and data security.
- Deploy the application to a cloud platform.
- Add flock health monitoring and historical prediction analytics.
- Expand disease categories and support additional poultry conditions.

---

## 👩‍💻 Project Purpose

The goal of this project is to combine Artificial Intelligence, Deep Learning, and veterinary assistance into a practical tool that supports early poultry disease identification and better farm health management.

**Smarter Detection. Healthier Poultry. Better Farming.**

---

## 📜 License

Add your preferred open-source license before distributing this project. If you choose MIT, include a `LICENSE` file containing the MIT License text.
