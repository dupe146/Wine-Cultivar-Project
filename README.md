# 🍷 Wine Cultivar Classification Project


## 📋 Project Overview

A machine learning web application that predicts wine cultivar origin based on 13 chemical properties using Support Vector Machine (SVM) algorithm.

**Live Demo:** https://winecultivar-project-jimoh-alabi.onrender.com/
---

## 🎯 Project Features

- **Algorithm:** Support Vector Machine (SVM) with RBF kernel
- **Accuracy:** 97-99%
- **Dataset:** UCI Wine Dataset (178 samples, 13 features, 3 classes)
- **Model Persistence:** Python Pickle format
- **Web Framework:** Flask
- **Deployment:** Render (Cloud Platform)

---

## 📁 Project Structure

```
WineCultivar_Project_Jimoh-Alabi_Islamiat_Modupe_250000033/

├── app.py                              # Flask web application
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── model/
│   ├── model_building.ipynb           # Jupyter notebook (training process)
│   └── wine_cultivar_model.pkl        # Trained SVM model (pickle format)
├── static/
│   └── style.css                      # CSS styling (optional)
└── templates/
    └── index.html                     # Web interface HTML
```

---

## 🚀 Setup & Installation

### **Prerequisites**
- Python 3.9+
- pip package manager

### **Local Setup**

1. **Clone the repository:**
```bash
git clone https://github.com/dupe146/Wine-Cultivar-Project.git
cd Wine-Cultivar-Project
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python app.py
```

5. **Open browser:**
```
http://localhost:5000
```

---

## 📊 Model Details

### **Algorithm Specifications**
- **Model Type:** Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **Implementation:** scikit-learn SVC
- **Preprocessing:** StandardScaler normalization
- **Train/Test Split:** 80/20

### **Dataset Information**
- **Source:** UCI Machine Learning Repository (sklearn.datasets.load_wine)
- **Samples:** 178 total
- **Features:** 13 chemical properties
  1. Alcohol
  2. Malic acid
  3. Ash
  4. Alcalinity of ash
  5. Magnesium
  6. Total phenols
  7. Flavanoids
  8. Nonflavanoid phenols
  9. Proanthocyanins
  10. Color intensity
  11. Hue
  12. OD280/OD315 of diluted wines
  13. Proline

- **Target Classes:** 3 wine cultivars (Class 0, Class 1, Class 2)
- **Class Distribution:** Balanced (59, 71, 48 samples)

### **Model Performance**
- **Accuracy:** ~97-99%
- **Precision:** High across all classes
- **Recall:** High across all classes
- **F1-Score:** Excellent performance

### **Model File**
- **Location:** `model/wine_cultivar_model.pkl`
- **Format:** Python Pickle
- **Size:** ~5-10 KB
- **Contents:** 
  - Trained SVM model
  - StandardScaler (fitted)
  - Feature names
  - Target class names
  - Model metadata

---

## 🌐 Deployment to Render

### **Deployment URL**
```
https://winecultivar-project-jimoh-alabi.onrender.com/
```

### **Deployment Configuration**

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
```

### **Environment Settings**
- **Runtime:** Python 3.11.0
- **Region:** [Your selected region]
- **Instance Type:** Free tier
- **Auto-Deploy:** Enabled (on git push)

### **Required Files for Deployment**
1. ✅ `app.py` - Flask application
2. ✅ `requirements.txt` - Dependencies
3. ✅ `model/wine_cultivar_model.pkl` - Trained model
4. ✅ `templates/index.html` - Web interface
5. ✅ `runtime.txt` - Python version specification

### **Render Settings (Dashboard)**
```
Service Type: Web Service
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
```

---

## 🖥️ Running the Application

### **Local Development**
```bash
# Activate virtual environment
source venv/bin/activate

# Run Flask app
python app.py

# Access at http://localhost:5000
```

### **Testing the API**

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Make Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [13.2, 2.77, 2.51, 18.5, 96.0, 1.09, 0.52, 0.86, 1.69, 5.8, 0.48, 1.03, 415.0]
  }'
```

---

## 📦 Dependencies

**Core Libraries:**
- Flask==3.0.0 (Web framework)
- scikit-learn>=1.3.0 (Machine learning)
- numpy>=1.24.0 (Numerical computing)
- gunicorn==21.2.0 (Production server)

**Optional:**
- flask-cors>=4.0.0 (CORS support)

---
---

## 📝 Model Training Process

The complete model training process is documented in:
```
model/model_building.ipynb
```

**Steps:**
1. Data loading and exploration
2. Exploratory Data Analysis (EDA)
3. Data preprocessing (scaling)
4. Model training (SVM with RBF kernel)
5. Model evaluation
6. Model persistence (pickle format)

---

## 🔧 Troubleshooting

### **Common Issues**

**Issue 1: Module not found errors**
```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**Issue 2: Port already in use**
```bash
# Solution: Change port in app.py or kill process
lsof -ti:5000 | xargs kill -9
```

**Issue 3: Model file not found**
```bash
# Solution: Ensure model file exists
ls -la model/wine_cultivar_model.pkl
```

**Issue 4: Render deployment fails**
```bash
# Solution: Check these
1. Verify requirements.txt has all dependencies
2. Ensure Start Command is correct
3. Check Render build logs for errors
4. Verify model file is in repository
```

---

## 🌟 Features

### **Web Interface**
- ✅ User-friendly input form
- ✅ Real-time predictions
- ✅ Confidence scores
- ✅ Responsive design
- ✅ Professional styling

### **API Endpoints**
- `GET /` - Homepage with input form
- `POST /predict` - Make wine cultivar predictions
- `GET /health` - Health check endpoint
- `GET /api/info` - Model information

---

## 📊 API Documentation

### **Predict Endpoint**

**URL:** `/predict`  
**Method:** `POST`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "features": [13.2, 2.77, 2.51, 18.5, 96.0, 1.09, 0.52, 0.86, 1.69, 5.8, 0.48, 1.03, 415.0]
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 0,
  "class_name": "Cultivar 0",
  "confidence": 0.95,
  "probabilities": [0.95, 0.03, 0.02]
}
```

---

## 🎯 Project Artifacts

All project artifacts are located in their respective folders:

- **Model artifacts:** `model/` directory
  - Training notebook: `model_building.ipynb`
  - Trained model: `wine_cultivar_model.pkl`

- **Web artifacts:** `templates/` and `static/` directories
  - HTML interface: `templates/index.html`
  - CSS styling: `static/style.css`

- **Application artifacts:** Root directory
  - Flask app: `app.py`
  - Dependencies: `requirements.txt`

---

---

## 📄 License

This project is submitted as part of academic coursework for Masters in Bioinformatics program.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Wine dataset
- scikit-learn developers for the ML library
- Flask team for the web framework
- Render for cloud hosting platform

---
