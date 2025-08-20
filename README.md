# Fabric_matcher
# 🧵 Fabric Image Matcher

A Fabric Image Matching Web App using **Flask**, **TensorFlow (MobileNetV2)**, and **SQLite**.

## ✅ Project Overview

This project allows you to:
- Upload fabric images via an API or web UI.
- Store deep feature vectors in a SQLite database.
- Retrieve similar fabric images based on cosine similarity.

---

## 📂 Folder Structure

```
fabric-image-matcher/
│
├── api.py                     # Flask app (API endpoints)
├── fabric_matcher.py          # Feature extraction and matching logic
├── templates/
│   └── index.html             # (Create this) Web UI for testing
├── fabrics/                   # Fabric image dataset (required)
├── database.db                # SQLite DB (auto-generated)
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation (this file)
```

---

## 🧰 Requirements

Create a file named `requirements.txt` with the following content:

```bash
flask
flask-cors
tensorflow
numpy
Pillow
scipy
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📸 Required Folder

You must create a folder named `fabrics/` in the root directory and add sample fabric images inside it.

**Example:**

```
fabric-image-matcher/
└── fabrics/
    ├── fabric1.jpg
    ├── fabric2.png
    └── fabric3.jpeg
```

---

## 🚀 How to Run

### Step 1: Populate Database

```bash
python fabric_matcher.py
```

This initializes `database.db` and stores image features from the `fabrics/` folder.

### Step 2: Start the API

```bash
python api.py
```

Flask API runs at: [http://localhost:5000](http://localhost:5000)

---

## 🌐 API Endpoints

### `GET /api/`
Check API status.

### `POST /api/upload`
Upload a new image and store its features.

```bash
curl -X POST -F "file=@path_to_image.jpg" http://localhost:5000/api/upload
```

### `POST /api/match`
Find matching images based on similarity.

```bash
curl -X POST -F "file=@path_to_image.jpg" http://localhost:5000/api/match
```

---

## 🖼️ Optional: Web UI

Create `templates/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
  <title>Fabric Matcher</title>
</head>
<body>
  <h1>Upload Fabric Image</h1>
  <form action="/api/match" method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required>
    <input type="submit" value="Match">
  </form>
</body>
</html>
```

Access via: `http://localhost:5000`

---

## 💡 Example Match Response

```json
{
  "matches": [
    {
      "image_path": "fabrics/fabric1.jpg",
      "similarity": 0.92
    }
  ]
}
```

---

## 🧠 Model Used

- MobileNetV2 (pre-trained on ImageNet)
- Used for feature extraction (`include_top=False`, `pooling='avg'`)

---

## 🔁 Resetting the System

To reset and reprocess images:
1. Delete `database.db`
2. Run:

```bash
python fabric_matcher.py
```

---

