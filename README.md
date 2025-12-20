# 🪖 Helmet Detection System (YOLOv8 + Flask + Docker)

A Computer Vision–based Helmet Detection System built using YOLOv8 and deployed as a Flask API, fully containerized with Docker.
The system detects whether motorcyclists are wearing helmets in images and videos.

## 📌 Features

✅ Helmet detection using YOLOv8

✅ Flask-based REST API

✅ Supports image and video uploads

✅ Dockerized for easy deployment

✅ Stores prediction results automatically

✅ Lightweight and deployment-ready

## 🧠 Model

- Model: YOLOv8 (Ultralytics)

- Weights: best.pt (custom trained)

- Classes: Helmet / No Helmet

- Framework: PyTorch

## 🗂️ Project Structure

```bash
helmet_docker/
│
├── app.py # Main Flask application
├── app_test.py # Test script for the API
├── best.pt # Trained YOLOv8 model weights
├── Dockerfile # Docker configuration
├── requirements.txt # Python dependencies
├── .dockerignore
│
├── templates/ # HTML templates
├── static/
│ ├── uploads/ # Uploaded images
│ ├── video_uploads/ # Uploaded videos
│ ├── results/ # Image detection results
│ ├── video_results/ # Video detection results
│ └── favicon.ico
│
├── runs/
│ └── detect/ # YOLOv8 prediction outputs
│
└── README.md
```

## ⚙️ Installation (Without Docker)

1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/helmet-detection.git
cd helmet_docker
```

2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```

3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

4️⃣ Run the Application

```bash
python app.py
```

📍 App will run at:

```bash
http://127.0.0.1:5000
```

## 🐳 Docker Setup (Recommended)

1️⃣ Build Docker Image

```bash
docker build -t helmet-detection .
```

2️⃣ Run Docker Container

```bash
docker run -p 5000:5000 helmet-detection
```

📍 Access the app at:

```bash
http://localhost:5000
```

## 📤 API Usage

🔹 Image Upload

- Upload an image through the web interface

- Detection results are saved in:

```swift
static/results/
```

🔹 Video Upload

- Upload a video file

- Processed videos are saved in:

```swift
static/video_results/
```

## 🧪 Testing

Use app_test.py to test API endpoints locally.

python app_test.py

## 🚀 Deployment Notes

Dockerized → Easy deployment on:

- AWS EC2

- Azure VM

- Google Cloud

Can be extended with:

- RTSP / CCTV streams

- FastAPI

- Database logging

- Cloud storage

## 📚 Technologies Used

- Python

- Flask

- YOLOv8 (Ultralytics)

- OpenCV

- PyTorch

- Docker

- HTML / CSS
