from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import uuid
from PIL import Image
import cv2  # Add OpenCV for video processing
import numpy as np
import subprocess


app = Flask(__name__)

# Configuration - ADD VIDEO EXTENSIONS
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # Increase to 100MB for videos
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "mp4",
    "avi",
    "mov",
    "mkv",
}  # Add video formats

# Create folders if they don't exist
for folder in [app.config["UPLOAD_FOLDER"], app.config["RESULT_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)

# Load your trained model
model = YOLO("best.pt")  # Your model file


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image_file(filename):
    image_extensions = {"png", "jpg", "jpeg", "gif", "bmp"}
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    return ext in image_extensions


def is_video_file(filename):
    video_extensions = {"mp4", "avi", "mov", "mkv"}
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    return ext in video_extensions


@app.route("/")
def index():
    """Main page with upload interface"""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle image OR video upload and run detection"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Generate unique filenames
    file_id = str(uuid.uuid4())[:8]
    original_filename = secure_filename(file.filename)
    upload_filename = f"{file_id}_{original_filename}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)

    # Save uploaded file
    file.save(upload_path)

    # Determine if it's image or video
    if is_image_file(original_filename):
        return process_image(upload_path, file_id, original_filename, upload_filename)
    elif is_video_file(original_filename):
        return process_video(upload_path, file_id, original_filename, upload_filename)
    else:
        return jsonify({"error": "Unsupported file type"}), 400


def process_image(upload_path, file_id, original_filename, upload_filename):
    """Process image file"""
    # Run YOLOv8 inference
    results = model.predict(upload_path, conf=0.25, save=False)

    # Process results
    result_filename = f"result_{file_id}_{original_filename}"
    result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)

    # Save annotated image
    plotted_image = results[0].plot(line_width=2, font_size=10)
    Image.fromarray(plotted_image[..., ::-1]).save(result_path)

    # Extract detection data
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        bbox = box.xyxy[0].tolist()

        detections.append(
            {
                "class": class_name,
                "confidence": round(confidence * 100, 2),
                "bbox": [round(x, 2) for x in bbox],
                "class_id": class_id,
            }
        )

    # Count statistics
    stats = {
        "helmet": len([d for d in detections if d["class"] == "Helmet"]),
        "no_helmet": len([d for d in detections if d["class"] == "NoHelmet"]),
        "total": len(detections),
    }

    return jsonify(
        {
            "success": True,
            "file_type": "image",
            "original_file": f"/static/uploads/{upload_filename}",
            "result_file": f"/static/results/{result_filename}",
            "detections": detections,
            "stats": stats,
        }
    )


def process_video(upload_path, file_id, original_filename, upload_filename):
    """Process video file"""
    # Create a unique subfolder for this video to avoid filename conflicts
    video_output_dir = os.path.join(app.config["RESULT_FOLDER"], f"video_{file_id}")
    os.makedirs(video_output_dir, exist_ok=True)

    # ===============================
    # Convert ORIGINAL uploaded video to browser-friendly MP4
    # ===============================
    original_play_filename = f"orig_{file_id}_{os.path.splitext(original_filename)[0]}.mp4"
    original_play_path = os.path.join(app.config["UPLOAD_FOLDER"], original_play_filename)

    cmd_orig = [
        "ffmpeg", "-y",
        "-i", upload_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        original_play_path
    ]

    try:
        subprocess.run(cmd_orig, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Failed to convert original video for browser playback",
            "details": e.stderr[-500:]
        }), 500


    results = model.predict(
        source=upload_path,
        conf=0.25,
        save=True,
        verbose=False,
    )


    print("Ultralytics save_dir:", getattr(results[0], "save_dir", None))


    # We will always output MP4 (browser friendly)
    result_filename = f"result_{file_id}_{os.path.splitext(original_filename)[0]}.mp4"
    final_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)

    # Find YOLO's saved video inside video_output_dir
    # Ultralytics actual folder where it saved outputs
    save_dir = str(getattr(results[0], "save_dir", video_output_dir))
    print("Using save_dir:", save_dir)

    # Search for the saved video file in save_dir (and subfolders just in case)
    yolo_saved_video = None
    for root, _, files in os.walk(save_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                yolo_saved_video = os.path.join(root, f)
                break
        if yolo_saved_video:
            break

    if yolo_saved_video is None:
        # Print directory listing for debugging
        print("No video found. Files in save_dir:", os.listdir(save_dir) if os.path.exists(save_dir) else "save_dir not found")
        return jsonify({"error": "YOLO did not produce an output video file"}), 500


    # Convert YOLO output to H.264 mp4 (browser-playable)
    cmd = [
        "ffmpeg", "-y",
        "-i", yolo_saved_video,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        final_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "FFmpeg conversion failed", "details": e.stderr[-500:]}), 500


    # Cleanup YOLO output folder
    try:
        for f in os.listdir(video_output_dir):
            try:
                os.remove(os.path.join(video_output_dir, f))
            except:
                pass
        os.rmdir(video_output_dir)
    except:
        pass


    # Calculate video statistics
    cap = cv2.VideoCapture(upload_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Get detections from first frame for demo
    frame_detections = []
    if results and len(results) > 0:
        for i, r in enumerate(results):
            if i >= 5:  # Check first 5 frames
                break
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                frame_detections.append(class_name)

    helmet_count = frame_detections.count("Helmet")
    no_helmet_count = frame_detections.count("NoHelmet")

    return jsonify(
        {
            "success": True,
            "file_type": "video",
            "original_file": f"/static/uploads/{original_play_filename}",
            "result_file": f"/static/results/{result_filename}",
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
                "resolution": f"{width}x{height}",
                "sample_detections": len(frame_detections),
            },
            "stats": {
                "helmet": helmet_count,
                "no_helmet": no_helmet_count,
                "total": len(frame_detections),
            },
            "message": f"Video processed! {total_frames} frames at {fps}FPS.",
        }
    )


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
