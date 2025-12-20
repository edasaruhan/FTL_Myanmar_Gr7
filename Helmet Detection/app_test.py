# app_test.py - Test Flask without model
from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    # Mock response for testing
    return jsonify(
        {
            "success": True,
            "original_image": "/static/test/original.jpg",
            "result_image": "/static/test/result.jpg",
            "detections": [
                {"class": "helmet", "confidence": 95.5, "bbox": [100, 150, 200, 250]},
                {
                    "class": "no_helmet",
                    "confidence": 87.2,
                    "bbox": [300, 200, 400, 300],
                },
            ],
            "stats": {"helmet": 1, "no_helmet": 1, "total": 2},
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
