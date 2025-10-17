from flask import Flask, request, jsonify, send_file
import base64
from io import BytesIO
from PIL import Image
import os
import time

app = Flask(__name__)

# saved images
SAVE_DIR = "/home/cgthesis/Desktop/thesis/cpe180images"
SAVE_PATH = os.path.join(SAVE_DIR, "received_image.png")

@app.route("/process", methods=["POST"])
def process_image():
    try:
        data = request.get_json()
        img_base64 = data.get("image", None)
        if not img_base64:
            return jsonify({"success": False, "error": "No image provided"}), 400

        # Ensure directory exists
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Handle base64 with or without header
        if img_base64.startswith("data:image"):
            img_base64 = img_base64.split(",")[1]

        # Decode and convert image
        image_data = base64.b64decode(img_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Save explicitly as PNG to avoid unknown extension issues
        image.save(SAVE_PATH, format="PNG")

        # Log success
        print("Image received and saved at:", SAVE_PATH)
        print("Image format:", image.format or "PNG")
        print("Image size:", image.size)

        # Optionally open the image on the Pi desktop (GUI only)
        try:
            os.system(f"xdg-open {SAVE_PATH} >/dev/null 2>&1 &")
        except Exception:
            pass

        result = {
            "success": True,
            "message": f"Image successfully received and saved at {SAVE_PATH}",
        }

        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Run on all interfaces so PC can connect
    app.run(host="0.0.0.0", port=8001)
