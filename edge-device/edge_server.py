import mindspore as ms
from mindspore import nn, ops, Tensor
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# ========== Logging Setup ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiamondClassifier(nn.Cell):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, pad_mode='pad', padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, pad_mode='pad', padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, pad_mode='pad', padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, pad_mode='pad', padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(512 * 14 * 14, 1024)
        self.fc2 = nn.Dense(1024, 512)
        self.fc3 = nn.Dense(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def construct(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DiamondClassificationSystem:

    def __init__(self, checkpoint_dir="./checkpoints"):
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.models = {}
        self.class_definitions = {}
        self._load_models()

    def _load_models(self):
        for model_type in ["cut", "shape", "color"]:
            ckpt_path = self.checkpoint_dir / f"{model_type}_model_best.ckpt"
            class_file = self.checkpoint_dir / f"{model_type}_classes.txt"

            # Load class labels
            if class_file.exists():
                with open(class_file, "r") as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                self.class_definitions[model_type] = classes
                logger.info(f"Loaded {model_type} classes: {classes}")
            else:
                logger.error(f"Missing class file: {class_file}")
                continue

            if not ckpt_path.exists():
                logger.error(f"Missing checkpoint: {ckpt_path}")
                continue

            # Initialize model
            model = DiamondClassifier(num_classes=len(classes))

            try:
                param_dict = ms.load_checkpoint(str(ckpt_path))
                ms.load_param_into_net(model, param_dict)

                model.set_train(False)  # disable dropout + BN update

                self.models[model_type] = model
                logger.info(f"Loaded model: {model_type}")

            except Exception as e:
                logger.error(f"Failed loading {model_type}: {e}")

    def preprocess_image(self, image_data):

        # Handle base64
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)

        # Load image
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to 224x224 (same as training)
        image = image.resize((224, 224), Image.BILINEAR)

        # Convert to numpy array (0–1)
        img_array = np.array(image).astype(np.float32) / 255.0

        # Normalize exactly like training
        mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        std  = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
        img_array = (img_array - mean) / std

        # HWC → CHW
        img_array = np.transpose(img_array, (2,0,1))

        # Add batch dimension
        img_array = np.expand_dims(img_array, 0)

        return Tensor(img_array, ms.float32)

    def predict(self, img_data):

        tensor = self.preprocess_image(img_data)
        results = {}

        for mtype, model in self.models.items():

            logits = model(tensor)
            probs = ops.softmax(logits, axis=1).asnumpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_class = self.class_definitions[mtype][pred_idx]
            conf = float(probs[pred_idx]) * 100

            top3_idx = probs.argsort()[::-1][:3]
            top3 = [
                {
                    "class": self.class_definitions[mtype][i],
                    "confidence": float(probs[i]) * 100
                }
                for i in top3_idx
            ]

            results[mtype] = {
                "prediction": pred_class,
                "confidence": conf,
                "top_3": top3
            }

        return results


app = Flask(__name__)
CORS(app)

system = DiamondClassificationSystem("./checkpoints")


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"success": False, "error": "No image provided"})

    results = system.predict(data["image"])

    confidences = [results[m]["confidence"] for m in results]
    overall = sum(confidences) / len(confidences)

    return jsonify({
        "success": True,
        "cut": results["cut"],
        "shape": results["shape"],
        "color": results["color"],
        "overall_confidence": overall
    })


if __name__ == "__main__":
    print("\n💎 Edge Server Started @ http://127.0.0.1:8001\n")
    app.run(host="127.0.0.1", port=8001, debug=False)
