from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from typing import Dict, Optional
from pathlib import Path
import logging

# MindSpore imports
try:
    import mindspore as ms
    from mindspore import nn, Tensor, load_checkpoint
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    logging.warning("MindSpore not available, using dummy models")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoGemGrade Cloud Server")

class AnalysisRequest(BaseModel):
    image: str  # Base64 encoded
    edge_metadata: Optional[Dict] = None

class DiamondClassifier(nn.Cell):
    """MindSpore model for diamond classification"""
    def __init__(self, num_classes):
        super(DiamondClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, pad_mode='pad', padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, pad_mode='pad', padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, pad_mode='pad', padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(128 * 28 * 28, 512)
        self.fc2 = nn.Dense(512, 256)
        self.fc3 = nn.Dense(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=0.5)
    
    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class DiamondGradingModels:
    """Container for all diamond grading models"""
    
    # Diamond classification categories
    CUT_CLASSES = ['Ideal', 'Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
    SHAPE_CLASSES = ['Round', 'Princess', 'Emerald', 'Asscher', 'Oval', 
                     'Radiant', 'Cushion', 'Marquise', 'Pear', 'Heart']
    COLOR_CLASSES = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.models_loaded = False
        
        if MINDSPORE_AVAILABLE:
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
            
            # Initialize model architectures
            self.cut_model = DiamondClassifier(len(self.CUT_CLASSES))
            self.shape_model = DiamondClassifier(len(self.SHAPE_CLASSES))
            self.color_model = DiamondClassifier(len(self.COLOR_CLASSES))
            
            # Load pre-trained weights from .ckpt files
            self.load_pretrained_models()
            
            logger.info("MindSpore models initialized")
        else:
            logger.warning("Using dummy models - MindSpore not available")
    
    def load_pretrained_models(self):
        """Load pre-trained model checkpoints uploaded from local training"""
        try:
            models_found = 0
            
            # Load cut model
            cut_path = self.checkpoint_dir / "cut_model_best.ckpt"
            if cut_path.exists():
                param_dict = load_checkpoint(str(cut_path))
                ms.load_param_into_net(self.cut_model, param_dict)
                logger.info(f"✓ Loaded cut model from {cut_path}")
                models_found += 1
            else:
                logger.warning(f"✗ Cut model not found at {cut_path} - using untrained model")
            
            # Load shape model
            shape_path = self.checkpoint_dir / "shape_model_best.ckpt"
            if shape_path.exists():
                param_dict = load_checkpoint(str(shape_path))
                ms.load_param_into_net(self.shape_model, param_dict)
                logger.info(f"✓ Loaded shape model from {shape_path}")
                models_found += 1
            else:
                logger.warning(f"✗ Shape model not found at {shape_path} - using untrained model")
            
            # Load color model
            color_path = self.checkpoint_dir / "color_model_best.ckpt"
            if color_path.exists():
                param_dict = load_checkpoint(str(color_path))
                ms.load_param_into_net(self.color_model, param_dict)
                logger.info(f"✓ Loaded color model from {color_path}")
                models_found += 1
            else:
                logger.warning(f"✗ Color model not found at {color_path} - using untrained model")
            
            if models_found == 3:
                self.models_loaded = True
                logger.info("✓ All models loaded successfully!")
            elif models_found > 0:
                logger.warning(f"Only {models_found}/3 models loaded")
            else:
                logger.error("No models found! Please upload .ckpt files to checkpoints/")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error("Models will use random initialization - results will be inaccurate!")
    
    def preprocess_for_model(self, img):
        """Preprocess image for model input"""
        # Ensure correct shape (224, 224, 3)
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to CHW format (MindSpore expects channels first)
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_cut(self, img):
        """Predict diamond cut quality"""
        if not MINDSPORE_AVAILABLE:
            return self._dummy_prediction(self.CUT_CLASSES)
        
        try:
            # Set to evaluation mode
            self.cut_model.set_train(False)
            
            img_tensor = Tensor(img, ms.float32)
            output = self.cut_model(img_tensor)
            probs = nn.Softmax()(output).asnumpy()[0]
            
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx] * 100)
            
            return {
                "prediction": self.CUT_CLASSES[pred_idx],
                "confidence": confidence,
                "all_probabilities": {
                    self.CUT_CLASSES[i]: float(probs[i] * 100) 
                    for i in range(len(self.CUT_CLASSES))
                }
            }
        except Exception as e:
            logger.error(f"Cut prediction error: {e}")
            return self._dummy_prediction(self.CUT_CLASSES)
    
    def predict_shape(self, img):
        """Predict diamond shape"""
        if not MINDSPORE_AVAILABLE:
            return self._dummy_prediction(self.SHAPE_CLASSES)
        
        try:
            # Set to evaluation mode
            self.shape_model.set_train(False)
            
            img_tensor = Tensor(img, ms.float32)
            output = self.shape_model(img_tensor)
            probs = nn.Softmax()(output).asnumpy()[0]
            
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx] * 100)
            
            return {
                "prediction": self.SHAPE_CLASSES[pred_idx],
                "confidence": confidence,
                "all_probabilities": {
                    self.SHAPE_CLASSES[i]: float(probs[i] * 100) 
                    for i in range(len(self.SHAPE_CLASSES))
                }
            }
        except Exception as e:
            logger.error(f"Shape prediction error: {e}")
            return self._dummy_prediction(self.SHAPE_CLASSES)
    
    def predict_color(self, img):
        """Predict diamond color grade"""
        if not MINDSPORE_AVAILABLE:
            return self._dummy_prediction(self.COLOR_CLASSES)
        
        try:
            # Set to evaluation mode
            self.color_model.set_train(False)
            
            img_tensor = Tensor(img, ms.float32)
            output = self.color_model(img_tensor)
            probs = nn.Softmax()(output).asnumpy()[0]
            
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx] * 100)
            
            return {
                "prediction": self.COLOR_CLASSES[pred_idx],
                "confidence": confidence,
                "all_probabilities": {
                    self.COLOR_CLASSES[i]: float(probs[i] * 100) 
                    for i in range(len(self.COLOR_CLASSES))
                }
            }
        except Exception as e:
            logger.error(f"Color prediction error: {e}")
            return self._dummy_prediction(self.COLOR_CLASSES)
    
    def _dummy_prediction(self, classes):
        """Generate dummy predictions for testing (when models not loaded)"""
        import random
        pred_idx = random.randint(0, len(classes) - 1)
        confidence = random.uniform(75, 95)
        
        logger.warning("Using dummy prediction - models not loaded!")
        
        return {
            "prediction": classes[pred_idx],
            "confidence": confidence,
            "all_probabilities": {
                cls: random.uniform(0, 100) for cls in classes
            }
        }

# Initialize models (will load .ckpt files from checkpoints/)
logger.info("Initializing Diamond Grading Models...")
models = DiamondGradingModels(checkpoint_dir='./checkpoints')

def decode_image(base64_string):
    """Decode base64 image to numpy array"""
    try:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AutoGemGrade Cloud Server",
        "status": "running",
        "version": "1.0.0",
        "mindspore_available": MINDSPORE_AVAILABLE,
        "models_loaded": models.models_loaded if MINDSPORE_AVAILABLE else False
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models.models_loaded if MINDSPORE_AVAILABLE else False,
        "mindspore_status": "available" if MINDSPORE_AVAILABLE else "unavailable",
        "checkpoint_dir": str(models.checkpoint_dir) if MINDSPORE_AVAILABLE else None
    }

@app.post("/analyze")
async def analyze_diamond(request: AnalysisRequest):
    """Analyze diamond image and return grading results"""
    try:
        logger.info("Starting diamond analysis")
        
        # Decode image
        img = decode_image(request.image)
        logger.info(f"Image decoded: {img.shape}")
        
        # Preprocess for models
        img_processed = models.preprocess_for_model(img)
        logger.info("Image preprocessed for models")
        
        # Run predictions using trained models
        cut_result = models.predict_cut(img_processed)
        shape_result = models.predict_shape(img_processed)
        color_result = models.predict_color(img_processed)
        
        logger.info("All predictions complete")
        
        # Compile results
        results = {
            "success": True,
            "cut": cut_result,
            "shape": shape_result,
            "color": color_result,
            "edge_metadata": request.edge_metadata,
            "overall_confidence": float(np.mean([
                cut_result["confidence"],
                shape_result["confidence"],
                color_result["confidence"]
            ])),
            "models_loaded": models.models_loaded if MINDSPORE_AVAILABLE else False
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/batch_analyze")
async def batch_analyze(images: list[str]):
    """Analyze multiple diamond images"""
    results = []
    
    for idx, img_base64 in enumerate(images):
        try:
            result = await analyze_diamond(
                AnalysisRequest(image=img_base64)
            )
            results.append({
                "image_id": idx,
                "result": result
            })
        except Exception as e:
            results.append({
                "image_id": idx,
                "error": str(e)
            })
    
    return {"batch_results": results}

if __name__ == "__main__":
    import uvicorn
    
    # Check if models are loaded
    if not models.models_loaded:
        logger.warning("=" * 60)
        logger.warning("WARNING: No trained models found!")
        logger.warning("Please upload your .ckpt files to ./checkpoints/")
        logger.warning("The system will use dummy predictions until models are loaded.")
        logger.warning("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)