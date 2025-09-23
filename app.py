import os
import io
import random
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    pipeline,
    ViTImageProcessor,
    ViTForImageClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, for private models

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = 0 if torch.cuda.is_available() else -1

# Lazy-loaded models
models = {
    "text_classifier": None,
    "url_tokenizer": None,
    "url_model": None,
    "image_processor": None,
    "image_model": None,
}

# Request schemas
class TextAnalysisRequest(BaseModel):
    text: str

class UrlAnalysisRequest(BaseModel):
    url: str

# Helper: format results
def format_result(content_type, raw_result, input_text):
    primary_result = raw_result[0] if isinstance(raw_result, list) else raw_result
    label = primary_result.get("label", "").lower()
    score = primary_result.get("score", 0.5)

    is_misinfo = any(label.startswith(x) for x in ["misinformation", "fake", "defacement", "malware"])
    is_phishing = any(label.startswith(x) for x in ["phishing", "malicious"])

    if is_misinfo or is_phishing:
        score = 1 - score

    return {
        "type": content_type,
        "credibilityScore": score,
        "analysis": f"The {content_type} has been flagged as potential {primary_result.get('label', 'misinformation')} with confidence {score:.2f}.",
        "flags": {
            "potentialMisinformation": is_misinfo,
            "needsFactChecking": is_misinfo or is_phishing,
            "biasDetected": "bias" in label,
            "manipulatedContent": is_misinfo or is_phishing,
        },
        "sources": [
            f"AI Model: {primary_result.get('label', 'N/A')}",
            "External Verification Service"
        ],
        "details": {
            "sentiment": "Negative" if is_misinfo else "Positive/Neutral",
            "confidence": primary_result.get("score", 0.5),
            "keyTerms": input_text.split()[:5] if input_text else []
        }
    }

# Fallback mock
def mock_analysis(content_type, input_text):
    mock_score = random.uniform(0.3, 0.9)
    is_misinfo = mock_score < 0.6
    return {
        "type": content_type,
        "credibilityScore": mock_score,
        "analysis": "Analysis failed: AI models could not be loaded. This is a mock result.",
        "flags": {
            "potentialMisinformation": is_misinfo,
            "needsFactChecking": is_misinfo,
            "biasDetected": random.choice([True, False]),
            "manipulatedContent": random.choice([True, False]),
        },
        "sources": ["Offline Mode (Mock)"],
        "details": {
            "sentiment": "N/A",
            "confidence": mock_score,
            "keyTerms": input_text.split()[:5] if input_text else []
        },
    }

# Lazy load models
async def load_text_model():
    if models["text_classifier"] is None:
        try:
            models["text_classifier"] = pipeline(
                "text-classification",
                model="dhruvpal/fake-news-bert",
                device=device
            )
            print("Text model loaded")
        except Exception as e:
            print(f"Failed to load text model: {e}")

async def load_url_model():
    if models["url_tokenizer"] is None or models["url_model"] is None:
        try:
            # Using a public text classifier to avoid token issues
            models["url_tokenizer"] = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            models["url_model"] = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("URL model loaded")
        except Exception as e:
            print(f"Failed to load URL model: {e}")

async def load_image_model():
    if models["image_processor"] is None or models["image_model"] is None:
        try:
            models["image_processor"] = ViTImageProcessor.from_pretrained("Wvolf/ViT_Deepfake_Detection")
            models["image_model"] = ViTForImageClassification.from_pretrained("Wvolf/ViT_Deepfake_Detection")
            print("Image model loaded")
        except Exception as e:
            print(f"Failed to load image model: {e}")

# Endpoints
@app.post("/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    await load_text_model()
    if not models["text_classifier"]:
        return {"result": mock_analysis("text", request.text)}
    try:
        result = await asyncio.to_thread(models["text_classifier"], request.text)
        return {"result": format_result("text", result, request.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.post("/analyze/url")
async def analyze_url(request: UrlAnalysisRequest):
    await load_url_model()
    if not models["url_model"]:
        return {"result": mock_analysis("url", request.url)}
    try:
        inputs = models["url_tokenizer"](request.url, return_tensors="pt")
        with torch.no_grad():
            outputs = models["url_model"](**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_id = torch.argmax(probs).item()
        label = models["url_model"].config.id2label[pred_id]
        score = probs[0][pred_id].item()
        return {"result": format_result("url", [{"label": label, "score": score}], request.url)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL analysis failed: {str(e)}")

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    await load_image_model()
    if not models["image_model"]:
        return {"result": mock_analysis("image", file.filename)}
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        inputs = models["image_processor"](images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = models["image_model"](**inputs)
        logits = outputs.logits
        pred_id = logits.argmax(-1).item()
        label = models["image_model"].config.id2label[pred_id]
        score = torch.nn.functional.softmax(logits, dim=1)[0][pred_id].item()
        return {"result": format_result("image", {"label": label, "score": score}, file.filename)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/analyze/video")
async def analyze_video():
    raise HTTPException(status_code=501, detail="Video analysis not yet implemented")
