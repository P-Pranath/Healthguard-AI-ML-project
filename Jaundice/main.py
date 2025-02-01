# from fastapi import FastAPI, File, UploadFile, HTTPException
# from pydantic import BaseModel
# import sqlite3
# import os
# import uuid
# import logging
# import whisper
# from fastapi.middleware.cors import CORSMiddleware

# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Whisper model setup
# try:
#     whisper_model = whisper.load_model("base")
#     print("Whisper model loaded successfully.")
# except Exception as e:
#     logger.error(f"Error loading Whisper model: {e}")
#     whisper_model = None
#     print(f"Error loading Whisper model: {e}")

# if not whisper_model:
#     raise HTTPException(status_code=500, detail="Whisper model failed to load")

# # Database setup
# DATABASE = "voice.db"

# def init_db():
#     with sqlite3.connect(DATABASE) as conn:
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS voice_files (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 file_path TEXT NOT NULL,
#                 uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
#             )
#         """)
#         conn.commit()

# init_db()

# class VoiceResponse(BaseModel):
#     id: int
#     file_path: str

# @app.get("/")
# async def root():
#     return {"message": "Voice Processing API"}

# async def save_audio(file: UploadFile) -> str:
#     file_extension = os.path.splitext(file.filename)[1]
#     unique_filename = f"{uuid.uuid4()}{file_extension}"
#     file_path = f"uploads/voice/{unique_filename}"
    
#     # Check if directory exists
#     if not os.path.exists(os.path.dirname(file_path)):
#         logger.error(f"Directory does not exist: {os.path.dirname(file_path)}")
#         raise HTTPException(status_code=500, detail="Directory does not exist")

#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
#     try:
#         with open(file_path, "wb") as buffer:
#             buffer.write(await file.read())
#         return file_path
#     except Exception as e:
#         logger.error(f"Failed to save audio: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to save audio: {e}")


# @app.post("/upload/voice/", response_model=VoiceResponse)
# async def upload_voice(file: UploadFile = File(...)):
#     if not file.content_type.startswith("audio/"):
#         raise HTTPException(status_code=400, detail="File must be an audio file")
    
#     try:
#         file_path = await save_audio(file)
#         with sqlite3.connect(DATABASE) as conn:
#             cursor = conn.cursor()
#             cursor.execute("INSERT INTO voice_files (file_path) VALUES (?)", (file_path,))
#             conn.commit()
#             file_id = cursor.lastrowid
#         return {"id": file_id, "file_path": file_path}
#     except Exception as e:
#         if 'file_path' in locals() and os.path.exists(file_path):
#             os.remove(file_path)
#         logger.error(f"Upload error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Voice upload failed: {str(e)}")

# @app.post("/process/voice/")
# async def process_voice():
#     try:
#         with sqlite3.connect(DATABASE) as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT id, file_path FROM voice_files ORDER BY uploaded_at DESC LIMIT 1")
#             audio = cursor.fetchone()
#     except Exception as e:
#         logger.error(f"Database error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to fetch audio file")

#     if not audio:
#         raise HTTPException(status_code=400, detail="No audio to process")

#     file_id, file_path = audio

#     logger.info(f"File path being processed: {file_path}")
#     print(f"File path being processed: {file_path}")  # This will show in the terminal

#     try:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Audio file not found: {file_path}")
#         logger.info(f"Full file path: {os.path.abspath(file_path)}")
        
#         if not whisper_model:
#             raise Exception("Whisper model not loaded")

#         result = whisper_model.transcribe(file_path, fp16=False)
#         return {
#             "id": file_id,
#             "transcription": result["text"],
#             "language": result["language"]
#         }

#     except Exception as e:
#         logger.error(f"Processing error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tempfile
import logging
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Jaundice Prediction API!"}


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jaundice detection parameters
SKIN_LOWER_HSV = np.array([15, 40, 120], np.uint8)
SKIN_UPPER_HSV = np.array([25, 150, 255], np.uint8)
SCLERA_LOWER_HSV = np.array([0, 0, 200], np.uint8)  # Normal sclera (white)
SCLERA_UPPER_HSV = np.array([30, 50, 255], np.uint8)  # Jaundiced sclera (yellowish)

def analyze_jaundice(image: np.ndarray):
    """Analyze image for jaundice indicators"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Detect skin regions
    skin_mask = cv2.inRange(hsv, SKIN_LOWER_HSV, SKIN_UPPER_HSV)
    skin_pixels = hsv[skin_mask > 0]
    
    # Detect sclera regions
    sclera_mask = cv2.inRange(hsv, SCLERA_LOWER_HSV, SCLERA_UPPER_HSV)
    sclera_pixels = hsv[sclera_mask > 0]
    
    # Calculate color averages
    skin_yellow = np.mean(skin_pixels[:, 0]) if len(skin_pixels) > 0 else 0
    sclera_yellow = np.mean(sclera_pixels[:, 0]) if len(sclera_pixels) > 0 else 0
    
    return {
        'skin_yellow': float(skin_yellow),
        'sclera_yellow': float(sclera_yellow),
        'jaundice_probability': min(100, max(0, (skin_yellow * 0.6 + sclera_yellow * 0.4) / 2.5))
    }

@app.post("/analyze/jaundice/")
async def analyze_jaundice_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid image file")
    
    try:
        # Read image directly into memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        # Resize for processing
        image_np = cv2.resize(image_np, (500, 500))
        
        # Analyze for jaundice indicators
        results = analyze_jaundice(image_np)
        
        # Medical disclaimer
        results['disclaimer'] = "This is not a medical diagnosis. Consult a healthcare professional."
        
        return results
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
