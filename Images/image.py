from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import sqlite3
import os
import uuid
import logging
from PIL import Image
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image processing pipeline
try:
    image_pipeline = pipeline("image-classification", model="google/vit-base-patch16-224")
except Exception as e:
    logger.error(f"Error loading image classification pipeline: {e}")
    image_pipeline = None

# Database setup
DATABASE = "images.db"

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()

class ImageResponse(BaseModel):
    id: int
    file_path: str

@app.get("/")
async def root():
    return {"message": "Image Processing API"}

async def save_image(file: UploadFile) -> str:
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"uploads/images/{unique_filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        return file_path
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

@app.post("/upload/image/", response_model=ImageResponse)
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image file")
    
    try:
        file_path = await save_image(file)
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO images (file_path) VALUES (?)", (file_path,))
            conn.commit()
            file_id = cursor.lastrowid
        return {"id": file_id, "file_path": file_path}
    except Exception as e:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

@app.post("/process/image/")
async def process_image():
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_path FROM images ORDER BY uploaded_at DESC LIMIT 1")
            image = cursor.fetchone()
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch image")

    if not image:
        raise HTTPException(status_code=400, detail="No image to process")

    file_id, file_path = image

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if not image_pipeline:
            raise Exception("Image processing model not loaded")

        with Image.open(file_path) as img:
            result = image_pipeline(img)
            processed_data = [{
                "label": res["label"],
                "score": float(res["score"])
            } for res in result]

        return {
            "id": file_id,
            "results": processed_data
        }

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")