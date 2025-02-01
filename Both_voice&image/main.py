# from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# from pydantic import BaseModel
# import sqlite3
# import os
# from typing import List, Optional
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# from transformers import pipeline
# import whisper
# import uuid
# import logging

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins (for development only)
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# # Initialize AI models
# try:
#     whisper_model = whisper.load_model("base")  # Load Whisper for voice transcription
# except Exception as e:
#     logger.error(f"Error loading Whisper model: {e}")
#     whisper_model = None

# try:
#     image_pipeline = pipeline("image-classification", model="google/vit-base-patch16-224")  # Load ViT for image classification
# except Exception as e:
#     logger.error(f"Error loading image classification pipeline: {e}")
#     image_pipeline = None

# # SQLite database setup
# DATABASE = "files.db"

# def init_db():
#     with sqlite3.connect(DATABASE) as conn:
#         cursor = conn.cursor()
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS files (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 file_type TEXT NOT NULL,
#                 file_path TEXT NOT NULL,
#                 processed_result TEXT
#             )
#         """)
#         # Ensure the `processed_result` column exists
#         cursor.execute("PRAGMA table_info(files)")
#         columns = cursor.fetchall()
#         column_names = [column[1] for column in columns]
#         if 'processed_result' not in column_names:
#             cursor.execute("ALTER TABLE files ADD COLUMN processed_result TEXT")
#         conn.commit()

# init_db()

# # Pydantic model for response
# class FileResponse(BaseModel):
#     id: int
#     file_type: str
#     file_path: str
#     processed_result: Optional[str] = None

# # Root route
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the FastAPI server!"}

# # Helper function to save files
# async def save_file(file: UploadFile, upload_folder: str) -> str:
#     file_extension = os.path.splitext(file.filename)[1]
#     unique_filename = f"{uuid.uuid4()}{file_extension}"
#     file_path = f"{upload_folder}/{unique_filename}"
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     try:
#         with open(file_path, "wb") as buffer:
#             buffer.write(await file.read())
#         return file_path
#     except Exception as e:
#         logger.error(f"Failed to save file: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

# # Endpoint to upload voice file
# @app.post("/upload/voice/", response_model=FileResponse)
# async def upload_voice(file: UploadFile = File(...)):
#     if not file.content_type.startswith("audio/"):
#         raise HTTPException(status_code=400, detail="File must be an audio file")

#     file_path = await save_file(file, "uploads/voice")

#     # Store file info in SQLite
#     try:
#         with sqlite3.connect(DATABASE) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO files (file_type, file_path)
#                 VALUES (?, ?)
#             """, ("voice", file_path))
#             conn.commit()
#             file_id = cursor.lastrowid
#     except Exception as e:
#         os.remove(file_path)  # Clean up the file if database insertion fails
#         logger.error(f"Database error: {e}")
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")

#     return {"id": file_id, "file_type": "voice", "file_path": file_path}

# # Endpoint to upload images
# @app.post("/upload/image/", response_model=FileResponse)
# async def upload_image(file: UploadFile = File(...)):
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File must be an image file")

#     file_path = await save_file(file, "uploads/images")

#     try:
#         with sqlite3.connect(DATABASE) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO files (file_type, file_path, processed_result)
#                 VALUES (?, ?, NULL)
#             """, ("image", file_path))
#             conn.commit()
#             file_id = cursor.lastrowid
#     except Exception as e:
#         os.remove(file_path)  # Clean up the file if database insertion fails
#         logger.error(f"Database error: {e}")
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")

#     # Automatically process the file after upload
#     await parse_files()

#     return {"id": file_id, "file_type": "image", "file_path": file_path}

# # Endpoint to retrieve all files
# @app.get("/files/", response_model=List[FileResponse])
# async def get_files():
#     try:
#         with sqlite3.connect(DATABASE) as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT id, file_type, file_path, processed_result FROM files")
#             files = cursor.fetchall()
#     except Exception as e:
#         logger.error(f"Database error: {e}")
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")

#     return [{"id": row[0], "file_type": row[1], "file_path": row[2], "processed_result": row[3]} for row in files]

# # Endpoint to parse files for AI models
# @app.post("/parse/")
# async def parse_files():
#     print("Starting file processing...")  # Debug

#     try:
#         with sqlite3.connect(DATABASE) as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT id, file_type, file_path FROM files WHERE processed_result IS NULL")
#             files = cursor.fetchall()
#     except Exception as e:
#         logger.error(f"Database error: {e}")
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")

#     if not files:
#         return {"message": "No new files to process"}

#     for file_id, file_type, file_path in files:
#         print(f"Processing file {file_id} ({file_type}) at {file_path}")  # Debug

#         processed_result = None
#         try:
#             if file_type == "voice" and whisper_model:
#                 print("Using Whisper Model...")  # Debug
#                 result = whisper_model.transcribe(file_path)
#                 processed_result = result["text"]
#             elif file_type == "image" and image_pipeline:
#                 print("Using Image Model...")  # Debug
#                 result = image_pipeline(file_path)
#                 processed_result = ", ".join([f"{res['score']:.2f}% {res['label']}" for res in result])

#                 # Ensure the file path is absolute
#                 full_file_path = os.path.abspath(file_path)
#                 if not os.path.exists(full_file_path):
#                     processed_result = f"Error: File not found at {full_file_path}"
#                 else:
#                     try:
#                         result = image_pipeline(full_file_path)
#                         print(f"Raw model output: {result}")  # Debug
                        
#                         # Process model output
#                         if result and isinstance(result, list):
#                             processed_result = ", ".join([f"{res['score']:.2f}% {res['label']}" for res in result])
#                         else:
#                             processed_result = "Error: No valid output from model"
                        
#                         print(f"Processed result: {processed_result}")  # Debug
#                     except Exception as e:
#                         processed_result = f"Error processing file: {e}"
#                         print(f"Model error: {e}")  # Debug

#             else:
#                 print("Model not available!")  # Debug
#                 processed_result = "Error: Model not available"
#         except Exception as e:
#             logger.error(f"Error processing file {file_id}: {e}")
#             processed_result = f"Error processing file: {e}"

#         # Update the database
#         try:
#             with sqlite3.connect(DATABASE) as conn:
#                 cursor = conn.cursor()
#                 print(f"Final processed_result: {processed_result}")  # Debug
#                 cursor.execute("""
#                     UPDATE files
#                     SET processed_result = ?
#                     WHERE id = ?
#                 """, (processed_result, file_id))
#                 conn.commit()
#                 print(f"Database update successful for file {file_id}")  # Debug
                
#                 # Fetch and verify the updated row
#                 cursor.execute("SELECT id, processed_result FROM files WHERE id = ?", (file_id,))
#                 updated_row = cursor.fetchone()
#                 print(f"Updated row: {updated_row}")  # Debug
#         except Exception as e:
#             logger.error(f"Failed to update database for file {file_id}: {e}")
#             print(f"Database update failed: {e}")  # Debug

#     return {"message": "Files parsed successfully", "results": [{"file_id": file[0], "result": processed_result} for file in files]}


from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import sqlite3
import os
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import whisper
import uuid
import logging
from PIL import Image
from transformers import pipeline
import json

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

# Initialize AI models
try:
    whisper_model = whisper.load_model("base")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    whisper_model = None

try:
    image_pipeline = pipeline("image-classification", model="google/vit-base-patch16-224")
except Exception as e:
    logger.error(f"Error loading image classification pipeline: {e}")
    image_pipeline = None

# Database setup
DATABASE = "files.db"

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        # Files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                result TEXT NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(id)
            )
        """)
        conn.commit()

init_db()

class FileResponse(BaseModel):
    id: int
    file_type: str
    file_path: str

class ParseRequest(BaseModel):
    file_ids: List[int]

class ProcessedResult(BaseModel):
    file_id: int
    result: str

@app.get("/")
async def root():
    return {"message": "File Processing API"}

async def save_file(file: UploadFile, upload_folder: str) -> str:
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = f"{upload_folder}/{unique_filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        return file_path
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

def clear_previous_files(file_type: str):
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            # Get existing file IDs
            cursor.execute("SELECT id FROM files WHERE file_type = ?", (file_type,))
            old_ids = [row[0] for row in cursor.fetchall()]
            
            # Delete related processed data
            if old_ids:
                cursor.executemany("DELETE FROM processed_data WHERE file_id = ?", [(id,) for id in old_ids])
            
            # Delete old files
            cursor.execute("DELETE FROM files WHERE file_type = ?", (file_type,))
            conn.commit()
    except Exception as e:
        logger.error(f"Database error during cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear previous files")

@app.post("/upload/voice/", response_model=FileResponse)
async def upload_voice(file: UploadFile = File(...)):
    # logger.info(f"Uploaded file: {file.filename}, MIME type: {file.content_type}")
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Clear previous voice files
    clear_previous_files("voice")
    
    file_path = await save_file(file, "uploads/voice")
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO files (file_type, file_path) VALUES (?, ?)", ("voice", file_path))
            conn.commit()
            file_id = cursor.lastrowid
        return {"id": file_id, "file_type": "voice", "file_path": file_path}
    except Exception as e:
        os.remove(file_path)
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.post("/upload/image/", response_model=FileResponse)
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image file")
    
    # Clear previous image files
    clear_previous_files("image")
    
    file_path = await save_file(file, "uploads/images")
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO files (file_type, file_path) VALUES (?, ?)", ("image", file_path))
            conn.commit()
            file_id = cursor.lastrowid
        return {"id": file_id, "file_type": "image", "file_path": file_path}
    except Exception as e:
        os.remove(file_path)
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.post("/parse/")
async def parse_files():
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.id, f.file_type, f.file_path 
                FROM files f
                LEFT JOIN processed_data p ON f.id = p.file_id
                WHERE p.file_id IS NULL
                ORDER BY f.uploaded_at DESC
                LIMIT 2
            """)
            files = cursor.fetchall()
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch files")

    if not files:
        return {"message": "No files to process"}

    results = []
    for file in files:
        try:
            file_id = file["id"]
            file_type = file["file_type"]
            file_path = file["file_path"]

            # Verify file exists
            if not os.path.exists(file_path):
                logger.warning(f"Skipping missing file: {file_path}")
                continue  # Skip to the next file

            logger.info(f"Processing file {file_id}: {file_path} (Type: {file_type})")

            # Type-specific processing
            if file_type == "voice":
                result = whisper_model.transcribe(file_path)
                processed_data = {"transcription": result["text"]}

            elif file_type == "image":
                with Image.open(file_path) as img:
                    result = image_pipeline(img)
                    processed_data = [{"label": res["label"], "score": res["score"]} for res in result]

            else:
                logger.warning(f"Skipping unsupported file type: {file_type}")
                continue  # Move to the next file

            # Store results
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO processed_data (file_id, result)
                    VALUES (?, ?)
                """, (file_id, json.dumps(processed_data)))
                conn.commit()

            results.append({"file_id": file_id, "status": "success"})

        except Exception as e:
            logger.error(f"Error processing {file_type} file {file_id}: {str(e)}")
            results.append({"file_id": file_id, "status": "error", "message": str(e)})

    return {"message": "Processing completed", "results": results}



@app.get("/results/", response_model=List[ProcessedResult])
async def get_results():
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.file_id, p.result 
                FROM processed_data p
                INNER JOIN files f ON p.file_id = f.id
            """)
            results = cursor.fetchall()
            return [{"file_id": row[0], "result": row[1]} for row in results]
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")