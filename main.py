import io
import os
import cv2
import numpy as np
import base64
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# Importamos as funções diretas do arquivo da cliente
from Predict import calculate_disease_percentages, save_mask_visualization

app = FastAPI(
    title="API de Diagnóstico Odontológico",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_PATH = os.path.join(os.getcwd(), "best_rebuilt.pt")

try:
    print(f"Carregando modelo YOLO a partir de: {WEIGHTS_PATH}...")
    model = YOLO(WEIGHTS_PATH)
    print("Modelo YOLO carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo YOLO: {e}")
    model = None


class DiagnosisResult(BaseModel):
    status: str
    filename: str
    plaque_percent: float
    gingivitis_percent: float
    image_base64: str


@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": model is not None
    }


@app.post("/analyze-image", response_model=DiagnosisResult)
async def analyze_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="O modelo de IA não está carregado no servidor"
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Imagem inválida. Tente outra"
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem.")

        # 1. Predição YOLO
        results = model.predict(source=img, save=False, task="segment")
        result = results[0]

        # 2. Porcentagens de placa e gengivite
        percentages = calculate_disease_percentages(result, source_path=file.filename)

        # 3. Gera a imagem mascarada usando a função nativa save_mask_visualization
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Salva temporariamente no disco
            visual_path = save_mask_visualization(result, percentages, save_root=tmpdirname)
            
            # Lê o arquivo gerado e converte para Base64
            with open(visual_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Formata no padrão de Data URI
            image_base64 = f"data:image/png;base64,{encoded_string}"

        # 4. Retorna tudo no mesmo JSON
        return {
            "status": "sucesso",
            "filename": file.filename,
            "plaque_percent": round(percentages["plaque_percent"], 2),
            "gingivitis_percent": round(percentages["gingivitis_percent"], 2),
            "image_base64": image_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da imagem: {str(e)}")