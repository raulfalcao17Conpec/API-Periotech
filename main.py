import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
'''
 Bibliotecas importadas e para que elas são usadas: 
 1. Com a NumPy, conseguimos pegar uma foto e transformá-la em uma matriz de  números, que é muito 
 mais fácil de ser operada com os métodos da biblioteca. 
 2. Com a OpenCV, a matriz gerada "via NumPy" é lida, e graças aos métodos da OpenCV é possível 
 desenhar as caixas e as máscaras. 
 3. Com a YOLO, a imagem transformada "via OpenCV" é lida e as diferentes classes (dente posterior, anterior,
placa,gengiva,etc) são identificadas pelo modelo treinado. Depois de divididas, é possível pegar os pontos que nos interessamm
# e realizar os cálculos necessários. 
'''

# Importamos as funções diretas do arquivo da cliente
from Predict import calculate_disease_percentages

'''
 Primeira etapa: criação de um instância do FastAPI. Por meio dela, é possível gerenciar todas 
 as rotas e documentação. Ela automaticamente cria a aplicação e gera as rotas de documentação (docs e redocs)
'''
app = FastAPI(
    title="API de Diagnóstico Odontológico",
    version="1.0.0"
)
'''
 Libera acesso para o aplicativo. Nos testes, o código do app roda no localhost:8081 e a 
API em localhost:8000. O que é comum é que o sistema operacional proíba que eles se comuniquem por questão de 
 de segurança. O que esses trechos fazem é só permitir que a comunicação aconteça 
 '''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite acesso de qualquer origem (App,web,etc)
    allow_credentials=True,
    allow_methods=["*"], # Permite qualquer método (POST,GET,DELETE)
    allow_headers=["*"],
)

''''
Carregamos o modelo antecipadamente. Como essa parte que mais demanda poder computacional,
 é essencial que façamos isso fora das rotas de requisição para não repetir sempre. 
 Com isso, esse trecho executa apenas 1 vez ao subir a API,e o modelo fica alocado na RAM do servidor
'''
WEIGHTS_PATH = os.path.join(os.getcwd(), "best_rebuilt.pt")

try:
    print(f"Carregando modelo YOLO a partir de: {WEIGHTS_PATH}...")
    model = YOLO(WEIGHTS_PATH)
    print("Modelo YOLO carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo YOLO: {e}")
    model = None

class DiagnosisResult(BaseModel):
#Define como vai ser a saída que o nosso app vai receber. Essa definição é possibilitada pela bibilioteca pydantic que tem a classe BaseModel.  

    status: str
    filename: str
    plaque_percent: float
    gingivitis_percent: float

# No FastAPI, usa-se decoradores para indicar os endpoints. 
# Endpoints: um link específico onde a nossa API recebe e fornece respostas
@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": model is not None
    }

# Executa o processamento da imagem. O mais interessante é que é usando a biblioteca OpenCV é possível
# tratar a imagem sem salvá-la no disco, o que deixa tudo mais rápido. 
@app.post("/analyze-image", response_model=DiagnosisResult)
async def analyze_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="O modelo de IA não carregado no servidor"
        )

    # Validação do arquivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Imagem inválida. Tente outra"
        )

    try:
        # 1. Lê o arquivo em bytes
        contents = await file.read()
        # Transforma os dados em uma matriz numérica 
        nparr = np.frombuffer(contents, np.uint8)
        # O openCV decodifica a matriz na imagem 
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem.")

        # 2. Executa a predição de segmentação do YOLO
        results = model.predict(source=img, save=False, task="segment")
        result = results[0] # Pega o resultado da primeira imagem

        # 3. Utiliza a função para calcular as porcentagens reais
        percentages = calculate_disease_percentages(result, source_path=file.filename)

        # 4. Retorna os índices calculados para o nosso aplicativo 
        return {
            "status": "sucesso",
            "filename": file.filename,
            "plaque_percent": round(percentages["plaque_percent"], 2),
            "gingivitis_percent": round(percentages["gingivitis_percent"], 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da imagem: {str(e)}")