"""
Módulo de Inferência em Tempo Real com Diagnóstico Gemini
Framework: FastAPI + Pydantic v2 + Google GenAI SDK
Modelo: Support Vector Machine (Kernel RBF)
"""

import os
import joblib
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Predictive Maintenance SVM + Gemini Engine",
    description="Diagnóstico físico com SVM RBF e parecer operacional automatizado com LLM.",
    version="2.0.0"
)

modelo_svm = None
scaler = None
gemini_client = None


@app.on_event("startup")
def inicializar_servicos():
    global modelo_svm, scaler, gemini_client
    caminho_modelo = MODELS_DIR / "svm_rbf_otimizado.joblib"
    caminho_scaler = MODELS_DIR / "scaler_sensores.joblib"

    if not caminho_modelo.exists() or not caminho_scaler.exists():
        raise RuntimeError(f"Artefatos não encontrados em {MODELS_DIR}.")

    modelo_svm = joblib.load(caminho_modelo)
    scaler = joblib.load(caminho_scaler)

    chave_api = os.getenv("GEMINI_API_KEY")
    if chave_api:
        gemini_client = genai.Client(api_key=chave_api)
        print("✓ Cliente Gemini inicializado com sucesso.")
    else:
        print("Aviso: GEMINI_API_KEY não configurada no ambiente.")


class TelemetriaEntrada(BaseModel):
    air_temperature_k: float = Field(..., ge=250.0, le=350.0)
    process_temperature_k: float = Field(..., ge=250.0, le=350.0)
    rotational_speed_rpm: float = Field(..., gt=0.0, le=5000.0)
    torque_nm: float = Field(..., ge=0.0, le=150.0)
    tool_wear_min: float = Field(..., ge=0.0, le=500.0)
    tipo_produto: str = Field(..., pattern="^[LMH]$")


class DiagnosticoSaida(BaseModel):
    alerta_falha: bool
    classe_predita: int
    distancia_hiperplano: float
    estado_ativo: str
    metricas_engenharia: Dict[str, float]
    parecer_tecnico_gemini: Optional[str] = None


def _chamar_gemini_sync(prompt: str) -> str:
    """Função síncrona isolada que executa em thread separada."""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Aviso na geração do parecer: {str(e)}"


async def gerar_parecer_tecnico_async(telemetria: TelemetriaEntrada, metricas: Dict[str, float], distancia: float, falha: bool) -> str:
    if not gemini_client:
        return "Parecer de IA indisponível: variável GEMINI_API_KEY não configurada no ambiente."

    prompt = f"""
    Atue como um Engenheiro Mecânico Perito em Manutenção Preditiva Industrial.
    Analise o ponto de operação avaliado pelo modelo SVM RBF:

    - Temp. Ambiente: {telemetria.air_temperature_k} K
    - Temp. Processo: {telemetria.process_temperature_k} K (ΔT: {metricas['delta_T_Kelvin']} K)
    - Rotação: {telemetria.rotational_speed_rpm} RPM | Torque: {telemetria.torque_nm} Nm
    - Potência Mecânica: {metricas['potencia_kW']} kW
    - Desgaste da Ferramenta: {telemetria.tool_wear_min} min (Esforço τ×t: {metricas['esforco_desgaste']} Nm·min)
    - Produto: Variante {telemetria.tipo_produto}

    Classificação SVM:
    - Estado: {"ALERTA DE FALHA DETECTADO" if falha else "OPERAÇÃO NORMAL"}
    - Distância ao Hiperplano: {distancia:.4f}

    Forneça um parecer direto em no máximo 2 tópicos:
    1. Análise Causal do subsistema sob estresse mecânico ou térmico.
    2. Ação Imediata Recomendada para a equipe de operação.
    """

    try:
        # Executa em thread sem bloquear o loop e com cancelamento estrito em 6 segundos
        loop = asyncio.get_running_loop()
        tarefa = loop.run_in_executor(None, _chamar_gemini_sync, prompt)
        return await asyncio.wait_for(tarefa, timeout=6.0)
    except asyncio.TimeoutError:
        return "Aviso: A resposta da API do Gemini demorou mais de 6s. O diagnóstico físico do SVM RBF e as métricas calculadas permanecem 100% válidos."
    except Exception as e:
        return f"Aviso: Não foi possível obter parecer ({str(e)}). Diagnóstico numérico do SVM válido."


@app.get("/health", status_code=status.HTTP_200_OK)
def verificar_integridade():
    return {
        "status": "online",
        "modelo_carregado": modelo_svm is not None,
        "scaler_carregado": scaler is not None,
        "gemini_disponivel": gemini_client is not None
    }


@app.post("/predict", response_model=DiagnosticoSaida, status_code=status.HTTP_200_OK)
async def predizer_falha(payload: TelemetriaEntrada):
    if modelo_svm is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço indisponível: modelos não carregados."
        )

    # 1. Cálculos de Engenharia
    delta_t = payload.process_temperature_k - payload.air_temperature_k
    potencia_rotacional = payload.torque_nm * (payload.rotational_speed_rpm * (2 * np.pi / 60)) / 1000.0
    esforco_desgaste = payload.torque_nm * payload.tool_wear_min

    tipo_m = 1.0 if payload.tipo_produto == "M" else 0.0
    tipo_l = 1.0 if payload.tipo_produto == "L" else 0.0

    features_dict = {
        'Air_temperature_K': payload.air_temperature_k,
        'Process_temperature_K': payload.process_temperature_k,
        'Rotational_speed_rpm': payload.rotational_speed_rpm,
        'Torque_Nm': payload.torque_nm,
        'Tool_wear_min': payload.tool_wear_min,
        'delta_T': delta_t,
        'potencia_rotacional_kW': potencia_rotacional,
        'esforco_desgaste': esforco_desgaste,
        'Tipo_M': tipo_m,
        'Tipo_L': tipo_l
    }

    # 2. Inferência SVM RBF
    df_amostra = pd.DataFrame([features_dict])
    amostra_escalada = scaler.transform(df_amostra)
    classe = int(modelo_svm.predict(amostra_escalada)[0])
    distancia = float(modelo_svm.decision_function(amostra_escalada)[0])
    falha = bool(classe == 1)

    metricas = {
        'delta_T_Kelvin': round(delta_t, 2),
        'potencia_kW': round(potencia_rotacional, 2),
        'esforco_desgaste': round(esforco_desgaste, 2)
    }

    # 3. Parecer Gemini com timeout estrito não-bloqueante
    parecer = await gerar_parecer_tecnico_async(payload, metricas, distancia, falha)

    return DiagnosticoSaida(
        alerta_falha=falha,
        classe_predita=classe,
        distancia_hiperplano=round(distancia, 4),
        estado_ativo="Alerta de Falha Detectado" if falha else "Operação Normal",
        metricas_engenharia=metricas,
        parecer_tecnico_gemini=parecer
    )