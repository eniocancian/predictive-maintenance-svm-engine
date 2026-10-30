"""
Módulo de Inferência em Tempo Real com Diagnóstico Pericial Híbrido
Classificador: Support Vector Machine (Kernel RBF)
Diagnóstico Causal: Motor Físico AI4I + Google Gemini
"""

import os
import joblib
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
    title="Predictive Maintenance SVM Engine",
    description="Diagnóstico físico com SVM RBF e parecer operacional automatizado.",
    version="2.1.0"
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
        raise RuntimeError("Artefatos do modelo nao encontrados na pasta models.")

    modelo_svm = joblib.load(caminho_modelo)
    scaler = joblib.load(caminho_scaler)

    chave_api = os.getenv("GEMINI_API_KEY")
    if chave_api:
        try:
            gemini_client = genai.Client(api_key=chave_api)
            print("✓ Cliente Gemini pronto.")
        except Exception as e:
            print(f"Aviso Gemini: {e}")
            gemini_client = None


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
    parecer_tecnico_gemini: str


def diagnostico_causal_engenharia(telemetria: TelemetriaEntrada, metricas: Dict[str, float], distancia: float, falha: bool) -> str:
    if not falha:
        return (
            "PARECER TÉCNICO DE MANUTENÇÃO: REGIME ESTÁVEL\n\n"
            "1. Análise Causal: Operação em Conformidade Nominal\n"
            f"   • O ponto de trabalho reside na região segura do hiperplano (distância = {distancia:.4f} < 0).\n"
            f"   • Gradiente térmico (ΔT = {metricas['delta_T_Kelvin']} K) e esforço acumulado ({metricas['esforco_desgaste']} Nm·min) dentro dos parâmetros normais.\n\n"
            "2. Ação Imediata Recomendada:\n"
            "   • Manter o ciclo contínuo de usinagem com monitoramento rotineiro de vibração e temperatura."
        )

    delta_t = metricas['delta_T_Kelvin']
    potencia = metricas['potencia_kW']
    esforco = metricas['esforco_desgaste']
    tipo = telemetria.tipo_produto

    limiar_osf = 11000.0 if tipo == 'L' else (12000.0 if tipo == 'M' else 13000.0)
    causas = []
    acoes = []

    if esforco > limiar_osf:
        causas.append(f"OSF (Overstrain Failure): Esforço mecânico integrado ({esforco:.1f} Nm·min) excedeu o limite de ruptura de {limiar_osf} Nm·min para a variante {tipo}.")
        acoes.append("Interromper a operação imediatamente e inspecionar contra deformação plástica no porta-ferramentas.")

    if telemetria.tool_wear_min >= 200.0:
        causas.append(f"TWF (Tool Wear Failure): Desgaste de corte atingiu {telemetria.tool_wear_min:.1f} min (zona terminal de 200 a 240 min).")
        acoes.append("Substituir o inserto de corte antes de iniciar a usinagem da próxima peça.")

    if delta_t < 8.6 and telemetria.rotational_speed_rpm < 1380.0:
        causas.append(f"HDF (Heat Dissipation Failure): Dissipação térmica insuficiente (ΔT = {delta_t:.1f} K < 8.6 K) sob baixa rotação.")
        acoes.append("Verificar o fluxo de refrigerante e desobstruir linhas de fluido de corte.")

    if potencia < 3.5 or potencia > 9.0:
        causas.append(f"PWF (Power Failure): Potência rotacional calculada ({potencia:.2f} kW) fora da faixa segura de trabalho (3.5 a 9.0 kW).")
        acoes.append("Inspecionar a alimentação elétrica do acionamento principal e alinhamento do motor.")

    if not causas:
        causas.append(f"Invasão Crítica de Hiperplano (Distância = +{distancia:.4f}) por estresse conjugado multivariável.")
        acoes.append("Parada preventiva para inspeção técnica e calibração dos sensores.")

    causas_formatadas = "\n   • ".join(causas)
    acoes_formatadas = "\n   • ".join(acoes)

    return (
        "PARECER TÉCNICO DE MANUTENÇÃO PREDITIVA\n\n"
        f"1. Análise Causal do Ponto de Estresse:\n   • {causas_formatadas}\n\n"
        f"2. Ação Imediata Recomendada para o Operador:\n   • {acoes_formatadas}"
    )


def obter_parecer(telemetria: TelemetriaEntrada, metricas: Dict[str, float], distancia: float, falha: bool) -> str:
    laudo_base = diagnostico_causal_engenharia(telemetria, metricas, distancia, falha)

    if not gemini_client:
        return laudo_base

    prompt = f"""
    Atue como Perito de Manutenção Industrial e resuma o seguinte diagnóstico em 2 tópicos objetivos:
    Falha={falha}, Distância SVM={distancia:.4f}, DeltaT={metricas['delta_T_Kelvin']}K, Potência={metricas['potencia_kW']}kW.
    Laudo Base:
    {laudo_base}
    """

    try:
        res = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        if res and res.text:
            return res.text.strip()
    except Exception:
        pass

    return laudo_base


@app.get("/health", status_code=status.HTTP_200_OK)
def checar_status():
    return {
        "status": "online",
        "modelo_carregado": modelo_svm is not None,
        "scaler_carregado": scaler is not None,
        "gemini_disponivel": gemini_client is not None
    }


@app.post("/predict", response_model=DiagnosticoSaida, status_code=status.HTTP_200_OK)
def predizer(payload: TelemetriaEntrada):
    if modelo_svm is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço indisponível: artefatos não carregados."
        )

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

    laudo = obter_parecer(payload, metricas, distancia, falha)

    return DiagnosticoSaida(
        alerta_falha=falha,
        classe_predita=classe,
        distancia_hiperplano=round(distancia, 4),
        estado_ativo="Alerta de Falha Detectado" if falha else "Operação Normal",
        metricas_engenharia=metricas,
        parecer_tecnico_gemini=laudo
    )
