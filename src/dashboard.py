"""
Interface Visual de Manutenção Preditiva com Diagnóstico Gemini
Framework: Streamlit
Consumo: API FastAPI (http://127.0.0.1:8000)
"""

import streamlit as st
import requests

st.set_page_config(
    page_title="Monitor de Manutenção Preditiva - SVM & Gemini",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Painel Operacional de Manutenção Preditiva (SVM RBF + Gemini)")
st.markdown("Simule os parâmetros operacionais dos sensores para avaliar o risco de falha mecânica/térmica em tempo real.")

st.sidebar.header("Conexão com a API")
api_url = st.sidebar.text_input("URL da API:", value="http://127.0.0.1:8000")

try:
    health_resp = requests.get(f"{api_url}/health", timeout=3)
    if health_resp.status_code == 200:
        dados_health = health_resp.json()
        status_gemini = "✓ Gemini Conectado" if dados_health.get("gemini_disponivel") else "⚠️ Gemini Sem Chave"
        st.sidebar.success(f"API Online | {status_gemini}")
    else:
        st.sidebar.warning("API respondeu com código de erro.")
except Exception:
    st.sidebar.error("API Offline. Certifique-se de que o Uvicorn está em execução na porta 8000.")

st.sidebar.markdown("---")
st.sidebar.subheader("Presets de Teste")
preset = st.sidebar.selectbox(
    "Carregar Condição Típica:",
    ["Personalizado", "Operação Segura / Estável", "Sobrecarga Crítica (Falha)"]
)

if preset == "Operação Segura / Estável":
    p_air, p_proc, p_rpm, p_torque, p_wear, p_type = 298.1, 308.6, 1551.0, 42.8, 5.0, "M"
elif preset == "Sobrecarga Crítica (Falha)":
    p_air, p_proc, p_rpm, p_torque, p_wear, p_type = 302.5, 312.0, 1300.0, 75.0, 230.0, "L"
else:
    p_air, p_proc, p_rpm, p_torque, p_wear, p_type = 299.0, 309.0, 1500.0, 40.0, 50.0, "M"

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Variáveis Térmicas")
    air_temp = st.slider("Temperatura do Ar [K]", min_value=280.0, max_value=320.0, value=float(p_air), step=0.1)
    proc_temp = st.slider("Temperatura do Processo [K]", min_value=290.0, max_value=330.0, value=float(p_proc), step=0.1)
    tipo_produto = st.selectbox("Variante do Produto", ["L", "M", "H"], index=["L", "M", "H"].index(p_type))

with col2:
    st.subheader("🔧 Variáveis Mecânicas")
    rot_speed = st.slider("Velocidade Rotacional [RPM]", min_value=1000.0, max_value=3000.0, value=float(p_rpm), step=10.0)
    torque = st.slider("Torque [Nm]", min_value=10.0, max_value=100.0, value=float(p_torque), step=0.5)
    tool_wear = st.slider("Desgaste da Ferramenta [min]", min_value=0.0, max_value=250.0, value=float(p_wear), step=1.0)

st.markdown("---")

if st.button("Executar Diagnóstico em Tempo Real", use_container_width=True):
    payload = {
        "air_temperature_k": air_temp,
        "process_temperature_k": proc_temp,
        "rotational_speed_rpm": rot_speed,
        "torque_nm": torque,
        "tool_wear_min": tool_wear,
        "tipo_produto": tipo_produto
    }

    try:
        with st.spinner("A computar projeção no hiperplano e a consultar o Gemini..."):
            response = requests.post(f"{api_url}/predict", json=payload, timeout=60)
        
        if response.status_code == 200:
            resultado = response.json()
            
            res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])
            
            with res_col1:
                if resultado["alerta_falha"]:
                    st.error(f"🚨 **{resultado['estado_ativo']}**")
                else:
                    st.success(f"✅ **{resultado['estado_ativo']}**")
                    
                st.metric(
                    label="Distância ao Hiperplano SVM",
                    value=f"{resultado['distancia_hiperplano']:.4f}",
                    delta="Região de Falha (> 0)" if resultado["distancia_hiperplano"] > 0 else "Região Segura (< 0)",
                    delta_color="inverse"
                )

            with res_col2:
                st.metric(
                    label="Diferencial Térmico (ΔT)",
                    value=f"{resultado['metricas_engenharia']['delta_T_Kelvin']} K"
                )
                st.metric(
                    label="Potência Mecânica",
                    value=f"{resultado['metricas_engenharia']['potencia_kW']} kW"
                )

            with res_col3:
                st.metric(
                    label="Esforço de Desgaste (τ × t)",
                    value=f"{resultado['metricas_engenharia']['esforco_desgaste']} Nm·min"
                )

            if resultado.get("parecer_tecnico_gemini"):
                st.markdown("---")
                st.subheader("🤖 Parecer Pericial de Manutenção (Gemini Engine)")
                st.info(resultado["parecer_tecnico_gemini"])
        else:
            st.error(f"Erro na resposta da API ({response.status_code}): {response.text}")
    except Exception as e:
        st.error(f"Não foi possível estabelecer ligação à API: {e}")