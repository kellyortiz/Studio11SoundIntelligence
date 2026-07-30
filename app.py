import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuração da Página
st.set_page_config(
    page_title="Studio 11 | Sound Intelligence MVP",
    page_icon="🎵",
    layout="wide"
)

# Estilização visual minimalista
st.title("🎧 Studio 11 — Sound Intelligence")
st.markdown("### Extração de DNA Musical e Explicabilidade de IA para Engenharia de Som")
st.markdown("---")

# Sidebar para instruções
with st.sidebar:
    st.header("Painel de Controle")
    st.info("Fase: MVP Bootstrap (Zero-Cost)")
    st.markdown("**Tecnologias:**")
    st.markdown("- Python & FastAPI / Streamlit")
    st.markdown("- Librosa (DSP)")
    st.markdown("- Modelos CLAP & AST (IA)")
    st.markdown("---")
    st.markdown("Desenvolvido por **Kelly Ortiz** & Wellington Marcondes")

# Upload do Arquivo de Áudio
uploaded_file = st.file_uploader("Envie seu arquivo de áudio (MP3 ou WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # Exibir player de áudio
    st.audio(uploaded_file, format='audio/mp3')
    
    with st.spinner("Processando DSP e extraindo DNA Musical..."):
        # Carregar o áudio com librosa
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 1. Métricas Básicas de DSP
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Lidar com o retorno do tempo dependendo da versão do librosa
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
            
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Layout em colunas para os resultados
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Relatório Técnico (DSP)")
        st.metric(label="BPM Estimado", value=f"{round(tempo, 1)}")
        st.metric(label="Duração", value=f"{round(duration, 2)} segundos")
        st.metric(label="Energia Média (RMS)", value=f"{round(float(rms), 4)}")
        st.metric(label="Centroide Espectral (Brilho)", value=f"{round(float(spectral_centroid), 2)} Hz")

    with col2:
        st.subheader("🧬 DNA Musical (% de Estilos / Textura)")
        # Simulação dos pesos percentuais retornados pelo modelo CLAP/AST
        # Em produção, aqui entra a inferência real do modelo de IA
        dna_data = pd.DataFrame({
            'Atributo / Estilo': ['Eletrônica / Synth', 'Orgânico / Acústico', 'Densidade Harmônica', 'Dinâmica / Transientes', 'Presença de Vocais'],
            'Porcentagem (%)': [68.5, 31.5, 82.0, 74.3, 45.0]
        })
        st.dataframe(dna_data, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Análise Visual da Caixa-Preta (Espectrograma)")
    
    # Gerando o gráfico do Espectrograma
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='coolwarm')
    ax.set(title='Espectrograma de Frequência - Mapeamento Studio 11')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    
    st.pyplot(fig)

    st.success("Análise concluída com sucesso! Relatório pronto para validação com Design Partners.")

else:
    st.warning("Por favor, faça o upload de um arquivo de áudio para iniciar a análise.")