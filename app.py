import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Studio 11 | Sound Intelligence",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CSS CUSTOMIZADA (DARK THEME CORPORATIVO) ---
st.markdown("""
    <style>
    /* Fundo principal e fontes */
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .sidebar .sidebar-content {
        background-color: #111827;
    }
    /* Estilização de títulos */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    /* Cards de métricas modernos */
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
        margin-top: 5px;
    }
    /* Divisor estilizado */
    hr {
        border-color: #374151;
    }
    </style>
""", unsafe_allow_html=True)

# --- CABEÇALHO DA MARCA ---
st.markdown("<p style='color: #38bdf8; font-weight: 600; letter-spacing: 2px; font-size: 14px; margin-bottom: 0px;'>STUDIO 11 SOUND INTELLIGENCE</p>", unsafe_allow_html=True)
st.title("Extração de DNA Musical & Insights DSP")
st.markdown("<p style='color: #9ca3af; font-size: 16px;'>Plataforma B2B de inteligência de áudio para estúdios, gravadoras e curadores musicais.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- BARRA LATERAL (CONTROLES E INPUTS) ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?q=80&w=300&auto=format&fit=crop", use_container_width=True)
    st.markdown("### Configurações de Análise")
    st.info("Envie um arquivo de áudio nos formatos suportados para iniciar o pipeline de extração de características acústicas.")
    
    uploaded_file = st.file_uploader("Arquivo de Áudio (.mp3 / .wav)", type=["mp3", "wav"])
    
    st.markdown("---")
    st.markdown("### Sobre o Projeto")
    st.markdown("<small style='color: #9ca3af;'><b>Tech Lead:</b> Kelly Ortiz<br><b>Parceiro:</b> Wellington Marcondes<br><b>Versão:</b> MVP 1.0 (DSP-First)</small>", unsafe_allow_html=True)

# --- CORPO PRINCIPAL DA APLICAÇÃO ---
if uploaded_file is not None:
    # Layout em colunas para player e informações iniciais
    col_player, col_info = st.columns([1.2, 1.8])
    
    with col_player:
        st.markdown("### 🎧 Reprodução")
        st.audio(uploaded_file, format='audio/mp3')
        st.success(f"Arquivo carregado: **{uploaded_file.name}**")

    with col_info:
        st.markdown("### 📊 Status do Pipeline")
        with st.spinner("Executando motor DSP e decompondo o DNA Musical..."):
            # Salvamento seguro em arquivo temporário para evitar LibsndfileError
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                y, sr = librosa.load(tmp_file_path, sr=None)
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
            
            # Extração de Métricas Reais via Librosa
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0]
                
            duration = librosa.get_duration(y=y, sr=sr)
            rms = np.mean(librosa.feature.rms(y=y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
        st.info("Processamento concluído com sucesso. Métricas extraídas do buffer de áudio bruto.")

    st.markdown("---")

    # --- MÉTRICAS PRINCIPAIS (CARDS ESTILIZADOS) ---
    st.markdown("### 🧬 Métricas de DNA Acústico")
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{tempo:.1f}</div>
                <div class='metric-label'>BPM Estimado</div>
            </div>
        """, unsafe_allow_html=True)
        
    with m2:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{minutes}:{seconds:02d}</div>
                <div class='metric-label'>Duração Total</div>
            </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{rms:.4f}</div>
                <div class='metric-label'>Energia RMS Média</div>
            </div>
        """, unsafe_allow_html=True)
        
    with m4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{spectral_centroid:.0f} Hz</div>
                <div class='metric-label'>Centroide Espectral</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SEÇÃO DE VISUALIZAÇÕES E ESPECTROGRAMA ---
    st.markdown("### 🔬 Espectrograma de Frequência")
    st.markdown("<p style='color: #9ca3af;'>Representação visual em escala logarítmica da distribuição de energia ao longo do tempo.</p>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#0b0f19')
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='coolwarm')
    
    ax.label_outer()
    ax.tick_params(colors='#9ca3af', which='both')
    ax.xaxis.label.set_color('#f3f4f6')
    ax.yaxis.label.set_color('#f3f4f6')
    
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#9ca3af')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#9ca3af')
    
    st.pyplot(fig)

else:
    # Estado inicial (vazio)
    st.markdown("""
        <div style='text-align: center; padding: 50px; background-color: #111827; border-radius: 10px; border: 1px dashed #374151;'>
            <h3 style='color: #9ca3af;'>Nenhum arquivo selecionado</h3>
            <p style='color: #6b7280;'>Utilize a barra lateral à esquerda para enviar uma faixa de áudio e gerar os relatórios de inteligência.</p>
        </div>
    """, unsafe_allow_html=True)
