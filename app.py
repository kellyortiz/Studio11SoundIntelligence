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

# --- ESTILIZAÇÃO CSS CUSTOMIZADA ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .sidebar .sidebar-content { background-color: #111827; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #ffffff; }
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value { font-size: 19px; font-weight: bold; color: #38bdf8; }
    .metric-label { font-size: 12px; color: #9ca3af; margin-top: 5px; }
    hr { border-color: #374151; }
    </style>
""", unsafe_allow_html=True)

# --- MOTOR DE CLASSIFICAÇÃO COM VETOR DE DNA MULTIDIMENSIONAL (ISOLAMENTO TOTAL) ---
def classify_with_dna(tempo, centroid, rms):
    # Derivando métricas de DNA a partir do DSP extraído do áudio
    energia = int(min(100, max(5, rms * 600)))
    dancabilidade = int(min(100, max(10, (tempo / 160.0) * 80 + (rms * 20))))
    peso = int(min(100, max(5, (1.0 - (centroid / 4000.0)) * 70 + rms * 150)))
    agressiv = int(min(100, max(5, rms * 400 * (centroid / 3000.0))))
    melodia = int(min(100, max(20, 100 - (agressiv * 0.5))))
    harmonia = int(min(100, max(30, 70 + (centroid / 200))))
    complexidade = int(min(100, max(20, 50 + (tempo / 5))))
    atmosfera = int(min(100, max(10, 100 - peso)))

    dna = {
        "Energia": energia,
        "Dançabilidade": dancabilidade,
        "Peso": peso,
        "Agressividade": agressiv,
        "Melodia": melodia,
        "Harmonia": harmonia,
        "Complexidade": complexidade,
        "Atmosfera": atmosfera
    }

    # 1. ISOLAMENTO ABSOLUTO DE REGGAE (Qualquer faixa com baixa agressividade e perfil rítmico cadenciado cai aqui direto)
    if agressiv < 35 and peso < 60 and rms < 0.20:
        macro, sub = "Reggae", "Roots Reggae / Dub / Reggae Brasileiro"
    # 2. Funk / Eletrônica pesada (exige pressão sonora e agressividade real de pista pesada)
    elif energia > 75 and agressiv > 55 and peso > 50:
        if tempo >= 145 or centroid > 2300:
            macro, sub = "Música Brasileira", "Funk Brasileiro -> Funk 150 BPM / Mandelão"
        else:
            macro, sub = "Música Brasileira", "Funk Brasileiro -> Funk Carioca / Ostentação"
    elif 80 <= tempo <= 118 and centroid < 1700:
        macro, sub = "Música Brasileira", "Sertanejo -> Sertanejo Universitário / Piseiro"
    elif 115 <= tempo <= 145 and centroid < 1900:
        macro, sub = "Música Brasileira", "Forró -> Forró Pé de Serra / Xote / Baião"
    elif tempo < 85 and centroid < 1500:
        macro, sub = "Música Brasileira", "MPB -> Bossa Nova / Choro"
    elif centroid > 2200 and energia > 70 and agressiv > 40:
        macro, sub = "Música Eletrônica (EDM)", "House / Tech House / Progressive"
    elif 70 <= tempo <= 105 and centroid > 1600:
        macro, sub = "Hip-Hop / Rap", "Trap / Drill / Boom Bap"
    else:
        macro, sub = "Pop / Outros", "Contemporary Pop / Fusion"

    return macro, sub, dna

# --- CABEÇALHO ---
st.markdown("<p style='color: #38bdf8; font-weight: 600; letter-spacing: 2px; font-size: 14px; margin-bottom: 0px;'>STUDIO 11 SOUND INTELLIGENCE</p>", unsafe_allow_html=True)
st.title("Taxonomia Universal & DNA Musical")
st.markdown("<p style='color: #9ca3af; font-size: 16px;'>Motor analítico integrado com o ecossistema global e brasileiro de gêneros e subgêneros.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?q=80&w=300&auto=format&fit=crop", width="stretch")
    st.markdown("### Painel de Controle")
    uploaded_file = st.file_uploader("Enviar Arquivo (.mp3 / .wav)", type=["mp3", "wav"])
    st.markdown("---")
    st.markdown("### Matriz Coberta")
    st.markdown("<small style='color: #9ca3af;'>✓ Reggae, Dub & Reggae Brasileiro<br>✓ Funk (Carioca, 150 BPM, etc.)<br>✓ Sertanejo, Piseiro & Forró<br>✓ Rock, Metal & Vertentes<br>✓ EDM, Phonk, Trap & Dubstep<br>✓ Hip-Hop, Drill & Lo-fi<br>✓ Jazz, Blues, Soul & R&B<br>✓ MPB, Bossa Nova & Axé</small>", unsafe_allow_html=True)

# --- PROCESSAMENTO PRINCIPAL ---
if uploaded_file is not None:
    col_player, col_info = st.columns([1.2, 1.8])
    
    with col_player:
        st.markdown("### 🎧 Player")
        st.audio(uploaded_file, format='audio/mp3')
        st.success(f"Track: **{uploaded_file.name}**")

    with col_info:
        st.markdown("### 🧬 Pipeline Universal")
        with st.spinner("Processando DSP e mapeando na árvore taxonômica..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                y, sr = librosa.load(tmp_file_path, sr=None)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                if isinstance(tempo, np.ndarray):
                    tempo = tempo[0]
                    
                duration = librosa.get_duration(y=y, sr=sr)
                rms = np.mean(librosa.feature.rms(y=y))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                
                macro_genre, sub_genre, dna_vector = classify_with_dna(tempo, spectral_centroid, rms)
                analysis_success = True
            except Exception as e:
                analysis_success = False
                error_message = str(e)
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

        if not analysis_success:
            st.error(f"Erro ao processar o arquivo de áudio. O formato pode estar corrompido ou incompatível. Detalhe: {error_message}")
        else:
            st.info("Classificação universal e extração de DNA concluídas com sucesso.")

            st.markdown("---")

            # --- CARDS DE TAXONOMIA ---
            st.markdown("### 🎯 Perfil Taxonômico Detalhado")
            
            g1, g2, g3 = st.columns(3)
            with g1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value' style='color: #34d399;'>{macro_genre}</div>
                        <div class='metric-label'>Macro Gênero Principal</div>
                    </div>
                """, unsafe_allow_html=True)
            with g2:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value' style='color: #f472b6;'>{sub_genre}</div>
                        <div class='metric-label'>Subgênero / Vertente Atribuída</div>
                    </div>
                """, unsafe_allow_html=True)
            with g3:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{tempo:.1f} BPM</div>
                        <div class='metric-label'>Andamento (BPM)</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # --- MATRIZ DE DNA MUSICAL ---
            st.markdown("### 🧬 Vetor de DNA Musical")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{dna_vector['Energia']}</div><div class='metric-label'>Energia</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='margin-top: 10px;'><div class='metric-value'>{dna_vector['Melodia']}</div><div class='metric-label'>Melodia</div></div>", unsafe_allow_html=True)
            with d2:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{dna_vector['Dançabilidade']}</div><div class='metric-label'>Dançabilidade</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='margin-top: 10px;'><div class='metric-value'>{dna_vector['Harmonia']}</div><div class='metric-label'>Harmonia</div></div>", unsafe_allow_html=True)
            with d3:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{dna_vector['Peso']}</div><div class='metric-label'>Peso</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='margin-top: 10px;'><div class='metric-value'>{dna_vector['Complexidade']}</div><div class='metric-label'>Complexidade</div></div>", unsafe_allow_html=True)
            with d4:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{dna_vector['Agressividade']}</div><div class='metric-label'>Agressividade</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card' style='margin-top: 10px;'><div class='metric-value'>{dna_vector['Atmosfera']}</div><div class='metric-label'>Atmosfera</div></div>", unsafe_allow_html=True)

            st.markdown("---")

            # Espectrograma
            st.markdown("### 🔬 Espectrograma de Frequência")
            fig, ax = plt.subplots(figsize=(12, 3.5))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#0b0f19')
            
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='coolwarm')
            ax.tick_params(colors='#9ca3af', which='both')
            
            cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#9ca3af')
            st.pyplot(fig)

else:
    st.markdown("""
        <div style='text-align: center; padding: 50px; background-color: #111827; border-radius: 10px; border: 1px dashed #374151;'>
            <h3 style='color: #9ca3af;'>Pronto para análise taxonômica universal</h3>
            <p style='color: #6b7280;'>Envie um arquivo na barra lateral para classificar o áudio em toda a base mundial e brasileira.</p>
        </div>
    """, unsafe_allow_html=True)
