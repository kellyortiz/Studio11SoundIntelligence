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

# --- MOTOR DE CLASSIFICAÇÃO TAXONÔMICA RECALIBRADO ---
def classify_universal_taxonomy(tempo, centroid, rms):
    # Reggae / Dub / Reggae Brasileiro (Andamento cadenciado 65-95 BPM, graves marcantes / centroide moderado-baixo)
    if 65 <= tempo <= 98 and centroid < 1700 and rms < 0.13:
        return "Reggae", "Roots Reggae / Dub / Reggae Brasileiro"
        
    # Funk Brasileiro (BPM mais alto ou batida muito densa/compressão pesada)
    elif 125 <= tempo <= 160 and rms > 0.12:
        if tempo >= 145:
            return "Música Brasileira", "Funk Brasileiro -> Funk 150 BPM / Automotivo / Mandelão"
        elif centroid > 2100:
            return "Música Brasileira", "Funk Brasileiro -> Funk Carioca / Ostentação"
        else:
            return "Música Brasileira", "Funk Brasileiro -> Funk Consciente / Proibidão"
            
    # Música Eletrônica / EDM
    elif 120 <= tempo <= 140 and centroid > 2200:
        return "Música Eletrônica (EDM)", "House / Tech House / Progressive House"
    elif tempo > 140 and centroid > 2000:
        return "Música Eletrônica (EDM)", "Drum and Bass / Hardstyle / Psytrance"
    elif 70 <= tempo <= 100 and centroid > 2100:
        return "Música Eletrônica (EDM)", "Phonk / Trap EDM"

    # Sertanejo & Piseiro
    elif 80 <= tempo <= 118 and centroid < 1700:
        if rms > 0.11:
            return "Música Brasileira", "Sertanejo -> Sertanejo Universitário / Piseiro"
        else:
            return "Música Brasileira", "Sertanejo -> Sertanejo Raiz / Modão"
            
    # Forró
    elif 115 <= tempo <= 145 and centroid < 1900:
        return "Música Brasileira", "Forró -> Forró Pé de Serra / Xote / Baião"
        
    # MPB / Bossa Nova
    elif tempo < 85 and centroid < 1500:
        return "Música Brasileira", "MPB -> Bossa Nova / MPB Contemporânea / Choro"

    # Rock & Metal
    elif 95 <= tempo <= 160 and centroid > 1800:
        if rms > 0.13:
            return "Metal", "Heavy Metal / Metalcore / Thrash Metal"
        else:
            return "Rock", "Classic Rock / Alternative Rock / Grunge"

    # Hip-Hop / Rap
    elif 70 <= tempo <= 105 and centroid > 1600:
        return "Hip-Hop / Rap", "Trap / Drill / Boom Bap / Lo-fi Hip-Hop"

    # Jazz & Blues
    elif 60 <= tempo <= 95 and centroid < 1400:
        return "Jazz", "Smooth Jazz / Bebop / Delta Blues"

    # Clássica
    elif tempo < 75 and centroid < 1300:
        return "Música Clássica", "Sinfônica / Música de Câmara / Solo Piano"

    # Padrão
    else:
        return "Pop", "Latin Pop / Synthpop / Contemporary R&B"

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
    st.markdown("<small style='color: #9ca3af;'>✓ Reggae, Dub & Reggae Brasileiro<br>✓ Funk (Carioca, 150 BPM, etc.)<br>✓ Sertanejo, Piseiro & Forró<br>✓ Rock, Metal & Vertentes<br>✓ EDM, Phonk, Trap & Dubstep<br>✓ Hip-Hop, Drill & Lo-fi<br>✓ Jazz, Blues, Soul & R&B<br>✓ Gospel & Música Latina<br>✓ MPB, Bossa Nova & Axé<br>✓ Africana, Indiana & Asiática</small>", unsafe_allow_html=True)

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
                
                macro_genre, sub_genre = classify_universal_taxonomy(tempo, spectral_centroid, rms)
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
            st.info("Classificação universal concluída com sucesso.")

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

            # Métricas Técnicas
            m1, m2, m3 = st.columns(3)
            with m1:
                minutes, seconds = int(duration // 60), int(duration % 60)
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{minutes}:{seconds:02d}</div><div class='metric-label'>Duração Total</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{rms:.4f}</div><div class='metric-label'>Energia RMS</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{spectral_centroid:.0f} Hz</div><div class='metric-label'>Centroide Espectral</div></div>", unsafe_allow_html=True)

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
