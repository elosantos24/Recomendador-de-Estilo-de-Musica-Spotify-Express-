"""
Recomendador de Estilos Musicais por Humor - Streamlit + TensorFlow
-------------------------------------------------------------------
O usuário informa dois indicadores (energia e tristeza) e o que deseja
sentir (nostalgia, animar o astral etc.). A aplicação recomenda o estilo
musical mais compatível (Rock, Lofi, Pop ou Sertanejo) e uma música de
um artista ou banda que combina com o momento.

Como funciona (visão geral):
1. Entradas: energia e tristeza (0 a 100), normalizadas para 0 a 1.
2. Uma rede neural pequena aprende a relação entre humor e estilo.
3. A saída (softmax) é a probabilidade de cada estilo; o maior vence.
4. Com o estilo definido, a intenção escolhida seleciona a música
   em um catálogo curado (CATALOGO).

Executar localmente:  streamlit run app.py
"""

from urllib.parse import quote

import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Humor & Música",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 2. IDENTIDADE VISUAL (inspirada no Spotify: fundo escuro + verde #1DB954)
# ---------------------------------------------------------------------------
VERDE = "#1DB954"

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: #121212; color: #FFFFFF; }}
        [data-testid="stSidebar"] {{ background-color: #000000; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        h1, h2, h3, label, p, span {{ color: #FFFFFF; }}
        /* Controle deslizante na cor da marca */
        div[data-testid="stSlider"] [role="slider"] {{ background-color: {VERDE}; }}
        .cartao {{
            background: linear-gradient(135deg, {VERDE} 0%, #0d7a37 100%);
            border-radius: 16px; padding: 32px; margin-bottom: 24px;
        }}
        .cartao h1 {{ margin: 0; font-size: 3rem; color: #FFFFFF; }}
        .cartao p {{ margin: 8px 0 0 0; font-size: 1.1rem; color: #F2F2F2; }}
        .selo {{ font-size: .8rem; letter-spacing: .12em; text-transform: uppercase;
                 color: #E8FFF0; font-weight: 700; }}
        .linha {{ margin-bottom: 14px; }}
        .linha .nome {{ display: flex; justify-content: space-between;
                        font-size: .95rem; margin-bottom: 4px; color: #FFFFFF; }}
        .trilho {{ background: #2A2A2A; border-radius: 999px; height: 10px; }}
        .barra {{ background: {VERDE}; border-radius: 999px; height: 10px; }}
        .barra.fraca {{ background: #535353; }}
        .nota {{ color: #B3B3B3; font-size: .85rem; }}
        .musica {{
            background: #181818; border: 1px solid #2A2A2A; border-left: 6px solid {VERDE};
            border-radius: 12px; padding: 20px 24px; margin: 8px 0 16px 0;
        }}
        .musica .rotulo {{ color: {VERDE}; font-size: .75rem; letter-spacing: .12em;
                           text-transform: uppercase; font-weight: 700; }}
        .musica .titulo {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
        .musica .artista {{ color: #B3B3B3; font-size: 1.05rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 3. CATÁLOGO DE ESTILOS
# ---------------------------------------------------------------------------
ESTILOS = {
    "Rock": {
        "emoji": "🎸",
        "descricao": "Energia alta com emoção intensa. Guitarras e bateria para colocar tudo para fora.",
        "busca": "rock",
    },
    "Lofi": {
        "emoji": "☕",
        "descricao": "Ritmo calmo e ambiente suave. Ideal para desacelerar, estudar ou relaxar.",
        "busca": "lofi",
    },
    "Pop": {
        "emoji": "✨",
        "descricao": "Energia alta e astral leve. Refrões marcantes para manter o bom humor.",
        "busca": "pop",
    },
    "Sertanejo": {
        "emoji": "🤠",
        "descricao": "Letras emotivas e melodias envolventes. Combina com momentos de sentimento à flor da pele.",
        "busca": "sertanejo",
    },
}
NOMES = list(ESTILOS.keys())

# ---------------------------------------------------------------------------
# 3.1 INTENÇÕES E CATÁLOGO DE MÚSICAS
# A intenção responde: "o que você quer sentir agora?".
# CATALOGO[estilo][intenção] = lista de (artista/banda, música).
# Para ampliar o catálogo, basta adicionar novos pares nas listas.
# ---------------------------------------------------------------------------
INTENCOES = {
    "Nostalgia": "🕰️ Nostalgia",
    "Animar": "🎉 Animar o astral",
    "Relaxar": "🌿 Relaxar",
    "Desabafar": "💭 Desabafar",
}

CATALOGO = {
    "Rock": {
        "Nostalgia": [("Journey", "Don't Stop Believin'"), ("Legião Urbana", "Pais e Filhos")],
        "Animar": [("Queen", "Don't Stop Me Now"), ("AC/DC", "Thunderstruck")],
        "Relaxar": [("Pink Floyd", "Wish You Were Here"), ("Oasis", "Champagne Supernova")],
        "Desabafar": [("Linkin Park", "Numb"), ("Pitty", "Na Sua Estante")],
    },
    "Lofi": {
        "Nostalgia": [("Nujabes", "Aruarian Dance"), ("Tomppabeats", "Monday Loop")],
        "Animar": [("Bonobo", "Kerala"), ("Tycho", "A Walk")],
        "Relaxar": [("Jinsang", "Affection"), ("Nujabes", "Feather")],
        "Desabafar": [("Nujabes", "Luv(sic) Pt. 3"), ("Tomppabeats", "Monday Loop")],
    },
    "Pop": {
        "Nostalgia": [("Michael Jackson", "Billie Jean"), ("ABBA", "Dancing Queen")],
        "Animar": [("Dua Lipa", "Levitating"), ("Pharrell Williams", "Happy")],
        "Relaxar": [("Billie Eilish", "Ocean Eyes"), ("Ed Sheeran", "Photograph")],
        "Desabafar": [("Adele", "Someone Like You"), ("Sam Smith", "Stay With Me")],
    },
    "Sertanejo": {
        "Nostalgia": [("Chitãozinho & Xororó", "Evidências"), ("Leandro & Leonardo", "Pense em Mim")],
        "Animar": [("Michel Teló", "Ai Se Eu Te Pego"), ("Gusttavo Lima", "Balada")],
        "Relaxar": [("Zezé Di Camargo & Luciano", "É o Amor"), ("Victor & Leo", "Borboletas")],
        "Desabafar": [("Marília Mendonça", "Infiel"), ("Henrique & Juliano", "Flor e o Beija-Flor")],
    },
}

# Perfil emocional típico de cada estilo: (energia, tristeza) em escala 0-100.
# ATENÇÃO: são premissas de negócio, não dados medidos de usuários reais.
PERFIS = {
    "Rock": (85, 65),
    "Lofi": (20, 30),
    "Pop": (75, 15),
    "Sertanejo": (45, 85),
}

SEMENTE = 42
AMOSTRAS_POR_ESTILO = 500
DISPERSAO = 12  # quanto os usuários variam em torno do perfil típico


# ---------------------------------------------------------------------------
# 4. DADOS E MODELO (treinamento com cache)
# O @st.cache_resource treina o modelo UMA vez e o reutiliza a cada interação.
# ---------------------------------------------------------------------------
def gerar_dados():
    """Gera exemplos sintéticos ao redor do perfil de cada estilo."""
    rng = np.random.default_rng(SEMENTE)
    entradas, rotulos = [], []
    for indice, nome in enumerate(NOMES):
        centro = np.array(PERFIS[nome], dtype="float32")
        pontos = rng.normal(centro, DISPERSAO, size=(AMOSTRAS_POR_ESTILO, 2))
        entradas.append(np.clip(pontos, 0, 100))
        rotulos.append(np.full(AMOSTRAS_POR_ESTILO, indice))
    x = np.concatenate(entradas) / 100.0  # normaliza para 0-1
    y = np.concatenate(rotulos)
    ordem = rng.permutation(len(x))
    return x[ordem].astype("float32"), y[ordem]


@st.cache_resource(show_spinner="Preparando o modelo de recomendação...")
def treinar_modelo():
    """Treina a rede neural e devolve (modelo, acurácia de validação)."""
    tf.keras.utils.set_random_seed(SEMENTE)
    x, y = gerar_dados()

    modelo = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),  # energia e tristeza
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            # Uma saída por estilo; softmax faz as probabilidades somarem 100%.
            tf.keras.layers.Dense(len(NOMES), activation="softmax"),
        ]
    )
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    historico = modelo.fit(
        x, y, epochs=40, batch_size=32, validation_split=0.2, verbose=0
    )
    return modelo, float(historico.history["val_accuracy"][-1])


def recomendar(modelo, energia: int, tristeza: int) -> np.ndarray:
    """Devolve a probabilidade de cada estilo para o humor informado."""
    entrada = np.array([[energia / 100.0, tristeza / 100.0]], dtype="float32")
    return modelo(entrada, training=False).numpy()[0]


def descrever_humor(energia: int, tristeza: int) -> str:
    """Resumo em linguagem natural do estado emocional informado."""
    nivel_e = "alta" if energia >= 66 else "média" if energia >= 34 else "baixa"
    nivel_t = "alta" if tristeza >= 66 else "média" if tristeza >= 34 else "baixa"
    return f"Energia {nivel_e} · Tristeza {nivel_t}"


modelo, acuracia = treinar_modelo()

# ---------------------------------------------------------------------------
# 5. BARRA LATERAL (entradas do usuário)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎧 Humor & Música")
    st.caption("Conte como você está e descubra o que ouvir agora.")
    st.divider()

    energia = st.slider("⚡ Nível de energia", 0, 100, 50, help="0 = sem energia, 100 = muita energia")
    tristeza = st.slider("🌧️ Nível de tristeza", 0, 100, 30, help="0 = nenhuma, 100 = muito triste")

    # Pergunta de intenção: define qual música do estilo será sugerida.
    intencao = st.radio(
        "🎯 O que você quer agora?",
        options=list(INTENCOES.keys()),
        format_func=lambda chave: INTENCOES[chave],
    )

    st.divider()
    st.caption(f"Acurácia do modelo (validação): {acuracia:.0%}")
    st.caption("Modelo treinado com dados simulados a partir de perfis emocionais por estilo.")

# ---------------------------------------------------------------------------
# 6. ÁREA PRINCIPAL (resultado)
# ---------------------------------------------------------------------------
probabilidades = recomendar(modelo, energia, tristeza)
vencedor = int(np.argmax(probabilidades))
estilo = NOMES[vencedor]
info = ESTILOS[estilo]

st.markdown(
    f"""
    <div class="cartao">
        <div class="selo">Recomendação para o seu momento</div>
        <h1>{info['emoji']} {estilo}</h1>
        <p>{info['descricao']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

coluna_a, coluna_b = st.columns([3, 2], gap="large")

with coluna_a:
    st.markdown("### Compatibilidade por estilo")
    ordem = np.argsort(probabilidades)[::-1]
    for i in ordem:
        pct = probabilidades[i] * 100
        classe = "barra" if i == vencedor else "barra fraca"
        st.markdown(
            f"""
            <div class="linha">
                <div class="nome"><span>{ESTILOS[NOMES[i]]['emoji']} {NOMES[i]}</span><span>{pct:.0f}%</span></div>
                <div class="trilho"><div class="{classe}" style="width:{pct:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with coluna_b:
    st.markdown("### Seu momento")
    st.metric("Energia", f"{energia}/100")
    st.metric("Tristeza", f"{tristeza}/100")
    st.caption(descrever_humor(energia, tristeza))


# --- Sugestão de música -----------------------------------------------------
# O contador permite alternar entre as músicas disponíveis para o mesmo
# estilo e intenção. O callback roda ANTES da página ser redesenhada.
if "rodada" not in st.session_state:
    st.session_state.rodada = 0


def proxima_sugestao():
    st.session_state.rodada += 1


sugestoes = CATALOGO[estilo][intencao]
artista, musica = sugestoes[st.session_state.rodada % len(sugestoes)]

st.markdown(
    f"""
    <div class="musica">
        <div class="rotulo">{INTENCOES[intencao]} · Música sugerida</div>
        <div class="titulo">{musica}</div>
        <div class="artista">{artista}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

botao_a, botao_b, _ = st.columns([2, 2, 3])
botao_a.link_button(
    "Ouvir no Spotify",
    f"https://open.spotify.com/search/{quote(f'{musica} {artista}')}",
    type="primary",
    use_container_width=True,
)
botao_b.button("Outra sugestão", on_click=proxima_sugestao, use_container_width=True)

st.markdown(
    '<p class="nota">Recomendação gerada por modelo de IA a partir dos indicadores informados. '
    "Serve como inspiração e não substitui o seu gosto pessoal.</p>",
    unsafe_allow_html=True,
)
