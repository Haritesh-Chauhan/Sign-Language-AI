import cv2
import joblib
import numpy as np
import streamlit as st
import tempfile
import time
import random
import string
from collections import Counter, deque
from keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from transformers import T5Tokenizer, T5ForConditionalGeneration

FRAME_COUNT = 25
MIN_CONFIDENCE = 0.25
SIGN_MODEL_FILE = "sign_language_model_100.keras"
LABEL_FILE = "label_encoder.pkl"
HAND_MODEL_FILE = "hand_landmarker.task"
SENTENCE_MODEL_NAME = "google/flan-t5-base"

st.set_page_config(
    page_title="Neural Sign — Motion Translation Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg: #08080c;
        --surface: #121218;
        --surface-2: #17171f;
        --line: #23232e;
        --ink: #f2f2f5;
        --muted: #85859a;
        --cyan: #4fd8ff;
        --violet: #9b7bff;
        --amber: #ffb454;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--ink);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background-image:
            radial-gradient(circle at 15% 10%, rgba(79, 216, 255, 0.10), transparent 40%),
            radial-gradient(circle at 85% 0%, rgba(155, 123, 255, 0.10), transparent 40%),
            radial-gradient(circle 1.5px at 20px 20px, rgba(255,255,255,0.05) 1.5px, transparent 0);
        background-size: auto, auto, 34px 34px;
    }

    [data-testid="collapsedControl"] { display: none; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ---------- Hero ---------- */
    .hero-wrap {
        text-align: center;
        padding-top: 3rem;
        padding-bottom: 1rem;
        position: relative;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        color: var(--cyan);
        text-transform: uppercase;
        background: rgba(79, 216, 255, 0.08);
        border: 1px solid rgba(79, 216, 255, 0.25);
        padding: 6px 14px;
        border-radius: 100px;
        margin-bottom: 22px;
    }
    .eyebrow .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--cyan);
        box-shadow: 0 0 8px var(--cyan);
        animation: pulse 1.6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.7); }
    }
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 4rem;
        line-height: 1.02;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(120deg, #ffffff 30%, var(--cyan) 65%, var(--violet) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: 14px;
        margin-bottom: 2.4rem;
        font-weight: 400;
    }

    /* ---------- Step indicator (real 3-step pipeline) ---------- */
    .steps {
        display: flex;
        justify-content: center;
        gap: 0;
        margin-bottom: 2.2rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .step {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--muted);
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0 18px;
    }
    .step .num {
        width: 22px; height: 22px;
        border-radius: 50%;
        border: 1px solid var(--line);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.72rem;
        color: var(--ink);
    }
    .step.active .num {
        border-color: var(--cyan);
        color: var(--cyan);
        box-shadow: 0 0 10px rgba(79,216,255,0.4);
    }
    .step-divider { color: var(--line); font-size: 0.9rem; padding-top: 2px; }

    /* ---------- Upload card ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface) !important;
        border: 1.5px dashed var(--line) !important;
        border-radius: 18px !important;
        transition: border-color 0.3s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--cyan) !important;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        width: 100%;
        border: 1px solid var(--line);
        background-color: var(--surface);
        color: var(--ink);
        border-radius: 12px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        letter-spacing: 0.02em;
        padding: 0.7rem;
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        border-color: var(--cyan);
        color: var(--cyan);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 216, 255, 0.15);
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(90deg, var(--cyan), var(--violet));
        color: #08080c;
        border: none;
        font-weight: 700;
    }
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 10px 26px rgba(155, 123, 255, 0.35);
        transform: translateY(-2px);
    }

    /* ---------- Word chips (detected gestures) ---------- */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: 14px;
    }
    .chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: var(--cyan);
        background: rgba(79, 216, 255, 0.08);
        border: 1px solid rgba(79, 216, 255, 0.3);
        padding: 5px 12px;
        border-radius: 8px;
    }

    /* ---------- Result box ---------- */
    .result-box {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 42px 34px;
        text-align: left;
        box-shadow: 0 24px 60px rgba(0,0,0,0.45);
        margin-top: 28px;
        position: relative;
        overflow: hidden;
    }
    .result-box::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
    }
    .result-label {
        color: var(--muted);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        margin-bottom: 6px;
    }
    .result-sentence {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        margin: 0;
        line-height: 1.35;
        font-size: 1.5rem;
    }
    .sentence-row {
        border-left: 3px solid var(--cyan);
        padding: 14px 18px;
        margin-bottom: 10px;
        background: rgba(79, 216, 255, 0.04);
        border-radius: 0 12px 12px 0;
    }
    .sentence-row .idx {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--muted);
        letter-spacing: 0.1em;
    }
    hr { border-top: 1px solid var(--line) !important; margin: 26px 0 !important; }
    </style>
""", unsafe_allow_html=True)

if "words" not in st.session_state:
    st.session_state.words = []
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "sentences" not in st.session_state:
    st.session_state.sentences = []


def clear_all():
    st.session_state.words = []
    st.session_state.show_result = False
    st.session_state.sentences = []


@st.cache_resource
def load_sign_model():
    sign_model = load_model(SIGN_MODEL_FILE)
    label_encoder = joblib.load(LABEL_FILE)
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_FILE),
        num_hands=2,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    return sign_model, label_encoder, hand_detector


@st.cache_resource
def load_sentence_model():
    tokenizer = T5Tokenizer.from_pretrained(SENTENCE_MODEL_NAME)
    text_model = T5ForConditionalGeneration.from_pretrained(SENTENCE_MODEL_NAME)
    return tokenizer, text_model


sign_model, label_encoder, hand_detector = load_sign_model()


def get_hand_points(frame):
    """Extracts landmark points for both hands from a single frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = hand_detector.detect(image)

    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if result and result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks):
            wrist = hand[0]
            values = []

            for point in hand:
                values.append(point.x - wrist.x)
                values.append(point.y - wrist.y)
                values.append(point.z - wrist.z)

            biggest = max([abs(v) for v in values], default=0.0)
            if biggest > 0.0:
                values = [v / biggest for v in values]

            side = result.handedness[i][0].category_name
            if side == "Left":
                left_hand = values
            else:
                right_hand = values

    return left_hand + right_hand


def get_word(frame, history):
    """Adds a new frame to the history and predicts the word using the model."""
    history.append(get_hand_points(frame))

    if len(history) < FRAME_COUNT:
        return "", 0.0

    input_data = np.expand_dims(np.array(history, dtype=np.float32), axis=0)
    scores = sign_model.predict(input_data, verbose=0)[0]
    best_index = int(np.argmax(scores))
    confidence = float(scores[best_index])

    if confidence < MIN_CONFIDENCE:
        return "", 0.0

    word = label_encoder.inverse_transform([best_index])[0]
    return word, confidence


def get_stable_word(word, confidence, history):
    """Chooses the most stable word by taking a vote from recent predictions."""
    if not word:
        return "", 0.0

    history.append((word, confidence))

    if confidence >= 0.70:
        return word, confidence

    if len(history) < 2:
        return "", 0.0

    votes = Counter(w for w, c in history)
    top_word, count = votes.most_common(1)[0]

    if count >= 2:
        avg_confidence = sum(c for w, c in history if w == top_word) / count
        return top_word, avg_confidence

    return "", 0.0


def make_sentences(words):
    """Converts a list of words into natural English sentences."""
    if not words:
        return []

    tokenizer, text_model = load_sentence_model()
    word_text = ", ".join(words)

    prompt = (
        "Turn the given words into one short, natural English sentence.\n\n"
        "Words: tea, sweet\n"
        "Sentence: The tea is sweet.\n\n"
        "Words: I, school, go\n"
        "Sentence: I go to school.\n\n"
        "Words: water, want\n"
        "Sentence: I want water.\n\n"
        f"Words: {word_text}\n"
        "Sentence:"
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = text_model.generate(
        input_ids,
        max_new_tokens=30,
        num_return_sequences=15,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        no_repeat_ngram_size=2,
    )

    unique_results = []
    for out in output_ids:
        res = tokenizer.decode(out, skip_special_tokens=True).strip().capitalize()
        if res and res not in unique_results:
            unique_results.append(res)
        if len(unique_results) == 5:
            break

    while len(unique_results) < 5:
        unique_results.append(" ".join(words).capitalize())

    return unique_results


def process_video(uploaded_file):
    """Extracts words from an uploaded video and saves them to the session state."""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video = cv2.VideoCapture(temp_file.name)

    if not video.isOpened():
        return

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // FRAME_COUNT) if total_frames > 0 else 1

    word_history = deque(maxlen=FRAME_COUNT)
    recent_words = deque(maxlen=5)
    last_word_added = None
    frame_number = 0

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        frame_number += 1
        if frame_number % step != 0:
            continue

        word, confidence = get_word(frame, word_history)
        if not word:
            continue

        final_word, _ = get_stable_word(word, confidence, recent_words)
        if not final_word or final_word == last_word_added:
            continue

        last_word_added = final_word
        if not st.session_state.words or st.session_state.words[-1] != final_word:
            st.session_state.words.append(final_word)

    video.release()


def render_chips(words):
    """Displays the detected words in a pill/chip style format."""
    if not words:
        return '<p style="color: var(--muted); font-family: JetBrains Mono, monospace; font-size: 0.85rem; text-align:center;">No gestures decoded yet.</p>'
    chips = "".join([f'<span class="chip">{w}</span>' for w in words])
    return f'<div class="chip-row">{chips}</div>'


def render_box(words_text, sentences_list, text_color="var(--ink)", glow=False):
    """Generates the HTML for the result box."""
    list_items = ""
    for i, s in enumerate(sentences_list):
        glow_style = "text-shadow: 0 0 16px rgba(79,216,255,0.35);" if glow else ""
        list_items += (
            '<div class="sentence-row">'
            f'<div class="idx">VARIANT {i + 1:02d}</div>'
            f'<p class="result-sentence" style="color: {text_color}; {glow_style}">{s}</p>'
            '</div>'
        )

    return f"""
        <div class="result-box">
            <div class="result-label">Input Sequence</div>
            <p style="color: var(--cyan); font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; margin-bottom: 24px;">{words_text}</p>
            <hr>
            <div class="result-label" style="margin-bottom: 16px;">Synthesized Variations · 05</div>
            {list_items}
        </div>
    """


st.markdown("""
    <div class="hero-wrap">
        <span class="eyebrow"><span class="dot"></span> LSTM Motion Model · 98 Gestures</span>
        <h1 class="main-title">NEURAL SIGN</h1>
        <p class="sub-title">Upload a gesture video — the model reads hand motion, decodes the words, and writes the sentence for you.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="steps">
        <div class="step active"><span class="num">1</span> Upload</div>
        <div class="step-divider">—</div>
        <div class="step"><span class="num">2</span> Decode</div>
        <div class="step-divider">—</div>
        <div class="step"><span class="num">3</span> Translate</div>
    </div>
""", unsafe_allow_html=True)

spacer1, center_col, spacer3 = st.columns([1, 2, 1])

with center_col:
    uploaded_files = st.file_uploader(
        "Drop video files here",
        type=["mp4", "mov", "avi"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.write("")
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        process_clicked = st.button("PROCESS", type="primary")
    with btn_col2:
        done_clicked = st.button("TRANSLATE")
    with btn_col3:
        st.button("RESET", on_click=clear_all)

    status_area = st.empty()
    status_area.markdown(render_chips(st.session_state.words), unsafe_allow_html=True)

if uploaded_files and process_clicked:
    st.session_state.words = []

    with center_col:
        with st.spinner("Decoding gestures..."):
            for uploaded_file in uploaded_files:
                process_video(uploaded_file)

        status_area.markdown(render_chips(st.session_state.words), unsafe_allow_html=True)

if done_clicked:
    if not st.session_state.words:
        with center_col:
            st.error("No gesture data found. Upload and process a video first.")
    else:
        st.session_state.show_result = True
        with center_col:
            with st.spinner("Synthesizing context variations..."):
                st.session_state.sentences = make_sentences(st.session_state.words)

if st.session_state.show_result:
    words_text = ", ".join(st.session_state.words) if st.session_state.words else "None"
    final_sentences = st.session_state.sentences

    with center_col:
        box = st.empty()

        if final_sentences:
            chars = string.ascii_uppercase + string.digits + "!@#$%^&*"
            for _ in range(10):
                glitched_list = [
                    "".join(random.choice(chars) if c != " " else " " for c in s)
                    for s in final_sentences
                ]
                box.markdown(
                    render_box(words_text, glitched_list, text_color="var(--violet)", glow=True),
                    unsafe_allow_html=True
                )
                time.sleep(0.05)

        box.markdown(render_box(words_text, final_sentences), unsafe_allow_html=True)