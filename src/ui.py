"""Presentation helpers: styling and the result panel. No model logic here."""
from html import escape

import streamlit as st

from . import config
from .model import Prediction

# Purely decorative — per-class emoji used only for display. Doesn't touch
# CLASS_NAMES, NO_TUMOR_INDEX or anything model-facing.
CLASS_EMOJI = {
    "Glioma": "🔴",
    "Meningioma": "🟠",
    "No tumor": "✅",
    "Pituitary": "🟣",
}


def _html(markup: str) -> str:
    # Collapse whitespace so Markdown never mistakes indented HTML for a code block.
    return " ".join(line.strip() for line in markup.splitlines())


def inject_css() -> None:
    css = config.STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        _html(
            f"""
            <header class="masthead">
              <div class="masthead__img" style="background-image:url('{config.HERO_IMAGE}');"></div>
              <div class="masthead__overlay"></div>
              <div class="masthead__content">
                <span class="masthead__eyebrow">🧠 NEURO-IMAGING AID</span>
                <h1>🧠 {config.APP_TITLE}</h1>
                <p>Upload one MRI slice. The model estimates how likely it is to show a tumor. 🔬</p>
              </div>
            </header>
            """
        ),
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        _html(
            f"""
            <div class="empty">
              <img src="{config.EMPTY_IMAGE}" alt="" />
              <div>
                <strong>🩻 No scan yet</strong>
                Upload an MRI slice to see the model's prediction here. 📤
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _class_rows(prediction: Prediction) -> str:
    top = max(range(len(prediction.class_probabilities)), key=lambda i: prediction.class_probabilities[i][1])
    rows = []
    for i, (name, prob) in enumerate(prediction.class_probabilities):
        kind = "clear" if i == config.NO_TUMOR_INDEX else "tumor"
        strong = " classes__row--top" if i == top else ""
        rows.append(
            f'<li class="classes__row{strong}">'
            f'<span class="classes__name">{CLASS_EMOJI.get(name, "")} {escape(name)}</span>'
            f'<span class="classes__bar"><span class="classes__fill classes__fill--{kind}" '
            f'style="width:{prob * 100:.1f}%"></span></span>'
            f'<span class="classes__pct">{prob * 100:.1f}%</span>'
            f"</li>"
        )
    return "".join(rows)


def render_result(prediction: Prediction, threshold: float) -> None:
    probability = prediction.tumor_probability
    is_tumor = probability >= threshold
    kind = "tumor" if is_tumor else "clear"
    title = "⚠️ Tumor indicated" if is_tumor else "✅ No tumor indicated"

    note = f"The model puts the chance of a tumor at {probability * 100:.1f}%."
    if is_tumor:
        note += f" The highest-scoring type is {prediction.top_tumor_class.lower()}."
    if abs(probability - threshold) < config.UNCERTAIN_MARGIN:
        note += " That is close to the threshold, so treat the result as uncertain."

    fill = probability * 100
    tick = threshold * 100
    label_pos = min(max(tick, 14), 86)  # keep the label clear of the 0% / 100% ends

    st.markdown(
        _html(
            f"""
            <section class="verdict verdict--{kind}" aria-live="polite">
              <h2 class="verdict__title">{title}</h2>
              <p class="verdict__note">{escape(note)}</p>
            </section>
            <div class="gauge" role="meter" aria-label="Tumor probability"
                 aria-valuemin="0" aria-valuemax="100" aria-valuenow="{fill:.1f}">
              <div class="gauge__track">
                <div class="gauge__fill gauge__fill--{kind}" style="width:{fill:.1f}%"></div>
                <div class="gauge__tick" style="left:{tick:.1f}%"></div>
              </div>
              <div class="gauge__scale">
                <span>0%</span>
                <span class="gauge__tick-label" style="left:{label_pos:.1f}%">🎯 threshold {tick:.0f}%</span>
                <span>100%</span>
              </div>
            </div>
            <p class="classes__title">📊 Model score for each class</p>
            <ul class="classes">{_class_rows(prediction)}</ul>
            """
        ),
        unsafe_allow_html=True,
    )


def render_fineprint() -> None:
    st.markdown(
        _html(
            """
            <p class="fineprint">
              ℹ️ This is a research demo, not a medical device. It can be wrong, and its output
              must not be used to diagnose a tumor or rule one out. Ask a clinician to review
              any scan.
            </p>
            """
        ),
        unsafe_allow_html=True,
    )
