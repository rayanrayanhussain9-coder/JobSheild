"""About JobShield and its scroll-driven globe."""
from base64 import b64encode
from pathlib import Path

import streamlit as st


HOME_ASSETS = Path(__file__).parent


@st.cache_resource
def globe_component():
    artwork = (HOME_ASSETS / "hero_art.html").read_text()
    image = b64encode((HOME_ASSETS / "globe.webp").read_bytes()).decode("ascii")
    scripts = "\n".join((HOME_ASSETS / filename).read_text() for filename in
                        ("globe_geometry.js", "globe_viewer.js"))
    return st.components.v2.component(
        "jobshield_globe", html=artwork.replace("__GLOBE_URL__", "data:image/webp;base64," + image),
        js=scripts, isolate_styles=False,
    )


def open_checker(sample=False):
    st.session_state.main_nav = "Check a job"
    st.session_state.hero_sample = sample


def render_about_intro():
    st.html("""
    <style>
    .st-key-about_hero {background:#fff;padding:clamp(32px,5vh,74px) 0 14px}
    .st-key-about_hero h1 {font:400 clamp(47px,6.3vw,94px)/.99 Georgia,serif;letter-spacing:-.057em;color:#1d1b23;margin:0;padding:0}
    .st-key-about_hero h1 em {color:#5743ae}
    .st-key-about_hero .js-eyebrow {color:#514777;font:650 11px/1.35 system-ui;letter-spacing:.14em;text-transform:uppercase;margin:0 0 24px}
    .st-key-about_hero .hero-intro {font:clamp(16px,1.45vw,21px)/1.55 system-ui;max-width:42ch;margin:22px 0 6px;color:#4c4956}
    .js-scroll {font:600 11px/1.5 system-ui;letter-spacing:.12em;text-transform:uppercase;color:#514777;margin:16px 0}
    .js-scroll span {display:inline-block;font:20px Georgia,serif;margin-right:8px}
    @media (max-width:700px) {.st-key-about_hero {padding-top:22px}.js-scroll {margin:0 0 12px}}
    </style>""")
    with st.container(key="about_hero"):
        left, right = st.columns([1, 1.05], vertical_alignment="center", gap="medium")
        with left:
            st.html("""<div class="js-eyebrow">About JobShield &nbsp; / &nbsp; AI job screening</div>
            <h1>Spot warning signs<br><em>before you apply.</em></h1>
            <p class="hero-intro">Paste a job ad to check for recruitment scam patterns. See the wording behind a warning before you decide what to do next.</p>""")
            with st.container(horizontal=True):
                st.button("Check a job", key="about_check", type="primary", on_click=open_checker)
                st.button("Try a sample ad", key="about_sample", on_click=open_checker, args=(True,))
        with right:
            globe_component()(key="about_globe")
    st.html('<p class="js-scroll"><span aria-hidden="true">↓</span> Scroll to explore the sample</p>')
    st.html(HOME_ASSETS / "home_sections.html")
