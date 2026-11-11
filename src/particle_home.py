"""JobShield's particle landing screen."""
from pathlib import Path

import streamlit as st


@st.cache_resource
def _particle_component():
    source = Path(__file__).parent
    logo = (source.parent / "design" / "jobshield-ribbon.svg").read_text()
    # The artwork has unused space to the right. Crop it for a centered lockup.
    logo = logo.replace('viewBox="0 0 540 142"', 'viewBox="21 10 441 119"')
    markup = (source / "particle_home.html").read_text().replace("<!-- ribbon -->", logo)
    return st.components.v2.component(
        "jobshield_particle_home",
        html=markup,
        css=(source / "particle_home.css").read_text(),
        js=(source / "particle_home.js").read_text(),
    )


def _open_checker():
    st.session_state.main_nav = "Check a job"
    st.session_state.hero_sample = False


def render_particle_home():
    _particle_component()(key="particle_home", on_check_change=_open_checker)
