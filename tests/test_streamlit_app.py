"""
JobShield Streamlit App Automated Tests
Uses Streamlit AppTest framework to verify UI execution,
form submission, button clicks, and tab rendering without exceptions.
"""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_loads_without_exceptions():
    """Verifies that the multi-tab application starts cleanly with 0 exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()
    assert not at.exception
    assert at.tabs[0].label == "Home"
    assert len(at.tabs) == 5


def test_single_ad_form_submission_with_scam():
    """Simulates filling in the screening form with a scam advertisement and submitting."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()

    # Enter text in description
    desc_area = at.text_area[0]
    desc_area.set_value(
        "Urgent data entry opening! Work from home and earn $4500 weekly. "
        "No interview or experience required. Candidates must wire transfer a refundable "
        "equipment registration fee of $150 to confirm selection."
    )

    # Click 'Screen Advertisement' button (button index 2)
    screen_btn = None
    for b in at.button:
        if "Screen Advertisement" in b.label:
            screen_btn = b
            break
    assert screen_btn is not None
    screen_btn.click().run()

    assert not at.exception


def test_single_ad_form_with_inadequate_input():
    """Simulates submitting an empty/too-short advertisement and checks graceful handling."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()

    desc_area = at.text_area[0]
    desc_area.set_value("Short")

    screen_btn = None
    for b in at.button:
        if "Screen Advertisement" in b.label:
            screen_btn = b
            break
    assert screen_btn is not None
    screen_btn.click().run()

    assert not at.exception
    # Verify input adequacy warning message appears
    assert len(at.warning) > 0


def test_side_by_side_comparison_run():
    """Simulates clicking the compare button in Tab 2."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.run()

    # Find and click the compare button
    compare_btn = None
    for b in at.button:
        if "Compare Both" in b.label:
            compare_btn = b
            break
    assert compare_btn is not None
    compare_btn.click().run()
    assert not at.exception
