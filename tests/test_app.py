"""Smoke tests: every page renders without exceptions on the processed data."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["analyze", "compare", "explore", "guide", "about"]

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "flights.parquet").exists(),
    reason="processed data not built (run src.download_data and src.data_processing)",
)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / "app_pages" / f"{page}.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_analyze_midnight_flight_shows_previous_evening():
    at = AppTest.from_file(str(ROOT / "app_pages" / "analyze.py"), default_timeout=60).run()
    from datetime import time
    at.time_input(key="an_time").set_value(time(1, 0)).run()
    assert not at.exception
    assert any("the evening before" in m.value for m in at.markdown)


def test_explore_shows_insight_and_baseline():
    at = AppTest.from_file(str(ROOT / "app_pages" / "explore.py"), default_timeout=60).run()
    assert not at.exception
    texts = [m.value for m in at.markdown]
    assert any("delays build up during the day" in t for t in texts)
    assert any("Baseline the M3 model must beat" in t for t in texts)


def test_explore_empty_filter_shows_warning():
    at = AppTest.from_file(str(ROOT / "app_pages" / "explore.py"), default_timeout=60).run()
    # flydubai only flies to Dubai, so FZ + Greece matches nothing
    at.multiselect(key="ex_air").set_value(["FZ"])
    at.multiselect(key="ex_cty").set_value(["Greece"]).run()
    assert not at.exception
    assert at.warning, "expected an empty-state warning when no flights match"
