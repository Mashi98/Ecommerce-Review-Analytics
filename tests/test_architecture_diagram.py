"""Smoke tests for the architecture diagram generator.

The layout itself is checked by eye on the rendered PNG; these only guard
against the generator breaking or writing nothing.
"""

import scripts.architecture_diagram as diagram


def test_draw_returns_a_figure_with_the_pipeline_labels():
    fig = diagram.draw()
    text = " ".join(t.get_text() for t in fig.axes[0].texts)
    for label in ("Raw reviews", "Baseline sentiment", "DistilBERT sentiment", "BERTopic topics", "Evaluation"):
        assert label in text


def test_main_writes_png_and_svg(tmp_path, monkeypatch):
    monkeypatch.setattr(diagram, "FIGURES_DIR", tmp_path)
    diagram.main()
    assert (tmp_path / "architecture.png").stat().st_size > 10_000
    assert (tmp_path / "architecture.svg").stat().st_size > 1_000
