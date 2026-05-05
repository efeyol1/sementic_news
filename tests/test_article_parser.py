from __future__ import annotations

from src.data.article_parser import parse_article_html


def test_parse_article_html_extracts_metadata_and_body():
    paragraphs = "".join(
        f"<p>Bu haber paragrafı yeterince uzun bir gövde metni içeriyor ve sıra numarası {i}.</p>"
        for i in range(5)
    )
    html = f"""
    <html>
      <head>
        <meta property="og:title" content="Gerçek Haber Başlığı">
        <meta property="og:description" content="Kısa haber özeti">
        <meta property="article:published_time" content="2026-05-05T10:00:00+03:00">
      </head>
      <body>
        <nav>Menü metni</nav>
        <article>{paragraphs}</article>
      </body>
    </html>
    """

    parsed = parse_article_html(html)

    assert parsed.parse_status == "parsed"
    assert parsed.title == "Gerçek Haber Başlığı"
    assert parsed.summary == "Kısa haber özeti"
    assert parsed.published_date == "2026-05-05T10:00:00+03:00"
    assert "Menü metni" not in parsed.cleaned_article_text
    assert "haber paragrafı" in parsed.cleaned_article_text


def test_parse_article_html_marks_short_body_empty():
    parsed = parse_article_html("<html><body><article><p>Kısa metin.</p></article></body></html>")

    assert parsed.parse_status == "empty"
    assert "shorter than" in (parsed.parse_error or "")
