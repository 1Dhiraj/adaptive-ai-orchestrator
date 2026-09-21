"""Small, deterministic PDF renderer used by artifact and report exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape


def write_pdf(path: Path | str, content: str, title: Optional[str] = None) -> str:
    """Render readable text/Markdown-like content as a real PDF.

    ReportLab is a project dependency because producing a text file with a
    ``.pdf`` suffix is worse than failing: the dashboard would advertise an
    artifact that no PDF viewer can open.
    """
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    PageBreak)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clean_title = (title or target.stem.replace("_", " ").replace("-", " ")).strip()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ArtifactTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=20, leading=24, spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="ArtifactBody", parent=styles["BodyText"], fontSize=10.5,
        leading=15, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="ArtifactBullet", parent=styles["BodyText"], fontSize=10.5,
        leading=15, leftIndent=14, firstLineIndent=-8, spaceAfter=5,
    ))

    story = [Paragraph(escape(clean_title or "Report"), styles["ArtifactTitle"])]
    lines = str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for raw in lines:
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 3 * mm))
            continue
        if line == "---":
            story.append(Spacer(1, 2 * mm))
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            style = styles["Heading1" if level == 1 else "Heading2"]
            story.append(Paragraph(escape(heading.group(2)), style))
            continue
        bullet = re.match(r"^(?:[-*]|\d+[.)])\s+(.+)$", line)
        if bullet:
            story.append(Paragraph("• " + escape(bullet.group(1)), styles["ArtifactBullet"]))
            continue
        if line == "[[PAGE_BREAK]]":
            story.append(PageBreak())
            continue
        story.append(Paragraph(escape(line), styles["ArtifactBody"]))

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColorRGB(0.35, 0.39, 0.44)
        canvas.drawString(18 * mm, 12 * mm, clean_title[:75])
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(target), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=19 * mm,
        title=clean_title, author="Adaptive AI Task Orchestrator",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    data = target.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-1024:]:
        target.unlink(missing_ok=True)
        raise ValueError("PDF renderer did not produce a valid PDF document")
    return str(target.resolve())
