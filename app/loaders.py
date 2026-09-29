"""
loaders.py
Turns each supported file type into a list of "units" -- the natural pieces
a citation should point at (a PDF page, a markdown section, one email).

Each unit: {"text", "page" (int, for ordering), "location" (label shown in
citations), optional "doc_ts" (epoch seconds, e.g. an email's sent date)}.
"""

import email
import email.policy
import mailbox
import os
import re
from email.utils import parsedate_to_datetime


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _sections(text):
    """Split markdown-ish text on '#' headings; each section is one unit."""
    sections, title, buf = [], "(intro)", []
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            sections.append((title, buf))
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    sections.append((title, buf))

    units = []
    for n, (title, buf) in enumerate(sections, 1):
        body = _clean(" ".join(buf))
        if not body:
            continue
        text_out = body if title == "(intro)" else f"{title}. {body}"
        units.append({"text": text_out, "page": n, "location": f"section: {title[:60]}"})
    return units


def load_pdf(path):
    from pypdf import PdfReader
    units = []
    for i, page in enumerate(PdfReader(path).pages, 1):
        text = _clean(page.extract_text() or "")
        if text:
            units.append({"text": text, "page": i, "location": f"page {i}"})
    return units


def load_markdown(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return _sections(f.read())


def load_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = _clean(f.read())
    return [{"text": text, "page": 1, "location": "text"}] if text else []


def load_docx(path):
    from docx import Document
    lines = []
    for p in Document(path).paragraphs:
        if p.style is not None and p.style.name.lower().startswith("heading"):
            lines.append(f"# {p.text}")
        else:
            lines.append(p.text)
    return _sections("\n".join(lines))


def _email_body(msg):
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    text = part.get_content()
    if part.get_content_type() == "text/html":
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def _email_unit(msg, n):
    subject = str(msg.get("Subject", "(no subject)"))
    sender = str(msg.get("From", ""))
    date = str(msg.get("Date", ""))
    body = _clean(_email_body(msg))
    if not body:
        return None
    unit = {
        "text": f"Email from {sender} on {date}. Subject: {subject}. {body}",
        "page": n,
        "location": f"email: {subject[:60]}",
    }
    try:
        unit["doc_ts"] = parsedate_to_datetime(date).timestamp()
    except Exception:
        pass
    return unit


def load_eml(path):
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)
    unit = _email_unit(msg, 1)
    return [unit] if unit else []


def load_mbox(path):
    units = []
    for n, raw in enumerate(mailbox.mbox(path), 1):
        msg = email.message_from_bytes(raw.as_bytes(), policy=email.policy.default)
        unit = _email_unit(msg, n)
        if unit:
            units.append(unit)
    return units


LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".docx": load_docx,
    ".eml": load_eml,
    ".mbox": load_mbox,
}
SUPPORTED = set(LOADERS)


def load_file(path):
    return LOADERS[os.path.splitext(path)[1].lower()](path)
