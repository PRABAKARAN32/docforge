"""Normalization + hashing: turn a page's markdown into a stable fingerprint.

Decision 5.4 (GUID.md): hash the *normalized* markdown, not the raw content. Docs
pages carry cosmetic noise that changes on every crawl without the meaning changing
-- "Last updated" dates, trailing whitespace, extra blank lines. If we hashed that
directly, every page would look "changed" every run and DocForge's whole point (touch
only what changed) would collapse. So we normalize away the noise first, then hash.

The hash is a short, fixed-length fingerprint of a page. Comparing fingerprints across
runs is how change detection works, and only the fingerprint (not the whole page) is
stored in the manifest.
"""

from __future__ import annotations

import hashlib
import re

# Lines that are pure metadata/volatile and should not count as "content". Matched
# case-insensitively against a whole line. Kept deliberately conservative -- these
# phrases almost never appear in real documentation prose, only in page chrome.
_VOLATILE_LINE = re.compile(
    r"^\s*(?:last updated|last modified|last edited|page last reviewed|edit this page)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)  # spaces/tabs at end of a line
_BLANK_RUN = re.compile(r"\n{3,}")  # 3+ newlines in a row -> a run of blank lines


def normalize_markdown(md: str) -> str:
    """Reduce markdown to its meaningful content, dropping cosmetic/volatile noise.

    Steps (each conservative and independent):
      1. Unify line endings (Windows/Mac -> ``\\n``).
      2. Drop volatile metadata lines (e.g. "Last updated: ...").
      3. Strip trailing whitespace on every line.
      4. Collapse runs of 3+ newlines into a single blank line.
      5. Trim leading/trailing whitespace and end with exactly one newline.

    The goal is that two crawls of an unchanged page produce identical output, so their
    hashes match -- while a real content edit still changes the output.
    """
    text = md.replace("\r\n", "\n").replace("\r", "\n")
    text = _VOLATILE_LINE.sub("", text)
    text = _TRAILING_WS.sub("", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip() + "\n"


def content_hash(md: str) -> str:
    """Return the SHA-256 fingerprint of a page's *normalized* markdown.

    Normalization happens here so callers can never forget it -- pass raw markdown and
    get back a stable, noise-insensitive hash.
    """
    normalized = normalize_markdown(md)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
