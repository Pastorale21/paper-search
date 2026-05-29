"""GROBID full-text PDF parser (extension point for owner A).

Parses a PDF into structured TEI/JSON (sections, references, method text) so method cards can be
built from full text rather than abstracts alone. Stubbed until the Docker GROBID service is wired
up; see data/HANDOFF.md. Reference client:
https://github.com/kermitt2/grobid-client-python
"""

from __future__ import annotations

from pathlib import Path


class GrobidClient:
    """Thin client over a running GROBID service (http://localhost:8070 by default)."""

    def parse_pdf(self, pdf_path: Path) -> dict:
        """Parse a PDF into a structured dict (sections, references, full text)."""
        raise NotImplementedError("TODO(A): implement with Docker grobid service, see HANDOFF.md")
