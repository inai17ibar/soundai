"""Rule-based path metadata extraction (no audio is read)."""

from soundai.metadata.extractor import ExtractedMetadata, extract_metadata
from soundai.metadata.taxonomy import RULES_VERSION

__all__ = ["ExtractedMetadata", "RULES_VERSION", "extract_metadata"]
