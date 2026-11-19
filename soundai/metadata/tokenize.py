"""Path tokenization for rule-based metadata extraction."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import PurePosixPath

_CAMEL = re.compile(r"([a-z])([A-Z])")
_LETTER_DIGIT = re.compile(r"([A-Za-z])(\d)")
_DIGIT_LETTER = re.compile(r"(\d)([A-Za-z])")
# Design §2.1: [ _\-\.\(\)\[\]]+ plus & and + which Splice uses in folder names.
# Triple underscores (Drum___Perc) collapse naturally.
_SEP = re.compile(r"[ _\-\.\(\)\[\]\{\},;+&]+")

SKIP_NAMES = frozenset({".ds_store"})


def is_junk_path(rel_path: str) -> bool:
    """Return True for AppleDouble, .DS_Store, and __MACOSX entries."""
    parts = PurePosixPath(rel_path).parts
    name = PurePosixPath(rel_path).name
    if name.startswith("._") or name.lower() in SKIP_NAMES:
        return True
    return any(p == "__MACOSX" or p.startswith("._") for p in parts)


def strip_extension(name: str) -> str:
    if "." in name and not name.startswith("."):
        return name.rsplit(".", 1)[0]
    return name


def tokenize_component(name: str, *, is_filename: bool = False) -> list[str]:
    """Normalize one path component into lowercase tokens.

    Splits on separators, camelCase, and letter/digit boundaries
    (``Kick01`` → ``kick``, ``01``; ``120BPM`` → ``120``, ``bpm``;
    ``perc80`` → ``perc``, ``80``).  ``#`` is kept so ``F#m`` stays one token.
    """
    text = unicodedata.normalize("NFC", name)
    if is_filename:
        text = strip_extension(text)
    text = _CAMEL.sub(r"\1 \2", text)
    text = _LETTER_DIGIT.sub(r"\1 \2", text)
    text = _DIGIT_LETTER.sub(r"\1 \2", text)
    text = text.lower()
    return [tok for tok in _SEP.split(text) if tok]


def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


# Folder names that are inventory / vendor containers rather than categories.
_PACK_LIKE_MARKERS = frozenset({
    "sample", "samples", "pack", "packs", "vol", "volume", "wav", "sku",
    "splice", "zenhiser", "presents",
})
_GENERIC_CONTAINERS = frozenset({
    "audio", "files", "wav", "wavs", "for", "splice",
})
# First-level folders that *are* useful category roots (TAIKO_drums, …).
_CATEGORY_TOKENS = frozenset({
    "drums", "drum", "kicks", "kick", "snares", "snare", "hats", "hihat",
    "hihats", "perc", "percussion", "claps", "clap", "toms", "cymbals",
    "loops", "loop", "shots", "oneshot", "oneshots", "hits",
    "bass", "keys", "piano", "pianos", "synth", "synths", "guitar", "guitars",
    "vocals", "vocal", "fx", "sfx", "tonal", "melodic", "brass", "strings",
    "winds", "pads", "leads", "organ", "rhodes",
})


def is_pack_like_folder(name: str, pack_tokens: set[str]) -> bool:
    """True when a folder is a pack-title echo, not an instrument category.

    ``Bedroom_Pianos_&_Electronics_-_WAV`` would otherwise label every file
    as piano; ``TAIKO_drums`` is a real category and must stay an ancestor.
    """
    tokens = tokenize_component(name)
    if not tokens:
        return True
    if tokens[0].startswith("_") or set(tokens) <= _GENERIC_CONTAINERS:
        return True
    if any(t in _PACK_LIKE_MARKERS for t in tokens) and len(tokens) >= 3:
        return True
    if len(tokens) >= 4 and len(set(tokens) & pack_tokens) >= 2:
        return True
    if len(tokens) >= 5:
        return True
    return False


@dataclass(frozen=True)
class PathLayers:
    """Token layers of a relative path, strongest signal first."""

    rel_path: str
    pack: str
    ext: str
    filename_stem: str
    filename: list[str]
    parent_name: str | None
    parent: list[str]
    ancestor_names: tuple[str, ...]
    ancestors: list[list[str]]
    pack_tokens: list[str]
    all_tokens: list[str] = field(default_factory=list)

    @property
    def parent_family_tokens(self) -> set[str]:
        return set(self.parent) | {t for layer in self.ancestors for t in layer}


def split_rel_path(rel_path: str) -> PathLayers:
    """Break a SSD-relative (or pack-relative) path into token layers."""
    path = PurePosixPath(rel_path.replace("\\", "/"))
    parts = list(path.parts)
    if not parts:
        raise ValueError(f"empty path: {rel_path!r}")

    pack = parts[0]
    ext = path.suffix.lower().lstrip(".")
    filename_stem = strip_extension(parts[-1])
    filename = tokenize_component(parts[-1], is_filename=True)

    parent_name: str | None = None
    parent: list[str] = []
    ancestor_names: list[str] = []
    ancestors: list[list[str]] = []

    middle = parts[1:-1]  # folders between pack and file
    pack_tokens = tokenize_component(pack)
    pack_token_set = set(pack_tokens)

    if middle:
        parent_name = middle[-1]
        parent = tokenize_component(parent_name)
        raw_ancestors = middle[:-1]
        for i, name in enumerate(raw_ancestors):
            # The first folder under the pack is often the pack title itself.
            if i == 0 and is_pack_like_folder(name, pack_token_set):
                continue
            ancestor_names.append(name)
            ancestors.append(tokenize_component(name))

    all_tokens: list[str] = []
    for layer in (filename, parent, *[a for a in ancestors], pack_tokens):
        for tok in layer:
            if tok not in all_tokens:
                all_tokens.append(tok)

    return PathLayers(
        rel_path=str(path),
        pack=pack,
        ext=ext,
        filename_stem=filename_stem,
        filename=filename,
        parent_name=parent_name,
        parent=parent,
        ancestor_names=tuple(ancestor_names),
        ancestors=ancestors,
        pack_tokens=pack_tokens,
        all_tokens=all_tokens,
    )
