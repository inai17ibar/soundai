"""Rule-based metadata extractor.  Paths only — no audio is read."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

from soundai.metadata.taxonomy import (
    BPM_CONTEXT_RANGE,
    BPM_EXPLICIT_RANGE,
    FLAT_TO_SHARP,
    GENERIC_LEAVES,
    GENRE_PHRASES,
    INSTRUMENT_KEYWORDS,
    Keyword,
    LEAF_TO_FAMILY,
    NEGATION_TOKENS,
    PACK_PREFIX_TOKENS,
    RULES_VERSION,
    SHARP_TO_CANON,
    SINGLE_INSTRUMENT_PACKS,
    TR808_TOKENS,
    TYPE_LOOP_TOKENS,
    TYPE_ONESHOT_TOKENS,
)
from soundai.metadata.tokenize import PathLayers, bigrams, is_junk_path, split_rel_path, tokenize_component

Source = Literal["filename", "parent", "ancestor", "pack"]
Conf = Literal["high", "medium", "low", "none"]
LayerName = Source

_SOURCE_RANK = {"filename": 4, "parent": 3, "ancestor": 2, "pack": 1}

# Key token: root + optional accidental + optional mode.  Longer mode first.
_KEY_RE = re.compile(
    r"^([a-g])(#|sharp|b|flat|s)?(minor|major|min|maj|m)?$"
)

# Folder names that are themselves a BPM (or contain an explicit bpm token).
_PURE_BPM_RE = re.compile(r"^(\d{2,3})$")


@dataclass
class Match:
    keyword: Keyword
    source: Source
    start: int  # token index in that layer
    negated: bool = False

    @property
    def family(self) -> str:
        return self.keyword.family

    @property
    def leaf(self) -> str:
        return self.keyword.leaf


@dataclass
class ExtractedMetadata:
    """One row of the catalog (design §2.5), plus a few convenience columns."""

    file_id: str
    rel_path: str
    pack: str
    ext: str
    duration_s: float | None = None
    sr: int | None = None
    channels: int | None = None
    family: str = "unknown"
    leaf: str = "unknown"
    instrument_conf: str = "none"
    instrument_source: str | None = None
    type: str = "unknown"
    type_conf: str = "none"
    type_source: str | None = None
    bpm: int | None = None
    bpm_source: str | None = None
    key_root: str | None = None
    key_mode: str | None = None
    key: str | None = None
    genre: str | None = None
    tokens: list[str] = field(default_factory=list)
    matched_rules: list[str] = field(default_factory=list)
    matched_tokens: list[str] = field(default_factory=list)
    conflict: bool = False
    rules_version: str = RULES_VERSION
    verified: bool = False
    verified_by: str | None = None
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_id_for(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()


def extract_metadata(rel_path: str, *, size_bytes: int | None = None) -> ExtractedMetadata:
    """Extract instrument / type / BPM / key labels from a relative path."""
    layers = split_rel_path(rel_path)
    rules: list[str] = []
    matched_tokens: list[str] = []

    family, leaf, inst_conf, inst_source, conflict, inst_rules, inst_toks = _extract_instrument(layers)
    rules.extend(inst_rules)
    matched_tokens.extend(inst_toks)

    typ, typ_conf, typ_source, typ_rules, typ_toks = _extract_type(layers)
    rules.extend(typ_rules)
    matched_tokens.extend(typ_toks)

    bpm, bpm_source, bpm_rules = _extract_bpm(layers)
    rules.extend(bpm_rules)

    key_root, key_mode, key_rules = _extract_key(layers)
    rules.extend(key_rules)

    genre, genre_rules = _extract_genre(layers)
    rules.extend(genre_rules)

    key = None
    if key_root:
        key = f"{key_root} {key_mode or 'unknown'}"

    # Deduplicate matched tokens, keep order
    seen: set[str] = set()
    uniq_toks: list[str] = []
    for t in matched_tokens:
        if t not in seen:
            seen.add(t)
            uniq_toks.append(t)

    return ExtractedMetadata(
        file_id=file_id_for(layers.rel_path),
        rel_path=layers.rel_path,
        pack=layers.pack,
        ext=layers.ext,
        family=family,
        leaf=leaf,
        instrument_conf=inst_conf,
        instrument_source=inst_source,
        type=typ,
        type_conf=typ_conf,
        type_source=typ_source,
        bpm=bpm,
        bpm_source=bpm_source,
        key_root=key_root,
        key_mode=key_mode,
        key=key,
        genre=genre,
        tokens=layers.all_tokens,
        matched_rules=rules,
        matched_tokens=uniq_toks,
        conflict=conflict,
        rules_version=RULES_VERSION,
        size_bytes=size_bytes,
    )


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

def _iter_layers(layers: PathLayers) -> Iterable[tuple[Source, list[str]]]:
    yield "filename", layers.filename
    if layers.parent:
        yield "parent", layers.parent
    for anc in layers.ancestors:
        if anc:
            yield "ancestor", anc
    if layers.pack_tokens:
        yield "pack", layers.pack_tokens


def _family_hint(layers: PathLayers) -> set[str]:
    """Families suggested by parent + ancestor folders (for short abbrev)."""
    hints: set[str] = set()
    for source, tokens in _iter_layers(layers):
        if source == "filename":
            continue
        for m in _raw_matches(tokens, source):
            if m.keyword.tag_only or m.keyword.short or m.negated:
                continue
            hints.add(m.family)
    return hints


def _has_tr808(tokens: list[str]) -> bool:
    if any(t in TR808_TOKENS or t in {"tr"} and "808" in tokens for t in tokens):
        # "tr" adjacent to "808"
        for a, b in bigrams(tokens):
            if a in {"tr", "tr808"} and b in {"808", "808s"}:
                return True
            if a == "tr808" or (a == "tr" and b.startswith("808")):
                return True
        if any(t in TR808_TOKENS for t in tokens):
            return True
    return False


def _raw_matches(tokens: list[str], source: Source) -> list[Match]:
    if not tokens:
        return []
    # Skip a leading pack-prefix token so "TAIKO_87_melodic_..." and
    # folders like "TAIKO_melodic_loops" don't fire the world/taiko keyword.
    # A later "taiko" token (KSHMR_orchestral_drum_taiko_hard) still matches.
    start_offset = 0
    searchable = tokens
    if tokens and tokens[0] in PACK_PREFIX_TOKENS:
        start_offset = 1
        searchable = tokens[1:]

    found: list[Match] = []
    n = len(searchable)
    # Longer phrases first so "bass drum" wins over "bass" and "drum".
    keywords = sorted(INSTRUMENT_KEYWORDS, key=lambda k: (-len(k.tokens), k.key))
    occupied = [False] * n
    for kw in keywords:
        klen = len(kw.tokens)
        for i in range(n - klen + 1):
            if searchable[i:i + klen] != list(kw.tokens):
                continue
            if any(occupied[i:i + klen]) and klen == 1:
                # A longer phrase already claimed these tokens.
                continue
            if klen > 1 and any(occupied[i:i + klen]):
                continue
            negated = _is_negated(searchable, i, klen)
            found.append(Match(kw, source, start_offset + i, negated=negated))
            if not kw.tag_only:
                for j in range(i, i + klen):
                    occupied[j] = True
    return found


def _is_negated(tokens: list[str], start: int, length: int) -> bool:
    """``No_Kick_Loops`` must not classify as kick."""
    if start > 0 and tokens[start - 1] in NEGATION_TOKENS:
        return True
    # A leading "no" in the same folder: no_kick already tokenized as no, kick
    # (handled above) or a single token "nokick".
    if start == 0 and tokens and tokens[0] in NEGATION_TOKENS:
        return True
    if any(t in NEGATION_TOKENS or t == "nokick" for t in tokens[max(0, start - 1):start + length]):
        if start > 0 and tokens[start - 1] in NEGATION_TOKENS:
            return True
    return False


def _filter_matches(matches: list[Match], layers: PathLayers, family_hints: set[str]) -> list[Match]:
    kept: list[Match] = []
    filename_tokens = layers.filename
    tr808 = _has_tr808(filename_tokens) or _has_tr808(layers.parent)

    for m in matches:
        if m.negated:
            continue
        if m.keyword.tag_only:
            continue
        if tr808 and m.keyword.tokens in (("808",), ("808s",)):
            continue
        if m.keyword.short and m.family not in family_hints and m.source == "filename":
            continue
        if m.keyword.short and m.source == "pack":
            continue
        # "taiko" as pack prefix already skipped via start_offset; as weak
        # keyword it should not come from the pack name itself.
        if m.keyword.weak and m.source == "pack":
            continue
        # Weak keywords in ancestors (e.g. "beat" in a leftover folder) are noisy.
        if m.keyword.weak and m.source == "ancestor":
            continue
        kept.append(m)
    return kept


def _drop_mixed_drum_container_leaves(matches: list[Match], layers: PathLayers) -> list[Match]:
    """``Drum_&_Perc_Loops`` is a mixed container, not a perc instrument."""
    container_layers: set[Source] = set()
    for source, tokens in _iter_layers(layers):
        has_drum = any(t in {"drum", "drums"} for t in tokens)
        has_perc = any(t in {"perc", "percussion", "percs"} for t in tokens)
        if has_drum and has_perc:
            container_layers.add(source)
    if not container_layers:
        return matches
    kept: list[Match] = []
    for m in matches:
        if m.source in container_layers and m.leaf in {"perc", "full"}:
            # Keep the family hint only — do not pick perc/full from the
            # container name.  A more specific parent (Snares, Kicks) still wins.
            continue
        kept.append(m)
    return kept


def _apply_no_kick_loop(matches: list[Match], layers: PathLayers) -> list[Match]:
    """``No_Kick_Loops`` is a drum loop (tops / full minus kick), not a kick."""
    def is_no_kick(tokens: list[str]) -> bool:
        if "nokick" in tokens:
            return True
        return any(a in NEGATION_TOKENS and b == "kick" for a, b in zip(tokens, tokens[1:]))

    if not any(
        is_no_kick(tokens)
        for _, tokens in _iter_layers(layers)
    ):
        return matches
    has_loop = any(
        t in TYPE_LOOP_TOKENS
        for _, tokens in _iter_layers(layers)
        for t in tokens
    )
    if not has_loop:
        return matches
    # Drop residual kick matches (already negated) and add drums/full.
    matches = [m for m in matches if m.leaf != "kick"]
    synthetic = Keyword(("no", "kick"), "drums", "full", weak=False)
    matches.append(Match(synthetic, "parent" if is_no_kick(layers.parent) else "filename", 0))
    return matches


def _disambiguate(matches: list[Match], layers: PathLayers) -> list[Match]:
    """Apply design §1.3 traps (bass drum, 808, pad, hit, vocal lead, …)."""
    if not matches:
        return matches

    matches = _drop_mixed_drum_container_leaves(matches, layers)
    matches = _apply_no_kick_loop(matches, layers)

    def has(*leaves_or_tokens: str, family: str | None = None) -> bool:
        for m in matches:
            if family and m.family != family:
                continue
            if m.leaf in leaves_or_tokens or m.keyword.key in leaves_or_tokens:
                return True
        return False

    tokens_by_source: dict[str, list[str]] = {}
    for source, tokens in _iter_layers(layers):
        tokens_by_source.setdefault(source, [])
        # Don't overwrite filename; ancestors may appear multiple times
        if source != "ancestor" or not tokens_by_source[source]:
            if source != "ancestor":
                tokens_by_source[source] = tokens
        if source == "ancestor":
            tokens_by_source.setdefault("ancestor_all", [])
            tokens_by_source["ancestor_all"].extend(tokens)

    filename = layers.filename
    parent = layers.parent
    folder_tokens = set(parent) | {t for layer in layers.ancestors for t in layer}

    # vocal + lead → drop synth/lead
    if any(m.family == "vocal" and m.leaf == "vocal" for m in matches):
        matches = [m for m in matches if not (m.family == "synth" and m.leaf == "lead")]

    # guitar + lead / guitar + chord / piano + chord → keep the real instrument
    if any(m.family == "guitar" for m in matches):
        matches = [m for m in matches if not (m.family == "synth" and m.leaf in {"lead", "chord", "arp"})]
    if any(m.family == "keys" for m in matches):
        matches = [m for m in matches if not (m.family == "synth" and m.leaf in {"chord"})]
    if any(m.family == "bass" and m.leaf in {"synth_bass", "sub_808", "acoustic_bass"} for m in matches):
        # "Bass_Pad" is a bass; "Drop_Bass_Sub" is a bass
        if any(m.leaf == "pad" for m in matches):
            # Keep pad only if there is no bass token in the same source's phrase
            # like "bass pad".  Drop pad when bass is also present in filename.
            if any(m.family == "bass" and m.source == "filename" for m in matches):
                matches = [m for m in matches if m.leaf != "pad"]

    # electric + lead (no guitar word) → synth/lead, not guitar
    if any(m.leaf == "lead" and m.family == "synth" for m in matches):
        matches = [m for m in matches if not (m.leaf == "electric" and m.keyword.tokens == ("electric",))]

    # acoustic + drum → not guitar/acoustic
    if any(m.family == "drums" for m in matches):
        matches = [
            m for m in matches
            if not (m.family == "guitar" and m.keyword.tokens in (("acoustic",), ("steel",)))
        ]

    # orchestral drum / taiko is not strings
    if any(m.family in {"drums", "world"} for m in matches):
        matches = [
            m for m in matches
            if not (m.family == "strings" and m.keyword.tokens in (("orchestra",), ("orchestral",)))
        ]
    # zither / koto / ... plus harp: keep world, drop the orchestral-harp default
    if any(m.family == "world" for m in matches):
        matches = [
            m for m in matches
            if not (m.family == "strings" and m.keyword.tokens == ("harp",))
        ]

    # 808 resolution
    has_808 = any(m.keyword.tokens[0] == "808" or "808" in m.keyword.tokens for m in matches)
    drums_folder = bool(folder_tokens & {"drums", "drum", "kicks", "kick", "hits", "hats", "snares"})
    bass_folder = "bass" in folder_tokens or "808s" in folder_tokens or "808" in folder_tokens
    kick_ctx = any(m.leaf == "kick" for m in matches) or bool(folder_tokens & {"kick", "kicks"})
    bass_ctx = any(m.family == "bass" and m.leaf != "sub_808" for m in matches) or bass_folder
    # 808 + key in a bass folder, or 808_C pattern, leans bass
    if has_808:
        if kick_ctx:
            matches = [m for m in matches if not (m.family == "bass" and m.keyword.tokens in (("808",), ("808s",)))]
        elif bass_ctx or _808_with_key(filename):
            matches = [m for m in matches if not (m.family == "drums" and m.keyword.tokens in (("808",), ("808s",)))]
        elif drums_folder and not bass_folder:
            # 808 under a drums tree with no kick/bass hint → keep drums, drop default sub_808
            matches = [
                m for m in matches
                if not (m.family == "bass" and m.keyword.tokens in (("808",), ("808s",)) and m.keyword.weak)
            ]
        # else: keep the weak default bass/sub_808 (low confidence later)

    # pad: if drums context ("drum pad" already mapped) leave perc; bare pad → synth
    # already encoded in keywords.

    # hit alone → fx; hit with a real instrument → drop the fx/hit, type handles one_shot
    instrumentish = [
        m for m in matches
        if not (m.family == "fx" and m.keyword.tokens == ("hit",))
        and m.leaf not in {"unknown"}
        and not (m.leaf == "full" and m.keyword.weak)
    ]
    if instrumentish and any(m.keyword.tokens == ("hit",) for m in matches):
        matches = [m for m in matches if m.keyword.tokens != ("hit",)]

    # fx token in the filename usually means the file *is* an FX
    # (``FX_Reversed_Kick``, ``SO_AF_fx_drum_roll``), unless a more specific
    # non-weak instrument is also in the *same* filename and fx is incidental.
    # Treat leading/category "fx" as decisive.
    fx_filename = [
        m for m in matches
        if m.family == "fx" and m.source == "filename" and m.keyword.tokens in (("fx",), ("sfx",))
    ]
    if fx_filename:
        # Keep fx and drop competing instruments from the same filename
        others_fn = [m for m in matches if m.source == "filename" and m.family != "fx"]
        if others_fn:
            matches = [m for m in matches if not (m.source == "filename" and m.family != "fx")]

    # Prefer a specific leaf over a generic family keyword in the same source
    return matches


def _808_with_key(tokens: list[str]) -> bool:
    """``808_C`` / ``808_Am`` — a key token after 808 leans bass."""
    for i, tok in enumerate(tokens):
        if tok in {"808", "808s"} and i + 1 < len(tokens):
            parsed = _parse_key_token(tokens[i + 1], allow_bare=True)
            if parsed:
                return True
    return False


def _specificity(m: Match) -> tuple[int, int, int]:
    """Higher is better.  Prefer longer, non-generic, non-weak, earlier tokens."""
    specific = 0 if m.leaf in GENERIC_LEAVES or m.leaf == "unknown" else 3
    if m.keyword.weak or m.keyword.short:
        specific -= 1
    return (specific, len(m.keyword.tokens), -m.start)


def _pick_instrument(
    matches: list[Match],
    layers: PathLayers,
) -> tuple[str, str, str, str | None, bool, list[str], list[str]]:
    """Return family, leaf, conf, source, conflict, rules, matched tokens."""
    if not matches:
        # Pack-level single-instrument fallback
        pack_hit = _pack_instrument(layers)
        if pack_hit:
            family, leaf, rule = pack_hit
            return family, leaf, "medium", "pack", False, [rule], []
        return "unknown", "unknown", "none", None, False, [], []

    by_source: dict[str, list[Match]] = {}
    for m in matches:
        by_source.setdefault(m.source, []).append(m)

    chosen_source: Source | None = None
    for src in ("filename", "parent", "ancestor", "pack"):
        if by_source.get(src):
            chosen_source = src  # type: ignore[assignment]
            break
    assert chosen_source is not None
    layer = by_source[chosen_source]

    # Conflict: another high-priority source names a different family
    conflict = False
    chosen_best = max(layer, key=_specificity)
    for src, group in by_source.items():
        if src == chosen_source:
            continue
        if _SOURCE_RANK[src] >= 3 and _SOURCE_RANK[chosen_source] >= 3:
            other = max(group, key=_specificity)
            if other.family != chosen_best.family:
                conflict = True

    # Multiple families in the same winning layer
    families = {m.family for m in layer if not (m.keyword.weak and m.leaf in GENERIC_LEAVES)}
    if not families:
        families = {m.family for m in layer}
    if len({m.family for m in layer}) > 1:
        # If a clearly more specific match exists, don't flag mild generics
        top = _specificity(chosen_best)[0]
        rivals = [m for m in layer if m.family != chosen_best.family and _specificity(m)[0] >= top]
        if rivals:
            conflict = True

    family = chosen_best.family
    leaf = chosen_best.leaf

    # If we only have a family-level drums/full from "drum" and a more
    # specific drums leaf is also present, use the specific one.
    same_family = [m for m in layer if m.family == family]
    specific = [m for m in same_family if m.leaf not in GENERIC_LEAVES and m.leaf != "unknown"]
    if specific:
        chosen_best = max(specific, key=_specificity)
        leaf = chosen_best.leaf

    # Family known, leaf not: synth with only "synth" → unknown leaf
    if leaf == "unknown":
        family = chosen_best.family
        leaf = "unknown"

    conf = _instrument_confidence(chosen_best, chosen_source, conflict)
    rules = [
        f"instrument:{chosen_source}:{chosen_best.keyword.key}->{family}/{leaf}"
    ]
    tokens = list(chosen_best.keyword.tokens)
    if conflict:
        rules.append("instrument:conflict")
    return family, leaf, conf, chosen_source, conflict, rules, tokens


def _instrument_confidence(match: Match, source: Source, conflict: bool) -> Conf:
    if conflict or match.keyword.weak:
        return "low"
    if match.keyword.short:
        return "medium"
    if source in {"filename", "parent"}:
        return "high"
    if source in {"ancestor", "pack"}:
        return "medium"
    return "low"


def _pack_instrument(layers: PathLayers) -> tuple[str, str, str] | None:
    name = layers.pack.lower()
    for needle, family, leaf in SINGLE_INSTRUMENT_PACKS:
        if needle in name:
            return family, leaf, f"instrument:pack:{needle}->{family}/{leaf}"
    return None


def _extract_instrument(
    layers: PathLayers,
) -> tuple[str, str, str, str | None, bool, list[str], list[str]]:
    family_hints = _family_hint(layers)
    raw: list[Match] = []
    tags: list[str] = []
    for source, tokens in _iter_layers(layers):
        for m in _raw_matches(tokens, source):
            if m.keyword.tag_only:
                tags.append(f"tag:{source}:{m.keyword.key}")
                continue
            raw.append(m)
    filtered = _filter_matches(raw, layers, family_hints)
    disamb = _disambiguate(filtered, layers)
    family, leaf, conf, source, conflict, rules, toks = _pick_instrument(disamb, layers)
    # 808 default lean is low confidence (design §1.3 / §2.4)
    if any(
        m.keyword.tokens in (("808",), ("808s",), ("sub",)) and m.keyword.weak
        for m in disamb
        if m.leaf == leaf
    ) and conf == "low":
        rules.append("instrument:ambiguous-default")
    rules.extend(tags)
    return family, leaf, conf, source, conflict, rules, toks


# ---------------------------------------------------------------------------
# Type (loop / one_shot)
# ---------------------------------------------------------------------------

def _extract_type(layers: PathLayers) -> tuple[str, str, str | None, list[str], list[str]]:
    """Return type, conf, source, rules, tokens."""
    # "os" only with context (One_Shots folder, or not a leading pack prefix).
    candidates: list[tuple[str, Source, str, list[str], bool]] = []
    # tuple: type, source, token, tokens, is_short

    def consider(tokens: list[str], source: Source, *, skip_leading_os: bool = False) -> None:
        toks = tokens
        start = 0
        if skip_leading_os and toks and toks[0] == "os":
            start = 1
        toks = toks[start:]
        grams = list(zip(toks, toks[1:]))
        if ("one", "shot") in grams or ("one", "shots") in grams:
            candidates.append(("one_shot", source, "one shot", ["one", "shot"], False))
        if any(t in TYPE_LOOP_TOKENS or t == "looping" for t in toks):
            hit = next(t for t in toks if t in TYPE_LOOP_TOKENS or t == "looping")
            candidates.append(("loop", source, hit, [hit], hit == "lp"))
        if any(t in TYPE_ONESHOT_TOKENS for t in toks):
            hit = next(t for t in toks if t in TYPE_ONESHOT_TOKENS)
            # "hit" / "hits" as type only; already also an instrument cue
            candidates.append(("one_shot", source, hit, [hit], False))
        # Contextual "os" (not leading pack prefix, and not short unless folder)
        if "os" in toks and not (skip_leading_os and tokens and tokens[0] == "os"):
            # Accept "os" as one_shot only outside the leading-prefix case,
            # and only if a type-ish folder is nearby or it's not first.
            if source in {"parent", "ancestor"} or (source == "filename" and tokens.index("os") > 0):
                candidates.append(("one_shot", source, "os", ["os"], True))

    consider(layers.filename, "filename", skip_leading_os=True)
    if layers.parent:
        consider(layers.parent, "parent")
    for anc in layers.ancestors:
        consider(anc, "ancestor")

    if not candidates:
        return "unknown", "none", None, [], []

    # Prefer filename, then parent, then ancestor; prefer explicit over short
    def rank(c: tuple[str, Source, str, list[str], bool]) -> tuple[int, int]:
        typ, source, token, toks, is_short = c
        return (_SOURCE_RANK[source], 0 if is_short else 1)

    typ, source, token, toks, is_short = max(candidates, key=rank)
    # If filename and parent disagree, filename wins and we already picked by rank
    conflict = any(c[0] != typ and c[1] in {"filename", "parent"} and source in {"filename", "parent"} for c in candidates)
    if source in {"filename", "parent"} and not is_short and not conflict:
        conf: Conf = "high"
    elif is_short or source == "ancestor":
        conf = "medium"
    else:
        conf = "medium"
    if conflict:
        conf = "low"
    rule = f"type:{source}:{token}->{typ}"
    return typ, conf, source, [rule], toks


# ---------------------------------------------------------------------------
# BPM
# ---------------------------------------------------------------------------

def _as_int(tok: str) -> int | None:
    if tok.isdigit() and not (len(tok) > 1 and tok.startswith("0") and tok != "0"):
        # allow "80", "120"; reject "003", "01" as BPM (serials) unless
        # the caller already validated via an explicit "bpm" neighbor.
        try:
            return int(tok)
        except ValueError:
            return None
    if tok.isdigit():
        try:
            return int(tok)
        except ValueError:
            return None
    return None


def _in_range(n: int, lo_hi: tuple[int, int]) -> bool:
    lo, hi = lo_hi
    return lo <= n <= hi


def _extract_bpm(layers: PathLayers) -> tuple[int | None, str | None, list[str]]:
    # 1. Number adjacent to a "bpm" token, any layer (filename first).
    for source, tokens in _iter_layers(layers):
        for i, tok in enumerate(tokens):
            if tok != "bpm":
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(tokens):
                    n = _as_int(tokens[j])
                    if n is not None and _in_range(n, BPM_EXPLICIT_RANGE):
                        return n, source, [f"bpm:{source}:adjacent:{n}"]

    # 2. Splice prefix template: one or two letter codes then a BPM
    #    (CAP_120_..., SO_LI_95_..., SC_ES_126_..., OS_MTS_150_...).
    fn = layers.filename
    i = 0
    while i < len(fn) and i < 2 and fn[i].isalpha() and 2 <= len(fn[i]) <= 8:
        i += 1
    if i >= 1 and i < len(fn):
        n = _as_int(fn[i])
        raw = fn[i]
        if n is not None and _in_range(n, BPM_CONTEXT_RANGE) and not raw.startswith("0"):
            return n, "filename", [f"bpm:filename:splice_prefix:{n}"]

    # 3. Number glued to an instrument/type token (am_drm80_..., am_perc80_...)
    #    already split into [drm, 80]; accept 60–200 next to those tokens.
    _ctx = TYPE_LOOP_TOKENS | {
        "drm", "drum", "drums", "perc", "kick", "snare", "hat", "loop", "kit",
    }
    for i, tok in enumerate(fn):
        n = _as_int(tok)
        if n is None or not _in_range(n, BPM_CONTEXT_RANGE):
            continue
        neighbors = []
        if i > 0:
            neighbors.append(fn[i - 1])
        if i + 1 < len(fn):
            neighbors.append(fn[i + 1])
        if any(nb in _ctx for nb in neighbors):
            return n, "filename", [f"bpm:filename:adjacent_context:{n}"]

    # 4. ``_NNN_`` 3-digit 60–200 in the filename when a loop is also in the path
    has_loop = any(
        t in TYPE_LOOP_TOKENS
        for t in layers.filename + layers.parent + [x for a in layers.ancestors for x in a]
    )
    if has_loop:
        for tok in fn:
            n = _as_int(tok)
            if n is not None and len(tok) == 3 and _in_range(n, BPM_CONTEXT_RANGE):
                if tok.startswith("0"):
                    continue
                return n, "filename", [f"bpm:filename:loop_number:{n}"]
        # 2- or 3-digit trailing number 60–200 with loop (OLIVER_..._90)
        if fn:
            n = _as_int(fn[-1])
            if (
                n is not None
                and _in_range(n, BPM_CONTEXT_RANGE)
                and not fn[-1].startswith("0")
                and not (len(fn[-1]) == 2 and n < 60)
            ):
                return n, "filename", [f"bpm:filename:trailing_with_loop:{n}"]

    # 5. Parent / ancestor folder is a bare number (``120``, ``100``) or
    #    contains a number + bpm (already handled in step 1).
    for source, tokens, name in (
        ("parent", layers.parent, layers.parent_name or ""),
        *[("ancestor", a, n) for a, n in zip(layers.ancestors, layers.ancestor_names)],
    ):
        if len(tokens) == 1:
            n = _as_int(tokens[0])
            if n is not None and _in_range(n, BPM_CONTEXT_RANGE) and not tokens[0].startswith("0"):
                return n, source, [f"bpm:{source}:folder:{n}"]
        # ``drum_120``, ``Drum_4_175``, ``Drum_Loop_4_149``
        nums = [(_as_int(t), t) for t in tokens]
        for n, raw in nums:
            if n is not None and len(raw) == 3 and _in_range(n, BPM_CONTEXT_RANGE) and not raw.startswith("0"):
                return n, source, [f"bpm:{source}:folder_number:{n}"]

    return None, None, []


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------

def _parse_key_token(tok: str, *, allow_bare: bool) -> tuple[str, str] | None:
    m = _KEY_RE.fullmatch(tok)
    if not m:
        return None
    root_l, acc, mode = m.group(1), m.group(2), m.group(3)
    # Bare letter (or bare + accidental without mode) only when allowed.
    if mode is None and not allow_bare:
        # Accidental-only (F#, Bb) is still a key if allow_bare; without it, skip
        return None
    if mode is None and acc is None and not allow_bare:
        return None

    root = root_l.upper()
    if acc in {"b", "flat"}:
        spelled = FLAT_TO_SHARP.get(root + "b", root + "b")
        # FLAT_TO_SHARP keys are like "Bb"; root is "B", acc is "b"
        spelled = FLAT_TO_SHARP.get(root + "b", None)
        if spelled is None and root + "b" in {"Cb", "Fb"}:
            spelled = FLAT_TO_SHARP[root + "b"]
        if spelled is None:
            # Unusual flat (e.g. "Fb" already in map).  If someone wrote "Cb"
            # we're good; "Gb" too.  "Fb" too.  Unknown flats keep letter+b.
            spelled = root + "b"
    elif acc in {"#", "sharp", "s"}:
        spelled = root + "#"
        spelled = SHARP_TO_CANON.get(spelled, spelled)
    else:
        spelled = root

    if mode in {"m", "min", "minor"}:
        key_mode = "minor"
    elif mode in {"maj", "major"}:
        key_mode = "major"
    else:
        key_mode = "unknown"
    return spelled, key_mode


def _is_key_position(tokens: list[str], index: int, *, is_filename: bool) -> bool:
    """Bare A/E tokens count only at the filename end or next to bpm/mode.

    Tokens with an accidental (``D#``, ``Bb``) or an explicit mode (``Am``,
    ``Fmaj``) are distinctive enough to accept in any slot except a leading
    pack prefix.  A ``bpm`` token anywhere in the same folder also unlocks
    bare letters (``EDM_Kit_G_128bpm``).
    """
    tok = tokens[index]
    m = _KEY_RE.fullmatch(tok)
    if not m:
        return False
    acc, mode = m.group(2), m.group(3)
    if index == 0 and is_filename and tok in PACK_PREFIX_TOKENS:
        return False
    if mode is not None or acc is not None:
        return True
    neighbors = []
    if index > 0:
        neighbors.append(tokens[index - 1])
    if index + 1 < len(tokens):
        neighbors.append(tokens[index + 1])
    if any(n == "bpm" or n in {"min", "minor", "maj", "major"} for n in neighbors):
        return True
    if "bpm" in tokens:
        return True
    if is_filename and index == len(tokens) - 1:
        return True
    return False


def _extract_key(layers: PathLayers) -> tuple[str | None, str | None, list[str]]:
    def scan(tokens: list[str], source: Source, *, is_filename: bool) -> tuple[str, str, str] | None:
        if not tokens:
            return None
        # Prefer the rightmost key token (Splice puts the key at the end).
        for i in range(len(tokens) - 1, -1, -1):
            tok = tokens[i]
            allow_bare = _is_key_position(tokens, i, is_filename=is_filename)
            # Leading pack prefix must never be a key
            if is_filename and i == 0 and tok in PACK_PREFIX_TOKENS:
                continue
            parsed = _parse_key_token(tok, allow_bare=allow_bare)
            if not parsed:
                continue
            if not allow_bare and _KEY_RE.fullmatch(tok) and _KEY_RE.fullmatch(tok).group(3) is None:
                continue
            return parsed[0], parsed[1], tok
        return None

    # Filename first, then parent, then ancestors.  Never the pack name.
    hit = scan(layers.filename, "filename", is_filename=True)
    source: Source = "filename"
    if not hit and layers.parent:
        hit = scan(layers.parent, "parent", is_filename=False)
        source = "parent"
    if not hit:
        for anc in layers.ancestors:
            hit = scan(anc, "ancestor", is_filename=False)
            if hit:
                source = "ancestor"
                break
    if not hit:
        return None, None, []
    root, mode, raw = hit
    return root, mode, [f"key:{source}:{raw}->{root} {mode}"]


# ---------------------------------------------------------------------------
# Genre
# ---------------------------------------------------------------------------

def _extract_genre(layers: PathLayers) -> tuple[str | None, list[str]]:
    # Prefer pack name, then any tokens.
    pools = [layers.pack_tokens, layers.all_tokens]
    for tokens in pools:
        grams = {pair for pair in bigrams(tokens)}
        tokset = set(tokens)
        for phrase, genre in GENRE_PHRASES:
            if len(phrase) == 1:
                if phrase[0] in tokset:
                    return genre, [f"genre:pack:{phrase[0]}"]
            elif len(phrase) == 2:
                if (phrase[0], phrase[1]) in grams:
                    return genre, [f"genre:pack:{' '.join(phrase)}"]
            else:
                # consecutive phrase
                if any(tokens[i:i + len(phrase)] == list(phrase) for i in range(len(tokens) - len(phrase) + 1)):
                    return genre, [f"genre:pack:{' '.join(phrase)}"]
    return None, []


# ---------------------------------------------------------------------------
# Public helpers used by the script
# ---------------------------------------------------------------------------

def should_skip(rel_path: str) -> bool:
    return is_junk_path(rel_path)
