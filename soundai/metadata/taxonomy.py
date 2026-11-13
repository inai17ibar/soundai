"""Instrument taxonomy, keyword dictionaries, and normalization tables.

Keywords follow docs/condition-eval-design.md §1.  Matching is token-based
after the tokenizer in ``soundai.metadata.tokenize``.
"""

from __future__ import annotations

from dataclasses import dataclass

RULES_VERSION = "0.1.0"

FAMILIES: tuple[str, ...] = (
    "drums",
    "bass",
    "keys",
    "synth",
    "guitar",
    "strings",
    "brass",
    "woodwind",
    "mallet",
    "vocal",
    "fx",
    "world",
    "unknown",
)

LEAVES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "drums": ("kick", "snare", "clap", "hihat", "cymbal", "tom", "perc", "full", "unknown"),
    "bass": ("synth_bass", "sub_808", "acoustic_bass", "unknown"),
    "keys": ("piano", "epiano", "organ", "other_keys", "unknown"),
    "synth": ("lead", "pad", "pluck", "arp", "chord", "unknown"),
    "guitar": ("electric", "acoustic", "unknown"),
    "strings": ("strings", "unknown"),
    "brass": ("brass", "unknown"),
    "woodwind": ("woodwind", "unknown"),
    "mallet": ("mallet", "unknown"),
    "vocal": ("vocal", "unknown"),
    "fx": ("fx", "unknown"),
    "world": ("world", "unknown"),
    "unknown": ("unknown",),
}

LEAF_TO_FAMILY: dict[str, str] = {
    leaf: family
    for family, leaves in LEAVES_BY_FAMILY.items()
    for leaf in leaves
    if not (family == "unknown" and leaf == "unknown")
}
LEAF_TO_FAMILY["unknown"] = "unknown"

# Leaves that come from a generic family word ("bass", "drums", "keys") rather
# than a specific instrument.  Prefer a more specific leaf when both match.
GENERIC_LEAVES = frozenset({
    "full",
    "synth_bass",
    "other_keys",
    "electric",  # default guitar
    "strings",
    "brass",
    "woodwind",
    "mallet",
    "vocal",
    "fx",
    "world",
    "unknown",
    "chord",  # generic "chord(s)" — keep below guitar/piano/keys
    "lead",   # generic "lead" — keep below vocal/guitar
    "pad",
})

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Keyword:
    """A unigram or phrase that maps to a family/leaf.

    Parameters
    ----------
    tokens:
        One or more lowercase tokens that must appear in order (adjacent).
    family, leaf:
        Taxonomy coordinates.
    short:
        If True, only accept the match when a higher-level folder already
        suggests the same family (design §1.3: sn, ep, os, ld, fl, …).
    weak:
        If True, only accept in the filename or the immediate parent — not
        in distant ancestors or pack-title folders (e.g. "beat", "kit").
    tag_only:
        Record the match for debugging but never use it as the instrument
        (e.g. ``rev`` / ``reverse``).
    """

    tokens: tuple[str, ...]
    family: str
    leaf: str
    short: bool = False
    weak: bool = False
    tag_only: bool = False

    @property
    def key(self) -> str:
        return " ".join(self.tokens)


def _kw(*token: str, family: str, leaf: str, short: bool = False, weak: bool = False, tag_only: bool = False) -> Keyword:
    return Keyword(tuple(token), family, leaf, short=short, weak=weak, tag_only=tag_only)


# Longer phrases first is handled at match time by token-length, not list order.
INSTRUMENT_KEYWORDS: tuple[Keyword, ...] = (
    # --- drums: kick ---
    _kw("bass", "drum", family="drums", leaf="kick"),
    _kw("bassdrum", family="drums", leaf="kick"),
    _kw("kick", "drum", family="drums", leaf="kick"),
    _kw("kickdrum", family="drums", leaf="kick"),
    _kw("sub", "kick", family="drums", leaf="kick"),
    _kw("808", "kick", family="drums", leaf="kick"),
    _kw("808", "bd", family="drums", leaf="kick"),
    _kw("kick", family="drums", leaf="kick"),
    _kw("kicks", family="drums", leaf="kick"),
    _kw("kik", family="drums", leaf="kick"),
    _kw("kck", family="drums", leaf="kick", short=True),
    _kw("bd", family="drums", leaf="kick", short=True),  # drums-folder only via short
    # --- drums: snare ---
    _kw("rim", "shot", family="drums", leaf="snare"),
    _kw("rimshot", family="drums", leaf="snare"),
    _kw("snare", family="drums", leaf="snare"),
    _kw("snares", family="drums", leaf="snare"),
    _kw("snr", family="drums", leaf="snare", short=True),
    _kw("sn", family="drums", leaf="snare", short=True),
    _kw("sd", family="drums", leaf="snare", short=True),
    _kw("rim", family="drums", leaf="snare"),
    _kw("rims", family="drums", leaf="snare"),
    # --- drums: clap ---
    _kw("hand", "clap", family="drums", leaf="clap"),
    _kw("handclap", family="drums", leaf="clap"),
    _kw("clap", family="drums", leaf="clap"),
    _kw("claps", family="drums", leaf="clap"),
    _kw("clp", family="drums", leaf="clap", short=True),
    _kw("snap", family="drums", leaf="clap"),
    _kw("snaps", family="drums", leaf="clap"),
    # --- drums: hihat ---
    _kw("closed", "hat", family="drums", leaf="hihat"),
    _kw("open", "hat", family="drums", leaf="hihat"),
    _kw("closed", "hats", family="drums", leaf="hihat"),
    _kw("open", "hats", family="drums", leaf="hihat"),
    _kw("closedhat", family="drums", leaf="hihat"),
    _kw("openhat", family="drums", leaf="hihat"),
    _kw("hi", "hat", family="drums", leaf="hihat"),
    _kw("hi", "hats", family="drums", leaf="hihat"),
    _kw("hihat", family="drums", leaf="hihat"),
    _kw("hihats", family="drums", leaf="hihat"),
    _kw("hat", family="drums", leaf="hihat"),
    _kw("hats", family="drums", leaf="hihat"),
    _kw("chh", family="drums", leaf="hihat"),
    _kw("ohh", family="drums", leaf="hihat"),
    _kw("hh", family="drums", leaf="hihat", short=True),
    # --- drums: cymbal ---
    _kw("cymbal", family="drums", leaf="cymbal"),
    _kw("cymbals", family="drums", leaf="cymbal"),
    _kw("cym", family="drums", leaf="cymbal", short=True),
    _kw("crash", family="drums", leaf="cymbal"),
    _kw("crashes", family="drums", leaf="cymbal"),
    _kw("ride", family="drums", leaf="cymbal"),
    _kw("splash", family="drums", leaf="cymbal"),
    _kw("china", family="drums", leaf="cymbal"),
    # --- drums: tom ---
    _kw("floor", "tom", family="drums", leaf="tom"),
    _kw("floortom", family="drums", leaf="tom"),
    _kw("hi", "tom", family="drums", leaf="tom"),
    _kw("lo", "tom", family="drums", leaf="tom"),
    _kw("low", "tom", family="drums", leaf="tom"),
    _kw("tom", family="drums", leaf="tom"),
    _kw("toms", family="drums", leaf="tom"),
    # --- drums: perc ---
    _kw("drum", "pad", family="drums", leaf="perc"),
    _kw("pad", "hit", family="drums", leaf="perc"),
    _kw("percussion", family="drums", leaf="perc"),
    _kw("perc", family="drums", leaf="perc"),
    _kw("percs", family="drums", leaf="perc"),
    _kw("shaker", family="drums", leaf="perc"),
    _kw("shakers", family="drums", leaf="perc"),
    _kw("shkr", family="drums", leaf="perc", short=True),
    _kw("tambourine", family="drums", leaf="perc"),
    _kw("tamb", family="drums", leaf="perc", short=True),
    _kw("conga", family="drums", leaf="perc"),
    _kw("congas", family="drums", leaf="perc"),
    _kw("bongo", family="drums", leaf="perc"),
    _kw("bongos", family="drums", leaf="perc"),
    _kw("cowbell", family="drums", leaf="perc"),
    _kw("clave", family="drums", leaf="perc"),
    _kw("triangle", family="drums", leaf="perc", weak=True),
    _kw("block", family="drums", leaf="perc", weak=True),
    # --- drums: full ---
    _kw("drum", "loop", family="drums", leaf="full"),
    _kw("drum", "loops", family="drums", leaf="full"),
    _kw("full", "drums", family="drums", leaf="full"),
    _kw("top", "loop", family="drums", leaf="full"),
    _kw("top", "loops", family="drums", leaf="full"),
    _kw("drumloop", family="drums", leaf="full"),
    _kw("drumloops", family="drums", leaf="full"),
    _kw("drums", family="drums", leaf="full"),
    _kw("drum", family="drums", leaf="full", weak=True),
    _kw("drm", family="drums", leaf="full"),
    _kw("beat", family="drums", leaf="full", weak=True),
    _kw("beats", family="drums", leaf="full", weak=True),
    _kw("groove", family="drums", leaf="full", weak=True),
    _kw("grooves", family="drums", leaf="full", weak=True),
    _kw("kit", family="drums", leaf="full", weak=True),
    _kw("tops", family="drums", leaf="full", weak=True),
    _kw("fill", family="drums", leaf="full", weak=True),
    _kw("fills", family="drums", leaf="full", weak=True),
    _kw("break", family="drums", leaf="full", weak=True),
    _kw("breaks", family="drums", leaf="full", weak=True),
    # --- bass ---
    _kw("synth", "bass", family="bass", leaf="synth_bass"),
    _kw("bass", "guitar", family="bass", leaf="acoustic_bass"),
    _kw("electric", "bass", family="bass", leaf="acoustic_bass"),
    _kw("double", "bass", family="bass", leaf="acoustic_bass"),
    _kw("upright", "bass", family="bass", leaf="acoustic_bass"),
    _kw("upright", family="bass", leaf="acoustic_bass", short=True),
    _kw("sub", "bass", family="bass", leaf="sub_808"),
    _kw("subbass", family="bass", leaf="sub_808"),
    _kw("808", "bass", family="bass", leaf="sub_808"),
    _kw("808", "sub", family="bass", leaf="sub_808"),
    _kw("sub", "808", family="bass", leaf="sub_808"),
    _kw("808s", family="bass", leaf="sub_808"),
    _kw("808", family="bass", leaf="sub_808", weak=True),  # resolved further in extractor
    _kw("sub", family="bass", leaf="sub_808", weak=True),
    _kw("reese", family="bass", leaf="synth_bass"),
    _kw("wobble", family="bass", leaf="synth_bass"),
    _kw("bass", family="bass", leaf="synth_bass"),
    _kw("bs", family="bass", leaf="synth_bass", short=True),
    # --- keys ---
    _kw("upright", "piano", family="keys", leaf="piano"),
    _kw("e", "piano", family="keys", leaf="epiano"),
    _kw("epiano", family="keys", leaf="epiano"),
    _kw("wurlitzer", family="keys", leaf="epiano"),
    _kw("wurli", family="keys", leaf="epiano"),
    _kw("rhodes", family="keys", leaf="epiano"),
    _kw("piano", family="keys", leaf="piano"),
    _kw("pianos", family="keys", leaf="piano"),
    _kw("pno", family="keys", leaf="piano", short=True),
    _kw("grand", family="keys", leaf="piano", short=True),
    _kw("hammond", family="keys", leaf="organ"),
    _kw("organ", family="keys", leaf="organ"),
    _kw("org", family="keys", leaf="organ", short=True),
    _kw("b3", family="keys", leaf="organ", short=True),
    _kw("ep", family="keys", leaf="epiano", short=True),
    _kw("clav", family="keys", leaf="other_keys"),
    _kw("celesta", family="keys", leaf="other_keys"),
    _kw("harpsichord", family="keys", leaf="other_keys"),
    _kw("keys", family="keys", leaf="other_keys"),
    _kw("key", family="keys", leaf="other_keys", short=True),
    # --- synth ---
    _kw("arpeggio", family="synth", leaf="arp"),
    _kw("arp", family="synth", leaf="arp"),
    _kw("sequence", family="synth", leaf="arp", weak=True),
    _kw("seq", family="synth", leaf="arp", short=True),
    _kw("atmosphere", family="synth", leaf="pad"),
    _kw("atmos", family="synth", leaf="pad"),
    _kw("drone", family="synth", leaf="pad"),
    _kw("ambient", family="synth", leaf="pad", weak=True),
    _kw("pad", family="synth", leaf="pad"),
    _kw("pads", family="synth", leaf="pad"),
    _kw("pd", family="synth", leaf="pad", short=True),
    _kw("pluck", family="synth", leaf="pluck"),
    _kw("plucks", family="synth", leaf="pluck"),
    _kw("plk", family="synth", leaf="pluck", short=True),
    _kw("stab", family="synth", leaf="chord"),
    _kw("stabs", family="synth", leaf="chord"),
    _kw("chord", family="synth", leaf="chord"),
    _kw("chords", family="synth", leaf="chord"),
    _kw("lead", family="synth", leaf="lead"),
    _kw("leads", family="synth", leaf="lead"),
    _kw("ld", family="synth", leaf="lead", short=True),
    _kw("synth", family="synth", leaf="unknown"),
    _kw("synths", family="synth", leaf="unknown"),
    # --- guitar ---
    _kw("electric", "guitar", family="guitar", leaf="electric"),
    _kw("acoustic", "guitar", family="guitar", leaf="acoustic"),
    _kw("power", "chord", family="guitar", leaf="electric"),
    _kw("guitar", family="guitar", leaf="electric"),
    _kw("guitars", family="guitar", leaf="electric"),
    _kw("gtr", family="guitar", leaf="electric"),
    _kw("egtr", family="guitar", leaf="electric"),
    _kw("agtr", family="guitar", leaf="acoustic"),
    _kw("nylon", family="guitar", leaf="acoustic", short=True),
    _kw("steel", family="guitar", leaf="acoustic", short=True),  # overridden by steel drum
    _kw("riff", family="guitar", leaf="electric"),
    _kw("riffs", family="guitar", leaf="electric"),
    _kw("strummed", family="guitar", leaf="electric"),
    _kw("strum", family="guitar", leaf="electric", weak=True),
    _kw("picked", family="guitar", leaf="electric", weak=True),
    _kw("electric", family="guitar", leaf="electric", weak=True),
    _kw("acoustic", family="guitar", leaf="acoustic", weak=True),
    # --- strings ---
    _kw("pizzicato", family="strings", leaf="strings"),
    _kw("pizz", family="strings", leaf="strings", short=True),
    _kw("orchestra", family="strings", leaf="strings", weak=True),
    _kw("orchestral", family="strings", leaf="strings", weak=True),
    _kw("strings", family="strings", leaf="strings"),
    _kw("string", family="strings", leaf="strings"),
    _kw("violin", family="strings", leaf="strings"),
    _kw("vln", family="strings", leaf="strings", short=True),
    _kw("viola", family="strings", leaf="strings"),
    _kw("cello", family="strings", leaf="strings"),
    _kw("harp", family="strings", leaf="strings"),  # orchestral harp; kora is world
    _kw("str", family="strings", leaf="strings", short=True),
    # --- brass ---
    _kw("brass", family="brass", leaf="brass"),
    _kw("horn", family="brass", leaf="brass"),
    _kw("horns", family="brass", leaf="brass"),
    _kw("trumpet", family="brass", leaf="brass"),
    _kw("tpt", family="brass", leaf="brass", short=True),
    _kw("trombone", family="brass", leaf="brass"),
    _kw("tbn", family="brass", leaf="brass", short=True),
    # --- woodwind (sax defaults here; pack may override) ---
    _kw("woodwind", family="woodwind", leaf="woodwind"),
    _kw("woodwinds", family="woodwind", leaf="woodwind"),
    _kw("winds", family="woodwind", leaf="woodwind"),
    _kw("flute", family="woodwind", leaf="woodwind"),
    _kw("flutes", family="woodwind", leaf="woodwind"),
    _kw("fl", family="woodwind", leaf="woodwind", short=True),
    _kw("clarinet", family="woodwind", leaf="woodwind"),
    _kw("oboe", family="woodwind", leaf="woodwind"),
    _kw("bassoon", family="woodwind", leaf="woodwind"),
    _kw("sax", family="woodwind", leaf="woodwind"),
    _kw("saxophone", family="woodwind", leaf="woodwind"),
    # --- mallet ---
    _kw("marimba", family="mallet", leaf="mallet"),
    _kw("vibraphone", family="mallet", leaf="mallet"),
    _kw("vibes", family="mallet", leaf="mallet"),
    _kw("glock", family="mallet", leaf="mallet"),
    _kw("kalimba", family="mallet", leaf="mallet"),
    _kw("bell", family="mallet", leaf="mallet", weak=True),
    _kw("bells", family="mallet", leaf="mallet"),
    # --- vocal ---
    _kw("vocal", "lead", family="vocal", leaf="vocal"),
    _kw("lead", "vocal", family="vocal", leaf="vocal"),
    _kw("acapella", family="vocal", leaf="vocal"),
    _kw("a", "cappella", family="vocal", leaf="vocal"),
    _kw("choir", family="vocal", leaf="vocal"),
    _kw("chant", family="vocal", leaf="vocal"),
    _kw("chants", family="vocal", leaf="vocal"),
    _kw("adlib", family="vocal", leaf="vocal"),
    _kw("vocal", family="vocal", leaf="vocal"),
    _kw("vocals", family="vocal", leaf="vocal"),
    _kw("vox", family="vocal", leaf="vocal"),
    _kw("voc", family="vocal", leaf="vocal", short=True),
    _kw("spoken", family="vocal", leaf="vocal", short=True),
    _kw("phrase", family="vocal", leaf="vocal", short=True),  # too generic alone
    _kw("shout", family="vocal", leaf="vocal", weak=True),
    _kw("shouts", family="vocal", leaf="vocal"),
    _kw("beatbox", family="vocal", leaf="vocal"),
    _kw("beatboxing", family="vocal", leaf="vocal"),
    # --- fx ---
    _kw("riser", family="fx", leaf="fx"),
    _kw("uplifter", family="fx", leaf="fx"),
    _kw("downlifter", family="fx", leaf="fx"),
    _kw("sweep", family="fx", leaf="fx"),
    _kw("impact", family="fx", leaf="fx"),
    _kw("whoosh", family="fx", leaf="fx"),
    _kw("foley", family="fx", leaf="fx"),
    _kw("transition", family="fx", leaf="fx"),
    _kw("noise", family="fx", leaf="fx", weak=True),
    _kw("crackle", family="fx", leaf="fx"),
    _kw("texture", family="fx", leaf="fx", weak=True),
    _kw("notification", family="fx", leaf="fx"),
    _kw("interface", family="fx", leaf="fx"),
    _kw("button", family="fx", leaf="fx", weak=True),
    _kw("sfx", family="fx", leaf="fx"),
    _kw("fx", family="fx", leaf="fx"),
    _kw("hit", family="fx", leaf="fx", weak=True),  # resolved further in extractor
    _kw("reverse", family="fx", leaf="fx", tag_only=True),
    _kw("reversed", family="fx", leaf="fx", tag_only=True),
    _kw("rev", family="fx", leaf="fx", tag_only=True),
    # --- world ---
    _kw("steel", "drum", family="world", leaf="world"),
    _kw("ukulele", family="world", leaf="world"),
    _kw("ukelele", family="world", leaf="world"),
    _kw("uke", family="world", leaf="world", short=True),
    _kw("sitar", family="world", leaf="world"),
    _kw("koto", family="world", leaf="world"),
    _kw("shamisen", family="world", leaf="world"),
    _kw("tabla", family="world", leaf="world"),
    _kw("kora", family="world", leaf="world"),
    _kw("zither", family="world", leaf="world"),
    _kw("doumbek", family="world", leaf="world"),
    _kw("taiko", family="world", leaf="world", weak=True),  # often a pack prefix
)

# Tokens that look like instruments but are usually pack / vendor codes when
# they are the *first* token of a filename.
PACK_PREFIX_TOKENS = frozenset({
    "am", "af", "bbl", "bedms", "bpe", "cap", "cpa", "dch", "es", "esm",
    "kshmr", "li", "mbb", "mts", "murda", "oliver", "os", "rare", "re",
    "sb", "sc", "so", "stp", "taiko", "tp", "zenhiser", "esm",
})

# Words that negate a following (or surrounding) instrument keyword.
NEGATION_TOKENS = frozenset({"no", "not", "without", "nokick"})

# Equipment names that contain 808 but are not the instrument.
TR808_TOKENS = frozenset({"tr808", "tr-808"})

# Immediate-parent / ancestor folder names that indicate a type.
TYPE_LOOP_TOKENS = frozenset({
    "loop", "loops", "lp", "looping",
})
TYPE_ONESHOT_TOKENS = frozenset({
    "oneshot", "oneshots", "shot", "shots", "hit", "hits", "single",
})
# "os" is applied only with family/type context (it is also a pack prefix).

# Packs whose name alone is a reliable single-instrument signal.
# Matched against the lowercased pack directory name as a substring.
SINGLE_INSTRUMENT_PACKS: tuple[tuple[str, str, str], ...] = (
    ("guitar loops and riffs", "guitar", "electric"),
    ("jazz drums", "drums", "full"),
    ("loud drums", "drums", "full"),
    ("ultimate drum loops", "drums", "full"),
    ("andreas klein", "drums", "full"),
)

# Genre cues, mainly from pack names (design §1.2).  Longer phrases first.
GENRE_PHRASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("drum", "bass"), "dnb"),
    (("d", "n", "b"), "dnb"),
    (("hip", "hop"), "hiphop"),
    (("lo", "fi"), "lofi"),
    (("r", "n", "b"), "rnb"),
    (("rnb",), "rnb"),
    (("lofi",), "lofi"),
    (("hiphop",), "hiphop"),
    (("house",), "house"),
    (("techno",), "techno"),
    (("trap",), "trap"),
    (("drill",), "drill"),
    (("edm",), "edm"),
    (("soul",), "soul"),
    (("jazz",), "jazz"),
    (("ambient",), "ambient"),
    (("afro",), "afro"),
    (("motown",), "motown"),
    (("disco",), "disco"),
    (("rock",), "rock"),
    (("pop",), "pop"),
    (("cinematic",), "cinematic"),
    (("electro",), "electro"),
    (("electronica",), "electronica"),
    (("warehouse",), "techno"),
    (("tropical",), "tropical"),
    (("dnb",), "dnb"),
    (("jungle",), "dnb"),
    (("dubstep",), "dubstep"),
    (("moombahton",), "moombahton"),
    (("moombah",), "moombahton"),
)

# Key accidental normalization: flats written as sharps (design §2.3).
FLAT_TO_SHARP = {
    "Bb": "A#",
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Cb": "B",
    "Fb": "E",
}
SHARP_TO_CANON = {
    "E#": "F",
    "B#": "C",
}

BPM_EXPLICIT_RANGE = (40, 220)
BPM_CONTEXT_RANGE = (60, 200)

# Tokens that are never keys, even when they match the key regex.
KEY_STOPWORDS = frozenset({
    "am",  # Ambient pack prefix (when in prefix position; end-of-name still allowed)
    "re", "so", "mi", "fa", "ti", "la",  # solfege / pack codes
    "es", "os", "as",  # pack prefixes; "as" as A# is still allowed in suffix position
})
