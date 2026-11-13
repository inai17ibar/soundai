"""Focused tests for the rule-based path metadata extractor."""

from __future__ import annotations

from soundai.metadata.extractor import extract_metadata
from soundai.metadata.tokenize import split_rel_path, tokenize_component


CAPSUN_PIANO = (
    "Capsun - Lofi Love Songs/CPA_Capsun_Lofi_Love_Songs_Sample_Pack/"
    "Melodic_Loops/Piano_&_Keys/CAP_120_tape_pianos_basement_find_Fm.wav"
)
BEDROOM_SNARE = (
    "Bedroom Beats & Lofi Hip-Hop/CPA_Bedroom_Beats___LoFi_HipHop/"
    "Drum___Perc_One_Shots/Snares/BBL_snare_flat_single.wav"
)
TAIKO_REVERSE = (
    "TAIKO Essentials Sample Pack/TAIKO_tonal/TAIKO_melodic_loops/"
    "TAIKO_87_melodic_loop_reverse_debilidad_Dbmin.wav"
)


def test_tokenize_splits_camel_case_and_letter_digit_boundaries():
    assert tokenize_component("Kick01.wav", is_filename=True) == ["kick", "01"]
    assert tokenize_component("120BPM.wav", is_filename=True) == ["120", "bpm"]
    assert tokenize_component("perc80") == ["perc", "80"]
    assert tokenize_component("Piano_&_Keys") == ["piano", "keys"]
    assert tokenize_component("Drum___Perc_One_Shots") == ["drum", "perc", "one", "shots"]
    assert tokenize_component("F#m.wav", is_filename=True) == ["f#m"]
    assert tokenize_component("one-shots") == ["one", "shots"]


def test_pack_is_first_path_component():
    layers = split_rel_path(CAPSUN_PIANO)
    assert layers.pack == "Capsun - Lofi Love Songs"
    assert layers.ext == "wav"
    assert "pianos" in layers.filename
    # The CPA_... folder is a pack-title echo, not an instrument ancestor.
    assert all("capsun" not in layer for layer in layers.ancestors)


def test_capsun_tape_pianos_loop():
    meta = extract_metadata(CAPSUN_PIANO)
    assert meta.pack == "Capsun - Lofi Love Songs"
    assert meta.leaf == "piano"
    assert meta.family == "keys"
    assert meta.type == "loop"
    assert meta.bpm == 120
    assert meta.key_root == "F"
    assert meta.key_mode == "minor"
    assert meta.key == "F minor"
    assert meta.instrument_conf == "high"
    assert meta.instrument_source == "filename"
    assert "pianos" in meta.matched_tokens
    assert meta.conflict is False


def test_bedroom_snare_one_shot():
    meta = extract_metadata(BEDROOM_SNARE)
    assert meta.leaf == "snare"
    assert meta.family == "drums"
    assert meta.type == "one_shot"
    assert meta.bpm is None
    assert meta.key is None
    assert meta.instrument_conf == "high"
    assert meta.instrument_source == "filename"
    assert "snare" in meta.matched_tokens
    assert meta.type_source in {"filename", "parent"}


def test_taiko_melodic_loop_reverse_is_not_fx():
    meta = extract_metadata(TAIKO_REVERSE)
    assert meta.leaf == "unknown"
    assert meta.family == "unknown"
    assert meta.type == "loop"
    assert meta.bpm == 87
    assert meta.key_root == "C#"  # Db → C#
    assert meta.key_mode == "minor"
    assert meta.instrument_conf == "none"
    # "reverse" must not become the instrument; "taiko" is only a pack prefix.
    assert meta.family != "fx"
    assert "reverse" not in (meta.matched_tokens or [])


def test_bass_drum_is_kick_not_bass():
    meta = extract_metadata("Pack/Kicks/Kick_Bass_Drum_01.wav")
    assert meta.family == "drums"
    assert meta.leaf == "kick"


def test_808_kick_and_808_bass():
    kick = extract_metadata("Pack/Kicks/808_kick_hard.wav")
    assert kick.leaf == "kick"
    assert kick.family == "drums"

    bass = extract_metadata("Pack/Bass/808_bass_C.wav")
    assert bass.leaf == "sub_808"
    assert bass.family == "bass"

    keyed = extract_metadata("Pack/Bass/808_C.wav")
    assert keyed.leaf == "sub_808"
    assert keyed.family == "bass"


def test_808_under_kicks_folder_without_keyword():
    meta = extract_metadata("Pack/Drums/Kicks/808_hard.wav")
    assert meta.leaf == "kick"
    assert meta.family == "drums"


def test_tr808_is_not_an_instrument():
    meta = extract_metadata("Pack/Loops/TR-808_demo_loop.wav")
    assert meta.leaf != "sub_808"
    assert meta.family != "bass" or meta.leaf != "sub_808"


def test_sub_rules():
    bass = extract_metadata("Pack/Bass/sub_bass_dark.wav")
    assert bass.leaf == "sub_808"
    assert bass.family == "bass"

    kick = extract_metadata("Pack/Kicks/sub_kick_boom.wav")
    assert kick.leaf == "kick"

    lone = extract_metadata("Pack/One_Shots/sub_rumble.wav")
    assert lone.family == "bass"
    assert lone.leaf == "sub_808"
    assert lone.instrument_conf == "low"


def test_pad_and_drum_pad():
    perc = extract_metadata("Pack/Perc/drum_pad_hit.wav")
    assert perc.family == "drums"
    assert perc.leaf == "perc"

    pad = extract_metadata("Pack/Pads/warm_pad.wav")
    assert pad.family == "synth"
    assert pad.leaf == "pad"
    assert pad.instrument_source in {"filename", "parent"}


def test_hit_alone_is_fx_one_shot_but_brass_hit_keeps_instrument():
    lone = extract_metadata("Pack/FX/orch_nope_hit.wav")
    # "hit" alone (no other instrument word besides maybe leftover) → fx + one_shot
    # This filename has no brass/orch keyword spelled as "orch hit"; use a cleaner case:
    lone = extract_metadata("Pack/FX/hit_01.wav")
    assert lone.family == "fx"
    assert lone.leaf == "fx"
    assert lone.type == "one_shot"

    brass = extract_metadata("Pack/Brass/brass_hit_stab.wav")
    assert brass.family == "brass"
    assert brass.leaf == "brass"
    assert brass.type == "one_shot"


def test_vocal_lead_is_vocal_not_synth_lead():
    meta = extract_metadata(
        "Afro Dimensions/One_Shots/Tonal/Vocals/SO_AF_92_vocal_lead_casamance_Amin.wav"
    )
    assert meta.family == "vocal"
    assert meta.leaf == "vocal"
    assert meta.key_root == "A"
    assert meta.key_mode == "minor"


def test_guitar_lead_stays_guitar():
    meta = extract_metadata(
        "Electric Soul - Guitar Loops and Riffs by Treehouz Vol 3/"
        "SC_ES/Lead_Loops/SC_ES_126_guitar_lead_thoughts_Amin.wav"
    )
    assert meta.family == "guitar"
    assert meta.leaf == "electric"
    assert meta.type == "loop"
    assert meta.bpm == 126
    assert meta.key == "A minor"


def test_short_abbrev_needs_folder_context():
    # "sn" in a snares folder is accepted (medium).
    in_context = extract_metadata("Pack/Drums/Snares/sn_01.wav")
    assert in_context.leaf == "snare"
    assert in_context.instrument_conf in {"medium", "high"}

    # "sn" with no drums folder is not a snare.
    bare = extract_metadata("Pack/Melodic/sn_melody_C.wav")
    assert bare.leaf != "snare"


def test_no_kick_loops_are_not_kick():
    meta = extract_metadata(
        "Capsun - Lofi Love Songs/CPA_Capsun_Lofi_Love_Songs_Sample_Pack/"
        "Drum_&_Perc_Loops/No_Kick_Loops/CAP_90_no_kick_loop_dusty.wav"
    )
    assert meta.leaf != "kick"
    assert meta.type == "loop"
    assert meta.family == "drums"
    assert meta.leaf == "full"


def test_ambient_pack_prefix_am_is_not_key_a_minor():
    meta = extract_metadata(
        "Ambient & Chill/Sample_Magic_-_Ambient_and_Chill_-_Wav/"
        "drum_hits/kick/am_kick_morrow.wav"
    )
    assert meta.leaf == "kick"
    assert meta.key is None
    assert meta.type == "one_shot"  # drum_hits


def test_bpm_from_glued_token_and_folder():
    glued = extract_metadata(
        "Ambient & Chill/Sample_Magic_-_Ambient_and_Chill_-_Wav/"
        "loops/80bpm/am_drm80_brooklinn_ful.wav"
    )
    assert glued.bpm == 80
    assert glued.family == "drums"
    assert glued.type == "loop"

    folder = extract_metadata("Pack/Loops/120/loop_warm.wav")
    assert folder.bpm == 120
    assert folder.type == "loop"


def test_bare_letter_key_only_at_filename_end():
    end = extract_metadata("Pack/Guitar/SO_AF_guitar_note_clean_A.wav")
    assert end.key_root == "A"
    assert end.key_mode == "unknown"
    assert end.family == "guitar"

    mid = extract_metadata("Pack/Loops/a_melody_loop.wav")
    assert mid.key is None


def test_os_pack_prefix_is_not_one_shot():
    meta = extract_metadata(
        "More Trap Samples/Origin_Sound_-_More_Trap_Samples/"
        "Drum_Loops/Full/OS_MTS_150_Drum_Loop_Dime_Bag.wav"
    )
    assert meta.type == "loop"
    assert meta.bpm == 150
    assert meta.family == "drums"
    assert meta.leaf == "full"


def test_fx_category_in_filename_beats_kick():
    meta = extract_metadata("Big EDM Stems/BEDMS1/002_BEDMS_-_EDM_Kit_D#_128bpm/002_-_10_FX_Reversed_Kick_-_BEDMS_Zenhiser.wav")
    assert meta.family == "fx"
    assert meta.leaf == "fx"
    assert meta.bpm == 128
    # Kit folder has D# and 128bpm
    assert meta.key_root == "D#"


def test_closed_hat_and_chh():
    hat = extract_metadata(
        "Dusty Beats & Chilled Hip-Hop/SKU/one_shots/hats/"
        "dch_hat_closed_one_shot_machine_808.wav"
    )
    assert hat.leaf == "hihat"
    assert hat.family == "drums"
    assert hat.type == "one_shot"
    # trailing 808 must not override the hat
    assert hat.leaf != "sub_808"

    chh = extract_metadata("Afro Dimensions/One_Shots/Hats/SO_AF_chh_iss.wav")
    assert chh.leaf == "hihat"
    assert chh.type == "one_shot"


def test_file_id_is_stable_sha1_of_path():
    a = extract_metadata(CAPSUN_PIANO)
    b = extract_metadata(CAPSUN_PIANO)
    assert a.file_id == b.file_id
    assert len(a.file_id) == 40
    other = extract_metadata(BEDROOM_SNARE)
    assert other.file_id != a.file_id


def test_splice_double_prefix_bpm():
    """SO_LI_95_... / SC_ES_90_... put a pack code between the prefix and the BPM."""
    piano = extract_metadata(
        "La Isla/Loops/Piano/SO_LI_95_piano_bodega_lofi_tape_crackling_Emin.wav"
    )
    assert piano.bpm == 95
    assert piano.leaf == "piano"
    assert piano.type == "loop"
    assert piano.key == "E minor"

    guitar = extract_metadata(
        "Electric Soul - Guitar Loops and Riffs by Treehouz Vol 3/"
        "SC_ES/Picked_Loops/SC_ES_90_guitar_chords_picked_Amin.wav"
    )
    assert guitar.bpm == 90
    assert guitar.family == "guitar"


def test_orchestral_taiko_is_not_strings():
    meta = extract_metadata(
        "Sounds of KSHMR Vol. 4/KSHMR_drums/KSHMR_orchestral_drums/"
        "KSHMR_orchestral_drum_taiko_hard.wav"
    )
    assert meta.family in {"drums", "world"}
    assert meta.family != "strings"
