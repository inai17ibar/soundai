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


# ---------------------------------------------------------------------------
# v0.2.0 dictionary and rule updates
# ---------------------------------------------------------------------------


def test_whistle_is_woodwind():
    inst = extract_metadata(
        "Sounds of KSHMR Vol. 4/KSHMR_tonal/KSHMR_instruments/"
        "KSHMR_104_whistle_loop_syncopated_Cmin.wav"
    )
    assert inst.family == "woodwind"
    assert inst.leaf == "woodwind"
    assert inst.type == "loop"
    assert inst.conflict is False


def test_songstarter_with_whistle_stays_unknown():
    meta = extract_metadata(
        "Sounds of KSHMR Vol. 4/KSHMR_tonal/KSHMR_songstarters/"
        "KSHMR_128_songstarter_loop_whistle_wit_it_Emin.wav"
    )
    assert meta.family == "unknown"
    assert meta.leaf == "unknown"
    assert meta.instrument_conf == "none"
    assert meta.type == "loop"


def test_hoover_is_synth_lead_not_drum_kit():
    meta = extract_metadata(
        "Big EDM Stems/BEDMS1/007_BEDMS_-_EDM_Kit_F_128bpm/"
        "007_-_09_Hoover_-__BEDMS_Zenhiser.wav",
        size_bytes=22_852_288,
    )
    assert meta.family == "synth"
    assert meta.leaf == "lead"
    assert meta.conflict is False
    assert meta.type == "loop"
    assert meta.bpm == 128
    assert meta.type_conf in {"medium", "low"}


def test_drum_fill_is_its_own_leaf():
    fill = extract_metadata(
        "TAIKO Essentials Sample Pack/TAIKO_drums/TAIKO_drum_fills/"
        "TAIKO_95_drum_fill_fresco.wav"
    )
    assert fill.family == "drums"
    assert fill.leaf == "fill"
    assert fill.type == "loop"
    assert fill.type_conf in {"medium", "high"}

    # A snare fill is still a fill, not drums/snare or drums/full.
    snare_fill = extract_metadata(
        "Big EDM Stems/BEDMS1/001_BEDMS_-_EDM_Kit_G_128bpm/"
        "001_-_23_Snare_Fill_-_BEDMS_Zenhiser.wav",
        size_bytes=23_845_830,
    )
    assert snare_fill.leaf == "fill"
    assert snare_fill.family == "drums"


def test_bass_loop_fill_stays_bass():
    meta = extract_metadata(
        "Bedroom Pianos & Electronics/Bedroom_Pianos_&_Electronics_-_WAV/"
        "loops/bass_loops/BPE_120_bass_loop_fill_Amin.wav"
    )
    assert meta.family == "bass"
    assert meta.leaf == "synth_bass"
    assert meta.leaf != "fill"


def test_big_edm_kit_stems_are_loops():
    kick = extract_metadata(
        "Big EDM Stems/BEDMS1/007_BEDMS_-_EDM_Kit_F_128bpm/"
        "007_-_10_Kick_-__BEDMS_Zenhiser.wav",
        size_bytes=22_852_288,
    )
    assert kick.family == "drums"
    assert kick.leaf == "kick"
    assert kick.type == "loop"
    assert kick.bpm == 128
    assert kick.conflict is False
    assert "kit_stem" in " ".join(kick.matched_rules)

    # "Snare_Hit" in a kit stem folder is a loop of that part, not a one-shot.
    hit = extract_metadata(
        "Big EDM Stems/BEDMS1/001_BEDMS_-_EDM_Kit_G_128bpm/"
        "001_-_24_Snare_Hit_-_BEDMS_Zenhiser.wav",
        size_bytes=23_845_830,
    )
    assert hit.leaf == "snare"
    assert hit.type == "loop"
    assert hit.conflict is False


def test_kit_stem_without_size_is_still_loop_at_low_confidence():
    meta = extract_metadata(
        "Big EDM Stems/BEDMS1/003_BEDMS_-_EDM_Kit_G_128bpm/"
        "003_-_01_Bass_-_BEDMS_Zenhiser.wav"
    )
    assert meta.type == "loop"
    assert meta.type_conf == "low"
    assert meta.family == "bass"


def test_synth_chord_lead_pad_from_filename():
    chord = extract_metadata(
        "La Isla/One_Shots/Tonal/Synth/SO_LI_synth_sint_ti_chord_Fmin.wav"
    )
    assert chord.family == "synth"
    assert chord.leaf == "chord"
    assert chord.instrument_conf == "high"
    assert chord.type == "one_shot"

    lead = extract_metadata(
        "La Isla/One_Shots/Tonal/Synth/SO_LI_synth_note_soft_lead_D.wav"
    )
    assert lead.family == "synth"
    assert lead.leaf == "lead"
    assert lead.instrument_conf == "high"

    pad = extract_metadata(
        "La Isla/Loops/Synth/SO_LI_95_synth_playita_pads_dramatic_Bbmin.wav"
    )
    assert pad.family == "synth"
    assert pad.leaf == "pad"
    assert pad.type == "loop"


def test_synth_swells_is_pad_at_medium_or_low_confidence():
    meta = extract_metadata(
        "Sounds of KSHMR Vol. 4/KSHMR_tonal/KSHMR_synths/"
        "KSHMR_85_synth_lofi_loop_swells_Cmaj.wav"
    )
    assert meta.family == "synth"
    assert meta.leaf == "pad"
    assert meta.instrument_conf in {"medium", "low"}
    assert meta.instrument_conf != "high"
    assert meta.type == "loop"


def test_synth_plus_bass_is_synth_bass():
    meta = extract_metadata(
        "Sounds of KSHMR Vol. 4/KSHMR_tonal/KSHMR_synths/"
        "KSHMR_89_synth_loop_bass_alarm_Gmin.wav"
    )
    assert meta.family == "bass"
    assert meta.leaf == "synth_bass"
    assert meta.conflict is False
    assert meta.type == "loop"


def test_synth_brass_in_synth_folder_is_synth_not_conflict():
    meta = extract_metadata(
        "Afro Dimensions/Loops/Tonal/Synth/SO_AF_100_synth_brass_riff_casamance_Emin.wav"
    )
    assert meta.family == "synth"
    assert meta.conflict is False


def test_plucked_is_pluck():
    meta = extract_metadata("Pack/Synths/warm_plucked_lead_C.wav")
    # "lead" and "plucked" both synth; either pluck or lead is acceptable as long
    # as we don't leave it as synth/unknown.  "plucked" is a named leaf.
    assert meta.family == "synth"
    assert meta.leaf in {"pluck", "lead"}


def test_mallet_token_is_mallet():
    meta = extract_metadata(
        "Bedroom Beats & Lofi Hip-Hop/CPA_Bedroom_Beats___LoFi_HipHop/"
        "Melodic_One_Shots/Lofi_Melodics/BBL_lofi_mallet_bimba_Am.wav"
    )
    assert meta.family == "mallet"
    assert meta.leaf == "mallet"
    assert meta.type == "one_shot"


def test_vocal_and_fx_default_to_one_shot():
    vox = extract_metadata(
        "TAIKO Essentials Sample Pack/TAIKO_vocals/TAIKO_vocal_otaku.wav"
    )
    assert vox.family == "vocal"
    assert vox.type == "one_shot"
    assert vox.type_conf == "low"

    fx = extract_metadata(
        "MURDA BEATZ PRESENTS BANDEMIC DRUM KIT/MURDA_fx/MURDA_fx_serpent_riser.wav"
    )
    assert fx.family == "fx"
    # "riser" is a type-neutral FX keyword; default still one_shot unless loop
    assert fx.type == "one_shot"


def test_fx_loop_keyword_still_wins_over_vocal_fx_default():
    meta = extract_metadata(
        "Sounds of KSHMR Vol. 4/KSHMR_fx/KSHMR_fx_loops/"
        "KSHMR_105_fx_loop_scratch_vocal_stutter.wav"
    )
    assert meta.family == "fx"
    assert meta.type == "loop"


def test_filename_instrument_vs_kit_folder_is_not_a_conflict():
    lead = extract_metadata(
        "Big EDM Stems/BEDMS1/001_BEDMS_-_EDM_Kit_G_128bpm/"
        "001_-_13_Lead_-_BEDMS_Zenhiser.wav",
        size_bytes=23_845_830,
    )
    assert lead.family == "synth"
    assert lead.leaf == "lead"
    assert lead.conflict is False
    assert lead.type == "loop"

    kit_loop = extract_metadata(
        "Bedroom Pianos & Electronics/Bedroom_Pianos_&_Electronics_-_WAV/"
        "loops/music_loops/BPE_120_kit_loop_heavy_Amin/"
        "BPE_120_kit_loop_heavy_piano_Amin.wav"
    )
    assert kit_loop.family == "keys"
    assert kit_loop.leaf == "piano"
    assert kit_loop.conflict is False
    assert kit_loop.type == "loop"


def test_vocal_riff_is_vocal_not_guitar():
    meta = extract_metadata(
        "Afro Dimensions/Loops/Tonal/Vocals/SO_AF_108_vocal_riff_thies_Cmin.wav"
    )
    assert meta.family == "vocal"
    assert meta.leaf == "vocal"
    assert meta.conflict is False


def test_ukulele_plucks_are_world_not_synth_pluck():
    meta = extract_metadata(
        "Capsun - Lofi Love Songs/CPA_Capsun_Lofi_Love_Songs_Sample_Pack/"
        "Melodic_Loops/Guitar_&_Plucked/CAP_150_ukulele_plucks_the_next_morning_A#m.wav"
    )
    assert meta.family == "world"
    assert meta.leaf == "world"
    assert meta.conflict is False


def test_bass_synth_ride_is_bass_not_cymbal():
    meta = extract_metadata(
        "Bedroom Pianos & Electronics/Bedroom_Pianos_&_Electronics_-_WAV/"
        "one-shots/bass_one-shots/BPE_bass_synth_ride_F.wav"
    )
    assert meta.family == "bass"
    assert meta.leaf == "synth_bass"
    assert meta.type == "one_shot"
    assert meta.conflict is False


def test_sax_in_horns_folder_stays_conflict():
    """Pack taxonomy (Horns/Brass) vs sax-as-woodwind stays flagged for review."""
    meta = extract_metadata(
        "La Isla/Loops/Horns/SO_LI_95_saxophone_bodega_tenor_Emin.wav"
    )
    assert meta.family == "woodwind"
    assert meta.leaf == "woodwind"
    assert meta.conflict is True
    assert meta.type == "loop"


def test_lifter_is_fx():
    meta = extract_metadata(
        "Big EDM Stems/BEDMS1/002_BEDMS_-_EDM_Kit_D#_128bpm/"
        "002_-_09_FX_Lifters_2_-_BEDMS_Zenhiser.wav",
        size_bytes=22_852_288,
    )
    assert meta.family == "fx"
    assert meta.leaf == "fx"
    assert meta.type == "loop"
    assert meta.conflict is False


def test_guitar_plucks_stay_guitar():
    meta = extract_metadata(
        "La Isla/One_Shots/Tonal/Guitar/SO_LI_guitar_playita_note_plucks_electric_tele_Bb.wav"
    )
    assert meta.family == "guitar"
    assert meta.leaf == "electric"
    assert meta.conflict is False
    assert meta.type == "one_shot"


def test_brass_stab_is_brass_not_synth_chord():
    meta = extract_metadata("La Isla/One_Shots/Tonal/Brass/SO_LI_brass_note_stab_C.wav")
    assert meta.family == "brass"
    assert meta.leaf == "brass"
    assert meta.conflict is False
    assert meta.type == "one_shot"


def test_rnb_vibes_is_not_vibraphone():
    meta = extract_metadata(
        "Sounds of KSHMR Vol. 4/KSHMR_vocals/KSHMR_hooks_and_chants/"
        "KSHMR_90_vocal_loop_freestyle_melody_rnb_vibes_Dmaj.wav"
    )
    assert meta.family == "vocal"
    assert meta.leaf == "vocal"
    assert meta.conflict is False
    assert meta.type == "loop"


def test_incidental_crackle_does_not_make_fx():
    meta = extract_metadata(
        "Capsun - Lofi Love Songs/CPA_Capsun_Lofi_Love_Songs_Sample_Pack/"
        "Drum_&_Perc_Loops/Drum_Loops/CAP_85_drum_loop_popcorn_crackle.wav"
    )
    assert meta.family == "drums"
    assert meta.leaf == "full"
    assert meta.conflict is False
    assert meta.type == "loop"


def test_bass_synth_brass_and_impact_stay_bass():
    brass = extract_metadata(
        "Bedroom Pianos & Electronics/Bedroom_Pianos_&_Electronics_-_WAV/"
        "one-shots/bass_one-shots/BPE_bass_synth_brass_A.wav"
    )
    assert brass.family == "bass"
    assert brass.leaf == "synth_bass"
    assert brass.conflict is False

    impact = extract_metadata(
        "Bedroom Pianos & Electronics/Bedroom_Pianos_&_Electronics_-_WAV/"
        "one-shots/bass_one-shots/BPE_bass_synth_impact_D.wav"
    )
    assert impact.family == "bass"
    assert impact.leaf == "synth_bass"
    assert impact.conflict is False


def test_revcym_is_cymbal():
    meta = extract_metadata(
        "Big EDM Stems/BEDMS1/008_BEDMS_-_EDM_Kit_F_128bpm/"
        "008_-_19_Revcym_-_BEDMS_Zenhiser.wav",
        size_bytes=26_826_456,
    )
    assert meta.family == "drums"
    assert meta.leaf == "cymbal"
    assert meta.type == "loop"
    assert meta.conflict is False
