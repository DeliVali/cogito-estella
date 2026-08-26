from cogito_estella import multilingual_factory as mf


def test_mix_sums_to_100():
    assert sum(s.pct for s in mf.sources()) == 100


def test_doc_targets_partition():
    t = mf.doc_targets(3_000_000)
    assert t["en"] == 660_000
    assert t["toolcalls_syn"] == 120_000


def test_sonar_lang_tags_valid_and_chinese_mapped():
    for s in mf.sources():
        assert "_" in s.sonar_lang and len(s.sonar_lang) == 8
    # chino: carga cmn_Hani pero encodea con zho_Hans (el tag válido de SONAR)
    zh = next(s for s in mf.sources() if s.name == "zh")
    assert zh.hf_config == "cmn_Hani" and zh.sonar_lang == "zho_Hans"


def test_modalities_present():
    mods = {s.modality for s in mf.sources()}
    assert mods == {"prose", "code", "toolcall"}


def test_synthetic_toolcalls_normalized():
    recs = list(mf.iter_records(
        mf.SourceSpec("t", "__synthetic__", None, None, "eng_Latn", "toolcall", 4), 5, 0))
    assert len(recs) == 5
    import json
    for r in recs:
        assert r.modality == "toolcall" and r.sonar_lang == "eng_Latn"
        json.loads(r.text)


def test_segment_toolcall_is_single_unit():
    rec = mf.DocRecord('{"name": "search", "arguments": {"q": "x"}}', "eng_Latn", "syn", "toolcall")
    # un tool-call corto es UNA unidad estructurada (no se parte en oraciones)
    units = mf.segment_record(rec, segmenter=None)
    assert units == [rec.text]
