import hil_eval as hil

from soundai.hil_review import PROJECT, build_stub_searcher, make_reviewed_search


def test_text_search_queues_audio_assets(tmp_path):
    hil.init(project=PROJECT, store=f"sqlite:///{tmp_path}/hil.db")
    search = make_reviewed_search(build_stub_searcher(tmp_path / "s"))
    res = search("punchy kick drum", k=3)
    assert res[0]["name"].startswith("kick")
    (item,) = hil.list_items()
    assert item.input["query"] == "punchy kick drum"
    assert all("$asset" in r["audio"] for r in item.output)
    assert len(item.assets) == 3
    assert all(a.mime == "audio/wav" for a in item.assets)


def test_text_search_assets_follow_rank(tmp_path):
    hil.init(project=PROJECT, store=f"sqlite:///{tmp_path / 'hil.db'}")
    search = make_reviewed_search(build_stub_searcher(tmp_path / "samples"))
    results = search("kick", k=3)
    assert all(isinstance(r["audio"], str) for r in results)
    [item] = hil.list_items(project=PROJECT)
    assert item.fn_name == "search_text"
    ordered = sorted(item.assets, key=lambda a: a.order)
    assert [a.sha256 for a in ordered] == [r["audio"]["$asset"] for r in item.output]
    assert [a.order for a in ordered] == list(range(len(item.output)))
