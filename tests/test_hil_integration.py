import asyncio
import selectors
import math
import wave

import hil_eval as hil
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hil_eval.server.app import create_app
from hil_eval.store.sqlite import SQLiteStore

from soundai.hil import (PROJECT, HilClient, esc50_confidence, judge_result,
                         precision_metrics, reviewed_search, simulate_reviews)
from soundai.index import EmbeddingIndex
from soundai.search import SampleSearcher


def test_loop():
    # Bound selector waits: this sandbox can lose cross-thread wakeups.
    class PollingSelector(selectors.DefaultSelector):
        def select(self, timeout=None):
            return super().select(.01 if timeout is None else min(timeout, .01))
    return asyncio.SelectorEventLoop(PollingSelector())


test_loop.__test__ = False


class FakeEmbedder:
    def embed_texts(self, texts):
        return np.tile([1., 0.], (len(texts), 1)).astype(np.float32)

    def embed_files(self, paths):
        return np.array([[.6, math.sqrt(.64)], [.1, math.sqrt(.99)]], dtype=np.float32)[:len(paths)]


def test_confidence_and_judge():
    assert esc50_confidence(.40, 0) == .5
    assert esc50_confidence(.6, .1) > esc50_confidence(.6, 0) > esc50_confidence(.45, 0)
    assert esc50_confidence(.2, 0) == pytest.approx(esc50_confidence(.6, 0))
    assert 0.5 <= esc50_confidence(-10, 0) <= 1
    j = judge_result({'score': .6, 'margin': .1}, {'proposed': 'relevant'})
    assert j.label == 'relevant'
    assert j.confidence == pytest.approx(esc50_confidence(.6, .1))


def test_decorator_audio_simulator_and_metrics(tmp_path):
    db = tmp_path / 'hil.db'
    hil.init(project=PROJECT, store=f'sqlite:///{db}')
    paths = [tmp_path / 'dog.wav', tmp_path / 'cat.wav']
    for path in paths:
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(8000)
            f.writeframes(b'\x00\x00' * 8)
    embedder = FakeEmbedder()
    searcher = SampleSearcher(embedder=embedder)
    searcher.index = EmbeddingIndex(dim=2)
    searcher.index.add(embedder.embed_files(paths), list(map(str, paths)))
    output = reviewed_search(searcher, 'dog barking', k=2, threshold=1.0, audit_rate=0)
    items = hil.list_items()
    assert len(items) == 2
    assert output == [{'proposed': 'relevant'}, {'proposed': 'not_relevant'}]
    with TestClient(create_app(SQLiteStore(db)), backend_options={"loop_factory": test_loop}) as http:
        for item in items:
            assert item.options == ['relevant', 'not_relevant']
            assert item.input['audio'] == {'$asset': item.assets[0].sha256}
            assert 'score' not in item.input
            assert item.assets[0].mime == 'audio/wav'
            assert http.get('/v1/blobs/' + item.assets[0].sha256).content == paths[0].read_bytes()
        client = HilClient(http)
        assert simulate_reviews(items, {'dog.wav': 'dog', 'cat.wav': 'cat'},
                                {'dog barking': 'dog'}, client) == 2
        assert simulate_reviews(hil.list_items(), {'dog.wav': 'dog', 'cat.wav': 'cat'},
                                {'dog barking': 'dog'}, client) == 0
    records = list(hil.export(fmt='eval-set', include_images=False))
    report = precision_metrics(hil.list_items(), records, 2)
    assert report['macro'] == {'judge_precision': .5, 'human_precision': .5}
    assert report['queries'][0]['complete']
    partial = precision_metrics(hil.list_items(), records[:1], 2)
    assert partial['queries'][0]['coverage'] == .5
    assert not partial['queries'][0]['complete']
    with pytest.raises(ValueError, match='Duplicate'):
        precision_metrics(hil.list_items() * 2, records, 2)
    # Routing belongs to the decorator; simulator never gates a second time.
    reviewed_search(searcher, 'high confidence', k=2, threshold=0, audit_rate=0)
    assert len(hil.list_items()) == 2
    reviewed_search(searcher, 'audited', k=2, threshold=0, audit_rate=1)
    assert len(hil.list_items()) == 4
