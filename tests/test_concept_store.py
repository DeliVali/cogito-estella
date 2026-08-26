import numpy as np

from cogito_estella.concept_store import ConceptDataset, ShardWriter


def test_write_and_read_roundtrip(tmp_path):
    w = ShardWriter(str(tmp_path), shard_size=1000)
    for i in range(5):
        w.add(np.full(1024, float(i), dtype=np.float16), f"texto {i}", "spa_Latn", "src", f"doc{i}")
    w.close()

    ds = ConceptDataset(str(tmp_path))
    assert len(ds) == 5
    emb, meta = ds[2]
    assert emb.shape == (1024,)
    assert np.allclose(emb, 2.0)
    assert meta["text"] == "texto 2"
    assert meta["lang"] == "spa_Latn"
    assert meta["doc_id"] == "doc2"


def test_shard_rotation(tmp_path):
    w = ShardWriter(str(tmp_path), shard_size=2)
    for i in range(5):
        w.add(np.zeros(1024, dtype=np.float16), f"t{i}", "eng_Latn", "s", f"d{i}")
    w.close()

    # 5 items, shard_size 2 -> 3 shards (2, 2, 1)
    shards = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert len(shards) == 3

    ds = ConceptDataset(str(tmp_path))
    assert len(ds) == 5
    texts = [ds[i][1]["text"] for i in range(5)]
    assert texts == ["t0", "t1", "t2", "t3", "t4"]


def test_empty_close_no_crash(tmp_path):
    w = ShardWriter(str(tmp_path), shard_size=10)
    w.close()
    ds = ConceptDataset(str(tmp_path))
    assert len(ds) == 0


def test_dtype_is_float16(tmp_path):
    w = ShardWriter(str(tmp_path), shard_size=10)
    w.add(np.ones(1024, dtype=np.float32), "x", "eng_Latn", "s", "d0")
    w.close()
    ds = ConceptDataset(str(tmp_path))
    emb, _ = ds[0]
    assert emb.dtype == np.float16
