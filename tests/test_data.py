from src.data.load_data import load_raw


def test_load_raw_shape_and_target():
    df = load_raw()

    # We expect exactly 1000 rows and 21 columns — if this fails,
    # either the wrong file got downloaded, or it's been edited/corrupted.
    assert df.shape == (1000, 21)

    # Target should only ever contain 0 and 1 after our remap.
    assert set(df["target"].unique()) == {0, 1}

    # The dataset's documented default rate is ~30%. abs=0.01 allows
    # a small tolerance rather than demanding an exact float match.
    assert df["target"].mean() == 0.3