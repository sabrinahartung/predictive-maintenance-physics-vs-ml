from pdm.data import COLUMN_MAP, FAILURE_MODES, FEATURES, TARGET, load_data


def test_load_data_schema():
    df = load_data()
    assert df.shape == (10_000, len(COLUMN_MAP))
    assert list(df.columns) == list(COLUMN_MAP.values())
    assert set(FEATURES + [TARGET, *FAILURE_MODES]) <= set(df.columns)


def test_load_data_types():
    df = load_data()
    assert df["type"].cat.ordered
    assert list(df["type"].cat.categories) == ["L", "M", "H"]
    assert df[TARGET].isin([0, 1]).all()
    assert df.isna().sum().sum() == 0
