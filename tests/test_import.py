def test_import() -> None:
    import lnPi

    import lnpy

    assert lnPi.__version__ is not None
    assert lnpy.__version__ is not None
