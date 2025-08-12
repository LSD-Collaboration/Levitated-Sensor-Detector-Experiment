import analysis_pipeline as ap


def test_version():
    assert isinstance(ap.__version__, str)
