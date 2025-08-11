
def test_import():
    import analysis_pipeline
    assert isinstance(analysis_pipeline.__version__, str)
