def test_sherpa_native_runtime_loads():
    import sherpa_onnx

    assert sherpa_onnx.OfflineSpeakerDiarizationConfig is not None
