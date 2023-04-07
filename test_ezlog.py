import os 

def test_timestamp():
    from ezlog import timestamp
    assert isinstance(timestamp, str)

def test_directory():
    from ezlog import directory
    with directory("tmp"):
        os.rmdir("../tmp")

def test_npimg():
    from ezlog import npimg
    import torch
    import numpy as np 
    x = npimg(torch.randn(1, 3, 256, 256))
    assert x.shape == (1, 256, 256, 3)
    assert x.dtype == np.uint8
    assert x.min() == 0
    assert x.max() == 255
