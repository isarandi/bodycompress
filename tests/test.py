import os

import numpy as np

import bodycompress


def test():
    vertices = np.random.randn(10, 2, 6890, 3).astype(np.float32) * 1000
    joints = np.random.randn(10, 2, 24, 3).astype(np.float32) * 1000

    with bodycompress.BodyCompressor(
            '/tmp/test.xz', dict(some='metadata'), quantization_mm=0.5) as compr:
        for v, j in zip(vertices, joints):
            compr.append(vertices=v, joints=j)

    size = os.path.getsize('/tmp/test.xz')
    uncompressed_size = vertices.nbytes + joints.nbytes
    assert size < uncompressed_size

    decompr = bodycompress.BodyDecompressor('/tmp/test.xz')
    assert decompr.metadata == dict(some='metadata')

    for v, j, frame in zip(vertices, joints, decompr):
        assert np.allclose(frame['vertices'], v, atol=1)
        assert np.allclose(frame['joints'], j, atol=1)
