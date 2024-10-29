# BodyCompress

This library compresses and serializes the output of nonparametric 3D human mesh estimators to disk.

Without compression, a sequence of 3D human meshes extracted from a video can take up huge amounts
of disk space, as
we need to store the coordinates for thousands of vertices in every frame. At 30 fps and 6890
vertices (like SMPL), this amounts to almost 9 GB/person/hour. If you want to save the estimation
result for a multi-person video, it will be proportionally more.

This library achieves a **compression ratio of over 8x** on temporal human mesh data, with minimal
loss in
information. It consists of the following steps:

1. **Quantization**: The floating-point coordinates of the vertices are quantized at 0.5 mm
   resolution.
2. **Differential Encoding**: The quantized coordinates are differentially encoded in the order of
   the vertices as listed in the template. In SMPL, this order is not random but has locality, so
   the
   differences between adjacent vertices tend to be small.
3. **Serialization**: `msgpack-numpy` is used to serialize the NumPy arrays to a byte stream.
3. **LZMA Compression**: The serialized byte stream is compressed using the lossless
   Lempel–Ziv–Markov chain algorithm with the `xz` binary tool. This can run in multi-threaded
   parallel mode and is reasonably fast at compression level 5. (The Python `lzma` module does not
   have multi-threading support and is too slow for our use case.)

The format supports storing additional metadata in the header, and several per-frame pieces of
information, such as vertices, joints, uncertainties, and camera parameters, compressing it all into
one sequentially readable file.

## Installation

```bash
pip install git+https://github.com/isarandi/bodycompress.git
```

## Usage

Use the `BodyCompressor` and `BodyDecompressor` classes to compress and decompress the data.
Both should be used as context managers. The compressor has an `append` method which should be
called with keyword arguments. The decompressor is an iterator over dictionaries with the same keys.

Note that seeking is not supported, the stream is compressed as a whole to achieve the best
compression ratio.

### Compression

```python
from bodycompress import BodyCompressor

with BodyCompressor('out.xz', metadata={'whatever': 'you want'}) as bcompr:
    for frame in frames:
        vertices, joints = estimate(frame)
        bcompr.append(vertices=vertices, joints=joints)
```

Any keyword arguments can be passed to `append` that are nested dicts/lists/tuples of primitive
types or NumPy arrays.
However the following keywords are handled specially:

* `vertices`: a `(..., num_verts, 3)` NumPy array of vertex coordinates (in millimeters)
* `joints`: a `(..., num_joints, 3)` NumPy array of joint coordinates (in millimeters)
* `vertex_uncertainties`: a `(..., num_verts)` NumPy array of vertex uncertainties (in meters)'
* `joint_uncertainties`: a `(..., num_verts)` NumPy array of joint uncertainties (in meters)'
* `camera`: a `cameralib.Camera` object

### Decompression

```python
from bodycompress import BodyDecompressor

with BodyDecompressor('out.xz') as bdecompr:
    print(bdecompr.metadata)  # {'whatever': 'you want'}
    for data in bdecompr:
        render(data['vertices'], data['joints'])
```
