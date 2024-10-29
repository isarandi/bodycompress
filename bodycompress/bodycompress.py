import os
import os.path as osp
import subprocess

import cameralib
import msgpack_numpy
import numpy as np
from scipy.spatial.transform import Rotation as R


class BodyCompressor:
    def __init__(self, path, metadata=None, quantization_mm=0.5, n_threads=0):
        os.makedirs(osp.dirname(path), exist_ok=True)
        self.f = open(path, 'wb')
        self.proc = subprocess.Popen(
            ['xz', f'--threads={n_threads}', '-5', '--to-stdout'],
            stdin=subprocess.PIPE, stdout=self.f)
        self.proc.stdin.write(msgpack_numpy.packb(metadata))
        self.quantization_mm = quantization_mm

    def append(self, **kwargs):
        compressed = compress(kwargs, quantization_mm=self.quantization_mm)
        packed = msgpack_numpy.packb(compressed)
        self.proc.stdin.write(packed)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.f.close()
            os.remove(self.f.name)
            self.proc.kill()
        else:
            self.close()


class BodyDecompressor:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.proc = subprocess.Popen(
            ['xz', '--threads=0', '--decompress', '--to-stdout'], stdin=self.f,
            stdout=subprocess.PIPE)
        self.unpacker = msgpack_numpy.Unpacker(self.proc.stdout)
        self.metadata = next(self.unpacker)

    def __iter__(self):
        return self

    def __next__(self):
        return decompress(dict(next(self.unpacker)))

    def close(self):
        self.proc.terminate()
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def quantize_diff(x, factor=2, axis=-2):
    return np.diff(np.round(x * factor), prepend=0, axis=axis).astype(np.int32), factor


def unquantize_diff(x, axis=-2):
    x, factor = x
    return np.cumsum(x, axis=axis).astype(np.float32) / factor


def compress(d, quantization_mm=0.5):
    factor = 1 / quantization_mm
    for name in ['vertices', 'joints']:
        d[f'qd_{name}'] = quantize_diff(d.pop(name), factor=factor, axis=-2)
    for name in ['vertex_uncertainties', 'joint_uncertainties']:
        if name in d:
            d[f'qd_{name}'] = quantize_diff(d.pop(name), factor=factor * 1000, axis=-1)

    if 'camera' in d:
        if isinstance(d['camera'], cameralib.Camera):
            d['camera'] = cam_to_dict(d['camera'])

    return d


def decompress(d, decode_camera=False):
    for name in ['vertices', 'joints']:
        if f'qd_{name}' in d:
            d[name] = unquantize_diff(d.pop(f'qd_{name}'), axis=-2)

    for name in ['vertex_uncertainties', 'joint_uncertainties']:
        if f'qd_{name}' in d:
            d[name] = unquantize_diff(d.pop(f'qd_{name}'), axis=-1)

    if 'camera' in d and decode_camera:
        d['camera'] = dict_to_cam(d['camera'])

    return d


def cam_to_dict(cam):
    d = dict(
        rotvec_w2c=mat2rotvec(cam.R),
        loc=cam.t,
        intr=cam.intrinsic_matrix[:2],
        up=cam.world_up)

    if cam.distortion_coeffs is not None and np.count_nonzero(cam.distortion_coeffs) > 0:
        d['distcoef'] = cam.distortion_coeffs
    return d


def dict_to_cam(d):
    return cameralib.Camera(
        rot_world_to_cam=rotvec2mat(d['rotvec_w2c']),
        optical_center=np.array(d['loc']),
        intrinsic_matrix=np.concatenate([d['intr'], np.array([[0, 0, 1]])]),
        distortion_coeffs=d.get('distcoef', None),
        world_up=d.get('up', (0, 0, 1))
    )


def rotvec2mat(rotvec):
    return R.from_rotvec(rotvec).as_matrix()


def mat2rotvec(rotmat):
    return R.from_matrix(rotmat).as_rotvec()
