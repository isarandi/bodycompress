import os
import os.path as osp
import queue
import subprocess
import threading
from collections.abc import Iterator, Iterable
from contextlib import AbstractContextManager

import cameravision
import msgpack_numpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional


class BodyCompressor(AbstractContextManager):
    """Compresses body data to a file using xz compression.

    The data is quantized, difference encoded, serialized using ``msgpack_numpy`` then compressed
    using ``xz`` and written to the file.

    Args:
        path: path to the output file
        metadata: metadata to be stored in the beginning of the file
        quantization_mm: quantization level for the vertices and joints in millimeters.
            Coordinates are rounded to the nearest multiple of this value.

    """

    def __init__(
        self,
        path: str,
        metadata: Optional[dict] = None,
        quantization_mm: float = 0.5,
        n_threads: int = 0,
        queue_size=8,
    ):
        os.makedirs(osp.dirname(path), exist_ok=True)
        self.f = open(path, "wb")
        self.length = 0
        self.data_start = None

        header_compr = self._make_header()
        self.f.write(header_compr)
        self.f.seek(64, os.SEEK_CUR)  # leave space for the header change
        self.metadata_start = self.f.tell()

        metadata_compr = subprocess.check_output(["xz", "-5", "-qq"], input=msgpack_numpy.packb(metadata))
        self.f.write(metadata_compr)
        self.f.flush()
        self.data_start = self.f.tell()

        self.proc = subprocess.Popen(
            ["xz", f"--threads={n_threads}", "-5", "--to-stdout"],
            stdin=subprocess.PIPE,
            stdout=self.f,
        )
        self.quantization_mm = quantization_mm
        self.q = queue.Queue(queue_size)
        self.thread = threading.Thread(target=self._write_thread)
        self.thread.start()

    def _make_header(self):
        header = dict(
            __bodycompress_version__=(0, 2, 1), length=self.length, data_start=self.data_start
        )
        return subprocess.check_output(["xz", "-5", "-qq"], input=msgpack_numpy.packb(header))

    def append(self, **kwargs):
        """Append data for frame to the file.

        Args:
            **kwargs: data to be stored. Supported keys are

                - vertices: (N, 3) float32 array of vertices in millimeters
                - joints: (N, 3) float32 array of joints in millimeters
                - vertex_uncertainties: (N,) float32 array of vertex uncertainties in meters
                - joint_uncertainties: (N,) float32 array of joint uncertainties in meters
                - camera: cameravision.Camera object or dict with camera parameters

            Other keys are also allowed, but they will not be quantized.
        """
        self.q.put(kwargs)

    def _write_thread(self):
        try:
            while True:
                kwargs = self.q.get()
                if kwargs == "forceful_close":
                    self._forceful_close()
                elif kwargs is None:
                    self._close()
                else:
                    compressed = compress(kwargs, quantization_mm=self.quantization_mm)
                    packed = msgpack_numpy.packb(compressed)
                    self.proc.stdin.write(packed)
                    self.length += 1

                self.q.task_done()

                if kwargs is None or kwargs == "forceful_close":
                    return
        except BaseException:
            self._forceful_close()
            raise

    def close(self):
        """Wait for the currently pending compression to finish then close the file."""
        self.q.put(None)
        self.q.join()
        self.thread.join()

    def forceful_close(self):
        self.q.put("forceful_close")
        self.q.join()
        self.thread.join()

    def _close(self):
        if self.proc.stdin:
            self.proc.stdin.close()

        self.proc.wait()
        new_header = self._make_header()
        if len(new_header) <= self.metadata_start:
            self.f.seek(0)
            self.f.write(new_header)
        else:
            raise ValueError("New header too large")

        self.f.close()

    def _forceful_close(self):
        self.proc.kill()
        self.f.close()
        os.remove(self.f.name)

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the file and remove it if an exception occurred."""
        if exc_type is not None:
            self.forceful_close()
        else:
            self.close()


class BodyDecompressor(Iterable):
    """Decompresses body data from a file compressed using BodyCompressor.

    The data is decompressed using xz, deserialized using msgpack_numpy, difference decoded and
    unquantized.

    Args:
        path: path to the compressed file
    """

    def __init__(self, path):
        self.path = path
        self.version, self.length, self.metadata, self._data_start = self._read_initial(path)

    @staticmethod
    def _read_initial(path):
        with open(path, "rb") as f:
            with subprocess.Popen(
                ["xz", "--threads=0", "--decompress", "--to-stdout", "-qq"],
                stdin=f,
                stdout=subprocess.PIPE,
            ) as proc:
                unpacker = msgpack_numpy.Unpacker(proc.stdout)
                header_or_meta = next(unpacker)
                if (
                    isinstance(header_or_meta, dict)
                    and '__bodycompress_version__' in header_or_meta
                ):
                    header = header_or_meta
                    version = header['__bodycompress_version__']
                    length = header['length']
                    data_start = header['data_start']
                    metadata = next(unpacker)
                else:
                    version = (0, 1, 0)
                    metadata = header_or_meta
                    for key in ['n_frames', 'num_frames', 'nframes']:
                        if key in metadata:
                            length = metadata[key]
                            break

                    data_start = None

        return version, length, metadata, data_start

    def __iter__(self):
        with open(self.path, "rb") as f:
            if self._data_start is not None:
                f.seek(self._data_start)

            with subprocess.Popen(
                ["xz", "--threads=0", "--decompress", "--to-stdout", "-qq"],
                stdin=f,
                stdout=subprocess.PIPE,
            ) as proc:
                unpacker = iter(msgpack_numpy.Unpacker(proc.stdout))
                if self._data_start is None:
                    next(unpacker)  # skip metadata

                for x in unpacker:
                    yield decompress(dict(x))

    def __len__(self):
        return self.length


def quantize_diff(x, factor=2, axis=-2):
    return np.diff(np.round(x * factor), prepend=0, axis=axis).astype(np.int32), factor


def unquantize_diff(x, axis=-2):
    x, factor = x
    return np.cumsum(x, axis=axis).astype(np.float32) / factor


def compress(d, quantization_mm=0.5):
    factor = 1 / quantization_mm
    for name in ["vertices", "joints"]:
        d[f"qd_{name}"] = quantize_diff(d.pop(name), factor=factor, axis=-2)
    for name in ["vertex_uncertainties", "joint_uncertainties"]:
        if name in d:
            d[f"qd_{name}"] = quantize_diff(d.pop(name), factor=factor * 1000, axis=-1)

    if "camera" in d:
        if isinstance(d["camera"], cameravision.Camera):
            d["camera"] = cam_to_dict(d["camera"])

    return d


def decompress(d, decode_camera=False):
    for name in ["vertices", "joints"]:
        if f"qd_{name}" in d:
            d[name] = unquantize_diff(d.pop(f"qd_{name}"), axis=-2)

    for name in ["vertex_uncertainties", "joint_uncertainties"]:
        if f"qd_{name}" in d:
            d[name] = unquantize_diff(d.pop(f"qd_{name}"), axis=-1)

    if "camera" in d and decode_camera:
        d["camera"] = dict_to_cam(d["camera"])

    return d


def cam_to_dict(cam):
    d = dict(
        rotvec_w2c=mat2rotvec(cam.R),
        loc=cam.t,
        intr=cam.intrinsic_matrix[:2],
        up=cam.world_up,
    )

    if cam.has_distortion():
        d["distcoef"] = cam.distortion_coeffs
    return d


def dict_to_cam(d):
    return cameravision.Camera(
        rot_world_to_cam=rotvec2mat(d["rotvec_w2c"]),
        optical_center=np.array(d["loc"]),
        intrinsic_matrix=np.concatenate([d["intr"], np.array([[0, 0, 1]])]),
        distortion_coeffs=d.get("distcoef", None),
        world_up=d.get("up", (0, 0, 1)),
    )


def rotvec2mat(rotvec):
    return R.from_rotvec(rotvec).as_matrix().astype(np.float32)


def mat2rotvec(rotmat):
    return R.from_matrix(rotmat).as_rotvec().astype(np.float32)
