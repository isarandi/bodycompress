from setuptools import setup

setup(
    name='bodycompress',
    version='0.1.0',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['bodycompress'],
    license='LICENSE',
    description='Tool for efficiently (de)serializing and (de)compressing nonparametric 3D human body pose and shape estimation results.',
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'msgpack_numpy',
        'cameralib @ git+https://github.com/isarandi/cameralib.git',
    ],
)
