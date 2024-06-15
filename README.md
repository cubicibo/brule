# Brule
Collection of C extension modules for SUPer with pure-python fallbacks
- Brule: Bitmap RUn LEngth coder and decoder.
- LayoutEngine: An optimal two-rects layout finder for a sequence of images.

## Brule brief
Brule implements 3 times the same run-length encoder and decoder.
- C (fastest)
- numba (fast)
- pure Python (slow)

For the encoder, the C implementation is up to 20 times faster than numba. The pure Python implementation does not compete and is only there for convenience.

## Install
```bash
$ python -m pip install brule
```

## Example

The run-length codec is operated like this:
```python
>>> from brule import Brule
>>> import numpy as np
>>> bitmap_orig = np.ones((1080, 1920), np.uint8)
>>> rle_data = Brule.encode(bitmap_orig)
>>> bitmap = Brule.decode(rle_data)
>>> assert np.all(bitmap == bitmap_orig)
True
```

## Credits
- https://github.com/pallets/markupsafe markupsafe for the pip install fallback mechanism with compiled extensions.
