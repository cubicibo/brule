# Brule
Collection of C extension modules for SUPer with pure-python fallbacks
- Brule: Bitmap RUn LEngth coder and decoder.
- LayoutEngine: An optimal two-rects layout finder for a sequence of images.
- HexTree: RGBA image quantizer (turbo, good quality) - based on hexagonal tree folding
- QtzrUTC: RGBA image quantizer (fast, excellent quality) - based on unidimensional threshold clustering

## Brule brief
Brule implements 2 or 3 times the same function, and select the fastest implementation available at runtime.
- C (fastest)
- numba (only for the RLE codec)
- pure Python (slow)

For the encoder, the C implementation is up to 20 times faster than numba. The pure Python implementation does not compete and is only there for convenience.

## Install
Given `./brule` the (cloned) repository:
```bash
$ python -m pip install ./brule
```

## Example (RLE Codec)

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

## Example (HexTree, QtzrUTC)
The image quantizer is used like this:
```python
>>> from brule import HexTree, QtzrUTC
>>> import numpy as np
>>> from PIL import Image
>>> rgba = np.asarray(Image.open(...).convert('RGBA'))
>>> #quantize with 255 colours 
>>> bitmap1, palette1 = HexTree.quantize(rgba, 255)
>>> bitmap2, palette2 = QtzrUTC.quantize(rgba, 255)
```

## Credits
- https://github.com/pallets/markupsafe markupsafe for the pip install fallback mechanism with compiled extensions.
- https://github.com/DarthSim/quantizr for WClustering
