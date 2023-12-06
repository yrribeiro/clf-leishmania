import config
from PIL import Image
from PIL.ImageEnhance import Contrast

"""
Enhances the contrast of all images in a specified directory and saves the enhanced images to another directory.

Requires:
- The `config` module, containing configurations such as `RAW_IMAGE_DIR`, `ENHANCED_IMG_DIR`, and `ENHANCER_FACTOR`.
- The Pillow library for image processing.

Note:
- The script assumes that the `ENHANCED_IMG_DIR` directory already exists.
- It only processes images with a '.jpg' extension.

Returns:
None
"""

total_imgs_found = len(list(config.RAW_IMAGE_DIR.glob('*.jpg')))

for path in config.RAW_IMAGE_DIR.glob('*.jpg'):
    with Image.open(path) as img:
        contrast = Contrast(img)
        img_contrast = contrast.enhance(config.ENHANCER_FACTOR)
        output_name = path.stem + '_enhanced.jpg'
        img_contrast.save(config.ENHANCED_IMG_DIR / output_name)

print(f'{total_imgs_found} enhanced images saved at {config.ENHANCED_IMG_DIR.name} folder!')