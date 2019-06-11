import logging

import azure.functions as func

from PIL import Image
from io import BytesIO
import sys


def main(inputImage: func.InputStream,
         upscaledImage: func.Out[bytes]):
    logging.info(f"Python blob trigger function processing blob \n"
                 f"Name: {inputImage.name}\n"
                 f"Blob Size: {inputImage.length} bytes")
    try:
        img = Image.open(inputImage)
    except OSError as _:
        logging.info(f"Unable to load image. Aborting...")
        sys.exit(129)
    newSize = tuple(dim * 2 for dim in img.size)
    upscaled = img.resize(newSize, Image.BILINEAR)
    logging.info(f"Upscaled image to new size {upscaled.size}")
    outStream = BytesIO()
    upscaled.convert("RGB").save(outStream, format="JPEG")
    upscaledImage.set(outStream.getvalue())
