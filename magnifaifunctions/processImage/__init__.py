import logging

import azure.functions as func

from PIL import Image
from io import BytesIO
import sys, os


def main(inputImage: func.InputStream,
         upscaledImage: func.Out[bytes]):
    # Make sure we can get around Python's import system
    sys.path.insert(0, os.path.join(os.getcwd(), 'processImage'))
    from upscale_image import upscale_image
    
    logging.info(f"Python blob trigger function processing blob \n"
                 f"Name: {inputImage.name}\n"
                 f"Blob Size: {inputImage.length} bytes")
    try:
        img = Image.open(inputImage)
    except OSError as _:
        logging.info(f"Unable to load image. Aborting...")
        sys.exit(129)
    #newSize = tuple(dim * 2 for dim in img.size)
    #upscaled = img.resize(newSize, Image.BILINEAR)
    upscaled = upscale_image(img)
    logging.info(f"Upscaled image to new size {upscaled.size}")
    outStream = BytesIO()
    upscaled.convert("RGB").save(outStream, format="JPEG")
    upscaledImage.set(outStream.getvalue())
