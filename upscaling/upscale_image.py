from ISR.models import RDN
import numpy as np
from PIL import Image
import sys

''' 
	Expects pillow images
	returns a Pillow Image object
'''
def upscale_image(img):
    img = img.convert('RGB')
    lr_img = np.array(img)
    assert(lr_img.shape[2] == 3)
    rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
    rdn.model.load_weights('image-super-resolution/weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')

    sr_img = rdn.predict(lr_img)
    return Image.fromarray(sr_img)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Please input path to single image to upscale")
    else:
        img_path = sys.argv[1]
        print("Upscaling image ", img_path)
        img = Image.open(img_path)
        img = upscale_image(img)
        img.save(img_path[:-4] + "_sr.png")
