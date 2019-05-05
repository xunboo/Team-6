from ISR.models import RDN
import numpy as np
from PIL import Image
import sys

''' 
	Expects local or direct path to image to upscale
	returns a Pillow Image object
'''
def upscale_image(img_path):

	img = Image.open(img_path)
	lr_img = np.array(img)
	rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
	rdn.model.load_weights('image_upscaling/weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')

	sr_img = rdn.predict(lr_img)
	return Image.fromarray(sr_img)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Please input path to single image to upscale")
		return
	img_path = sys.argv[2]
	img = upscale_image(img_path)
	img.save(img_path[:-3] + "_sr.png")