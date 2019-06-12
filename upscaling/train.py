from ISR.models import RDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
from ISR.train import Trainer
import argparse


def main(args):
	lr_train_patch_size = args.patch_size
	layers_to_extract = [5, 9]
	hr_train_patch_size = lr_train_patch_size * args.scale
	if args.model == 'rdn':
		model = RDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':args.scale}, patch_size=lr_train_patch_size)
	else: 
		model = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':args.scale}, patch_size=lr_train_patch_size)
	f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
	discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)
	loss_weights = {
   		'generator': 0.0,
    	'feature_extractor': 0.0833,
    	'discriminator': 0.01,
	}

	trainer = Trainer(
	    generator=model,
	    discriminator=discr,
	    feature_extractor=f_ext,
	    lr_train_dir='low_res/training/images',
	    hr_train_dir='high_res/training/images',
	    lr_valid_dir='low_res/validation/images',
	    hr_valid_dir='high_res/validation/images',
	    loss_weights=loss_weights,
	    dataname=args.name,
	    logs_dir='./logs',
	    weights_dir='./weights',
	    weights_generator=None,
	    weights_discriminator=None,
	    n_validation=40,
	    lr_decay_frequency=30,
	    lr_decay_factor=0.5,
	)

	trainer.train(epochs=args.num_epochs,
				  steps_per_epoch=args.epoch_steps,
				  batch_size,args.batch_size)


def get_args():
	parser = argparse.ArgumentParser('Train a super resolution model')
    parser.add_argument('--scale',
                        type=int,
                        default=2,
                        help='Scale factor.')
    parser.add_argument('--patch_size',
                        type=int,
                        default=40,
                        help='Training patch size.')
    parser.add_argument('--model',
                        type=str,
                        default='rdn',
                        help='Model type to use.')
    parser.add_argument('--name',
    					'-n',
    					type=str,
    					required=True,
    					help='Name for the data.')
    parser.add_argument('--batch_size',
    					type=int,
    					default=16,
    					help='Batch size'.)
    parser.add_argument('--num_epochs',
    					type=int,
    					default=100,
    					help='Number of epochs.')
    parser.add_argument('--epoch_steps',
    					type=int,
    					default=500,
    					help='Steps per epoch.')
if __name__ == '__main__':
	main(get_args())