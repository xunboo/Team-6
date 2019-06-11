## Introduction

The entrypoint for the colorization model is colorize.py, which takes, as input, a gray image and outputs a color version of the images.
In this case, `python colorize.py [path to filename]`. We have provided some sample black and white images under images. For example, you can try running `python colorize.py images/citizen kane.png`.

Here is an example of running `colorize.py`:

<img src="colorized_images/citizen kane gray.png" width=40%><img src="colorized_images/citizen kane.png" width=40%>


## Code Organization
- `colorize.py`: entry point that can colorize 1 image, used by website.
- `model.py`: Describes the model architecture
- `config.py`: Describes the configuration (hyperparameters) for training and inference.
- `images/`: some sample grayscale images to play with as input to `colorize.py`
- `colorized images/`: colored images that `colorize.py` has created
- `models/`: contains model's saved parameter data from training checkpoints. `model.06-2.5489.hdf5` is from Foam Liu.
- `data/`: Contains hyperparameters for translating image output from LAB space to RGB space. In particular, we utilize the pretrained `pts_in_hull.npy`


## Acknowledgements

We would like to thank Richard Zhang and his project [Colorful Image Colorization](http://richzhang.github.io/colorization/) as well as [Foam Liu](https://github.com/foamliu/Colorful-Image-Colorization) for giving us background information, training data, hyperparamters, architecture suggestions, as well as many parts of training pipeline that helped us on this project.
