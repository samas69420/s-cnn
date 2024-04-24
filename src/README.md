# Overview

this is the (unofficial) implementation of the spiking convolution and spiking pooling layers described in the papers:
1) https://arxiv.org/pdf/1903.12272.pdf
2) https://arxiv.org/pdf/1611.01421.pdf
(full network implementation including the final classification layers coming soon)

pooling layers don't have trainable weights and convolution layer is trained individually in a full unsupervised manner using a simplified version of the stdp (spike timing dependent plasticity)

in this implementation the only dependencies needed are matplotlib and numpy to handle tensors, everything else is done from scratch, this was done to have the lowest total number of dependencies as possible in order to make the code portable and easy to use without messing too much with python versions, package managers and so on 
unfortunately though some "hot" functions like convolutions are executed entirely in python and this makes the training super slow (1 iteration of the training loop in like 1 second on my machine), for this reason compiled code will be probably included in the future

# How to use

You can start the training of the convolution layer locally by just using the following commands
```
$ git clone https://github.com/samas69420/s-cnn
$ cd s-cnn/src
$ python trainconv.py
```
the training will start and the trained weights will be saved automatically every 100 iterations
after the training is over you can also use other scripts to have more informations about what happens to the data during the process
- `testconv.py` will process one spiking image with a convolution layer and generate various plots like total spikes accumulations for the whole layer, spikes of a map at every timestep and potential of a neuron
- `testpool.py` will process the output of a convolution layer with a pooling layer, the results for each timestep will be plotted
- `dataset.py` will read and unserialize a regular image from mnist dataset files and finally turn it into a spiking image, ready to be fed into the fist convolution layer
- `plotkernels.py` kinda self explainatory lol
- `animate.py` will both show and save an animation of some layers' state while processing a spiking image (saving the animation may take some seconds)


# Training results

after 2500 iterations of the training loop it is clear that the most frequent features have been learned (basically vertical and horizontal edges), however some features are not very definite yet and more training iterations are needed to make kernels converge
