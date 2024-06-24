import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
import numpy
from dataset import spiking_dataset
from network import network
import argparse

parser = argparse.ArgumentParser(description="render an animation for the forward pass of the network with one image")
parser.add_argument("--save", action="store_true", default=False, help="use this flag if you want to save the animation as .gif file")
args = parser.parse_args()

TIMESTEPS = spiking_dataset.timesteps

network.load_weights_numpy()
network.set_training(False)

spiking_image = spiking_dataset[0][0]

buffer = numpy.zeros((TIMESTEPS,10*network.layers[0].output_shape[-2],3*network.layers[0].output_shape[-1])) 
buffer2 = numpy.zeros((TIMESTEPS,10*network.layers[1].output_shape[-2],3*network.layers[1].output_shape[-1])) 
buffer3 = numpy.zeros((TIMESTEPS,50*network.layers[2].output_shape[-2],15*network.layers[2].output_shape[-1])) 
buffer4 = numpy.zeros((TIMESTEPS,50*network.layers[3].output_shape[-2],15*network.layers[3].output_shape[-1])) 
buffer5 = numpy.zeros((TIMESTEPS,network.layers[4].output_shape,1)) 

# preload buffers with convolution and pool output for each timestep
for t in range(TIMESTEPS):
    spiking_frame = spiking_image[t]

    c1_output = network.layers[0](spiking_frame)
    p1_output = network.layers[1](c1_output[:,1:,1:])
    c2_output = network.layers[2](p1_output)
    p2_output = network.layers[3](c2_output)
    class_output = network.layers[4](p2_output.flatten())

    c1_output_reshaped = c1_output.reshape((10*network.layers[0].output_shape[-2],3*network.layers[0].output_shape[-1]))
    p1_output_reshaped = p1_output.reshape((10*network.layers[1].output_shape[-2],3*network.layers[1].output_shape[-1]))
    c2_output_reshaped = c2_output.reshape((50*network.layers[2].output_shape[-2],15*network.layers[2].output_shape[-1]))
    p2_output_reshaped = p2_output.reshape((50*network.layers[3].output_shape[-2],15*network.layers[3].output_shape[-1]))
    class_output_reshaped = class_output.reshape((network.layers[4].output_shape,1))
    
    buffer[t] = c1_output_reshaped
    buffer2[t] = p1_output_reshaped
    buffer3[t] = c2_output_reshaped
    buffer4[t] = p2_output_reshaped
    buffer5[t] = class_output_reshaped

FACECOLOR = "#181135"
ANIMATION_FILE = 'animation.gif'

fig = plt.figure(figsize=(12,10))
fig.set_facecolor(FACECOLOR)
gs = gridspec.GridSpec(2,6)

axs1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
axs1.set_title("ON-center spikes")

axs2 = fig.add_subplot(gs[1, 0]) # row 0, col 1
axs2.set_title("OFF-center spikes")

axs3 = fig.add_subplot(gs[:, 1]) # col 1, span all rows 
axs3.set_title("C1 maps spikes")

axs4 = fig.add_subplot(gs[:, 2])
axs4.set_title("P1 maps spikes")

axs5 = fig.add_subplot(gs[:, 3])
axs5.set_title("C2 maps spikes")

axs6 = fig.add_subplot(gs[:, 4]) 
axs6.set_title("P2 maps spikes")

axs7 = fig.add_subplot(gs[:, 5]) 
axs7.set_title("classification spikes")

im  = axs1.imshow(spiking_image[0][0],  animated=True)
im2 = axs2.imshow(spiking_image[0][1],  animated=True)
im3 = axs3.imshow(buffer[0],            animated=True)
im4 = axs4.imshow(buffer2[0],           animated=True)
im5 = axs5.imshow(buffer3[0],           animated=True)
im6 = axs6.imshow(buffer4[0],           animated=True)
im7 = axs7.imshow(buffer5[0],           animated=True)

i=0

def updatefig(*args):

    global i

    i+=1
    spiking_frame = spiking_image[i%10]

    im.set_array(spiking_frame[0])
    im2.set_array(spiking_frame[1])

    im3.set_array(buffer[i%10])

    # https://stackoverflow.com/questions/43232260/python-matplotlib-animation-doesnt-show
    im3.autoscale() 

    im4.set_array(buffer2[i%10])
    im4.autoscale() 

    im5.set_array(buffer3[i%10])
    im5.autoscale() 

    im6.set_array(buffer4[i%10])
    im6.autoscale() 

    im7.set_array(buffer5[i%10])
    im7.autoscale() 

    return (im,im2,im3,im4,im5,im6,im7)

anim = animation.FuncAnimation(fig, updatefig, interval=100, cache_frame_data=False, frames=numpy.arange(0,200))#, blit=True)

if args.save:
    print("saving animation, it may take some time...")
    anim.save(ANIMATION_FILE)
else:
    print("No save option provided, use --save if you want to save the animation")

plt.show()

# TODO

# refactor to remove duplicate code
