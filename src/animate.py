import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
import numpy
from sconv import SConv
from spool import SPool
from dataset import spiking_dataset

timesteps = spiking_dataset.timesteps

conv_layer = SConv(spiking_dataset.data_shape[1:],
                   num_kernels = 30,
                   kernel_shape = (2,5,5),
                   threshold = 15)

conv_layer.load_weights_numpy("weightsfile")

pool_layer = SPool((conv_layer.output_shape[0],
                    conv_layer.output_shape[1]-1,
                    conv_layer.output_shape[2]-1),2)

conv_layer2 = SConv(pool_layer.output_shape,
                    num_kernels = 100,
                    kernel_shape = (30,5,5),
                    threshold = 10)

spiking_image = spiking_dataset[1][0]

# buffer.size * buffer.itemsize = 1269600 bytes (1.2 MB)
buffer = numpy.zeros((timesteps,10*conv_layer.output_shape[-2],3*conv_layer.output_shape[-1])) 

buffer2 = numpy.zeros((timesteps,10*pool_layer.output_shape[-2],3*pool_layer.output_shape[-2])) 

buffer3 = numpy.zeros((timesteps,10*conv_layer2.output_shape[-2],10*conv_layer2.output_shape[-1])) 

# preload buffers with convolution and pool output for each timestep
for t in range(spiking_dataset.timesteps):
    spiking_frame = spiking_image[t]

    c1_output = conv_layer(spiking_frame)
    p1_output = pool_layer(c1_output[:,1:,1:])
    c2_output = conv_layer2(p1_output)

    c1_output_reshaped = c1_output.reshape((10*conv_layer.output_shape[-2],3*conv_layer.output_shape[-1]))
    p1_output_reshaped = p1_output.reshape((10*pool_layer.output_shape[-2],3*pool_layer.output_shape[-1]))
    c2_output_reshaped = c2_output.reshape((10*conv_layer2.output_shape[-2],10*conv_layer2.output_shape[-1]))
    
    buffer[t] = c1_output_reshaped
    buffer2[t] = p1_output_reshaped
    buffer3[t] = c2_output_reshaped

FACECOLOR = "#181135"
ANIMATION_FILE = 'animation.gif'

fig = plt.figure(figsize=(12,10))
fig.set_facecolor(FACECOLOR)
gs = gridspec.GridSpec(2,4)

axs1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
axs1.set_title("ON-center spikes")

axs2 = fig.add_subplot(gs[1, 0]) # row 0, col 1
axs2.set_title("OFF-center spikes")

axs3 = fig.add_subplot(gs[:, 1]) # col 1, span all rows 
axs3.set_title("C1 maps spikes")

axs4 = fig.add_subplot(gs[:, 2]) # col 2, span all rows 
axs4.set_title("P1 maps spikes")

axs5 = fig.add_subplot(gs[:, 3]) # col 3, span all rows 
axs5.set_title("C2 maps spikes")

im  = axs1.imshow(spiking_image[0][0],  animated=True)
im2 = axs2.imshow(spiking_image[0][1],  animated=True)
im3 = axs3.imshow(buffer[0],            animated=True)
im4 = axs4.imshow(buffer2[0],           animated=True)
im5 = axs5.imshow(buffer3[0],           animated=True)

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
    return (im,im2,im3,im4,im5)

anim = animation.FuncAnimation(fig, updatefig, interval=100, cache_frame_data=False, frames=numpy.arange(0,200))#, blit=True)

print("saving animation...")
anim.save(ANIMATION_FILE)

plt.show()

# TODO

# refactor to remove duplicate code
