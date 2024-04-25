import numpy
import matplotlib.pyplot as plt
from dataset import spiking_dataset
from layers.sconv import SConv
from layers.spool import SPool

WEIGHTS_FILE = "weightsfile"
TEST_MAP = 0

FACECOLOR = "#181135"

spiking_image = spiking_dataset[0][0] # shape = (10, 2, 27, 27)

timesteps = spiking_dataset.timesteps

conv_layer = SConv(spiking_image.shape[1:],
                   num_kernels = 30,
                   kernel_shape = (2,5,5),
                   threshold = 15)

conv_layer.load_weights_numpy(WEIGHTS_FILE)

pool_layer = SPool((conv_layer.output_shape[0],
                    conv_layer.output_shape[1]-1,
                    conv_layer.output_shape[2]-1),2)

fig, axs = plt.subplots(2,timesteps//2)
fig.suptitle("conv out ON")
fig.set_facecolor(FACECOLOR)

fig2, axs2 = plt.subplots(2,timesteps//2)
fig2.suptitle("pool ON")
fig2.set_facecolor(FACECOLOR)

for t in range(timesteps):
    spiking_frame = spiking_image[t]
    out_conv_t = conv_layer(spiking_frame)
    out_pool_t = pool_layer(out_conv_t[:,1:,1:])
    axs[t//(timesteps//2),t%(timesteps//2)].imshow(out_conv_t[TEST_MAP]) 
    axs2[t//(timesteps//2),t%(timesteps//2)].imshow(out_pool_t[TEST_MAP])

plt.show()
