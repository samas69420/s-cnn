import numpy
import matplotlib.pyplot as plt
from sconv import SConv
from dataset import spiking_dataset

WEIGHTS_FILE = "weightsfile"
TEST_MAP = 0
TEST_NEURON = (5,10)

FACECOLOR = "#181135"

timesteps = spiking_dataset.timesteps

conv_layer = SConv(spiking_dataset.data_shape[1:],
                   num_kernels = 30,
                   kernel_shape = (2,5,5),
                   threshold = 15,
                   training = False)

conv_layer.load_weights_numpy(WEIGHTS_FILE)

spiking_image = spiking_dataset[0][0]

fig, axs = plt.subplots(2,timesteps//2)
fig.suptitle("Potentials")
fig.set_facecolor(FACECOLOR)

fig2, axs2 = plt.subplots(1)
fig2.suptitle(f"neuron {TEST_NEURON} potential")
fig2.set_facecolor(FACECOLOR)

fig3, axs3 = plt.subplots(2,timesteps//2)
fig3.suptitle("Accumulations")
fig3.set_facecolor(FACECOLOR)

fig4, axs4 = plt.subplots(2,timesteps//2)
fig4.suptitle(f"Output spikes from map {TEST_MAP}")
fig4.set_facecolor(FACECOLOR)

plotpot = []
accum_tot = numpy.zeros(conv_layer.output_shape[1:])

# force spike in test map (for testing purposes)
#conv_layer.kernels[TEST_MAP] = numpy.ones((2,5,5)) 

for t in range(timesteps):

    spiking_frame = spiking_image[t]
    s = conv_layer.forward(spiking_frame)
    accum_tot += s.sum(0)

    # record potential of test neuron
    plotpot.append(conv_layer.potentials[TEST_MAP,*TEST_NEURON]) 

    axs[t//(timesteps//2),t%(timesteps//2)].imshow(conv_layer.potentials[TEST_MAP]) # plot first map of conv layer
    axs3[t//(timesteps//2),t%(timesteps//2)].imshow(s.sum(0))
    axs4[t//(timesteps//2),t%(timesteps//2)].imshow(s[TEST_MAP])

    # show test neuron's spike in the same figure as test neuron's potential
    if s[TEST_MAP, *TEST_NEURON] == 1:
        axs2.scatter(t, conv_layer.threshold, color="red")

axs2.step(numpy.linspace(0,timesteps-1,timesteps),numpy.array(plotpot))


fig5, axs5 = plt.subplots(1)
fig5.suptitle("Total accumulations")
fig5.set_facecolor(FACECOLOR)
axs5.imshow(accum_tot)

plt.show()
