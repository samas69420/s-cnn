import numpy
import matplotlib.pyplot as plt
from sconv import SConv
from dataset import spiking_dataset

plt.style.use('dark_background')
FACECOLOR = "#181135"

TRAINING_ITERATIONS = 2500
WEIGHTS_FILE = "weightsfile"

timesteps = spiking_dataset.timesteps

conv_layer = SConv(spiking_dataset.data_shape[1:],
                   num_kernels = 30,
                   kernel_shape = (2,5,5),
                   threshold = 15,
                   training = True)

conv_layer.load_weights_numpy(WEIGHTS_FILE)

num_w_in_kernel = numpy.prod(conv_layer.kernel_shape)
num_w_total = num_w_in_kernel * conv_layer.num_kernels

def train_one_image(spiking_image):
    global timesteps
    global conv_layer
    for t in range(timesteps):
        spiking_frame = spiking_image[t]
        conv_layer(spiking_frame)
    conv_layer.reset_state()

fig, axs = plt.subplots(3,10)
fig.set_facecolor(FACECOLOR)

# save a copy of all kernels before training
kernels_at_start = []
for k in conv_layer.kernels:
    kernels_at_start.append(k.copy())

# plot first 10 kernels before training (ON-center side only)
for _ in range(10):
    axs[0,_].imshow(kernels_at_start[_][0])

# training loop
for i in range(TRAINING_ITERATIONS):

    print(f"iteration: {i} / {TRAINING_ITERATIONS}")
    spiking_image = spiking_dataset[i][0]
    train_one_image(spiking_image)

    # calculate and print convergence of the layer, when it is < 0.01 the
    # training for the individual layer can be stopped
    if i % 5 == 0:
        avg_convergence = 0
        for kernel in conv_layer.kernels:
            avg_convergence += numpy.sum((kernel * (1-kernel)))
        avg_convergence = avg_convergence / num_w_total 
        print(f"avg convergence: {avg_convergence} ")

    if i % 1000 == 0:
        conv_layer.a_plus *= 2
        conv_layer.a_minus *= 2

    if i % 100 == 0:
        conv_layer.save_weights_numpy(WEIGHTS_FILE)

# plot first 10 kernels after training (ON-center side only)
for _ in range(10):
    axs[1,_].imshow(conv_layer.kernels[_][0]) 

# plot first 10 kernels deltas (ON-center side only)
for _ in range(10):
    axs[2,_].imshow(numpy.abs(conv_layer.kernels[_][0]-kernels_at_start[_][0]))

plt.show()
