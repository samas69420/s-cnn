import numpy
import matplotlib.pyplot as plt
from layers.rstdp import RSTDP

"""
in this script a full connected spiking layer is trained to recognize a simple 
pattern using the reward modulated spike timing dependent plasticity (R-STDP)
this task is performed only for testing purposes and is slightly different than
the one the layer will do when used with the rest of the spiking convolutional net

in particular in this experiment when a pattern defined as following
            pattern = [1,1,1,1,1,0,0...0] for all timesteps
is presented to the network the output should be a 0, otherwise if random spikes
are presented the output should be 9
"""

plt.style.use('dark_background')
FACECOLOR = "#181135"

fig, axs = plt.subplots(1,2)
fig.set_facecolor(FACECOLOR)
axs[0].set_title("n_hit/n")
axs[1].set_title("n_miss/n")

INPUT_SHAPE = 100
OUTPUT_SHAPE = 10
TRAIN_ITERATIONS = 100000
TIMESTEPS = 10
SAMPLE_N_RATE = 10
PATTERN_PROB = 0.5

classif_layer = RSTDP(INPUT_SHAPE, OUTPUT_SHAPE, 
                      training=True, 
                      decay = 0.01, 
                      a_p_plus=0.0005)

n_hit_values = []
n_miss_values = []

for i in range(TRAIN_ITERATIONS):

    input_spikes = numpy.random.randint(0,2,(TIMESTEPS,INPUT_SHAPE)) 
    desired_output = numpy.array([0,0,0,0,0,0,0,0,0,1])

    if numpy.random.random() < PATTERN_PROB:
        input_spikes = numpy.zeros((TIMESTEPS,INPUT_SHAPE)) 
        input_spikes[:,0:5] = 1 # pattern: constant [1,1,1,1,1,0...0] in TIMESTEPS timesteps
        desired_output = numpy.array([1,0,0,0,0,0,0,0,0,0])

    for t in range(TIMESTEPS):
        classif_layer(input_spikes[t])

    classif_layer.rstdp(desired_output)
    classif_layer.reset_state()

    if i % SAMPLE_N_RATE == 0:
        n_hit_values.append(classif_layer.n_hit_mem.sum()/classif_layer.n)
        n_miss_values.append(classif_layer.n_miss_mem.sum()/classif_layer.n)

axs[0].plot(numpy.arange(0,TRAIN_ITERATIONS, SAMPLE_N_RATE)/SAMPLE_N_RATE,n_hit_values)
axs[1].plot(numpy.arange(0,TRAIN_ITERATIONS, SAMPLE_N_RATE)/SAMPLE_N_RATE,n_miss_values)

print(f"last hits in {classif_layer.n} iters")
print(classif_layer.n_hit_mem.sum())

print(f"last misses in {classif_layer.n} iters")
print(classif_layer.n_miss_mem.sum())

classif_layer.training = False

# simulate the pattern
input_spikes = numpy.zeros((TIMESTEPS,INPUT_SHAPE)) 
input_spikes[:,0:5] = 1
for t in range(TIMESTEPS):
    result = classif_layer(input_spikes[t])
print("final test 1 (pattern presented, expected 1 at index 0)",result)

classif_layer.reset_state()

# simulate the random input
input_spikes = numpy.random.randint(0,2,(TIMESTEPS,INPUT_SHAPE))
for t in range(TIMESTEPS):
    result = classif_layer(input_spikes[t])
print("final test 2 (random input, expected 1 at index 9)",result)

fig2, axs2 = plt.subplots(1)
fig2.set_facecolor(FACECOLOR)
axs2.imshow(classif_layer.weights)


plt.show()

# NOTE

# with a_p_plus=0.005 training looks much more better in the test scenario
