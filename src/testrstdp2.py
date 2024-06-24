import numpy
import matplotlib.pyplot as plt
from layers.rstdp import RSTDP

"""
in this script a full connected spiking layer is trained to recognize a simple 
pattern using the reward modulated spike timing dependent plasticity (R-STDP)
this task is performed only for testing purposes and is slightly different than
the one the layer will do when used with the rest of the spiking convolutional net

in this experiment the pattern is not constant in time like in the testrstdp.py 
and the network should classify the input considering what happens at every 
timestep, the network will perform classification between two fixed patterns
(also called stimulus) without using random input
"""

plt.style.use('dark_background')
FACECOLOR = "#181135"

fig, axs = plt.subplots(1,2)
fig.set_facecolor(FACECOLOR)
axs[0].set_title("n_hit/n")
axs[1].set_title("n_miss/n")

fig2, axs2 = plt.subplots(1)
fig2.set_facecolor(FACECOLOR)

INPUT_SHAPE = 4
OUTPUT_SHAPE = 2
TRAIN_ITERATIONS = 100000
TIMESTEPS = 10
SAMPLE_N_RATE = 5
PATTERN_PROB = 0.5

classif_layer = RSTDP(INPUT_SHAPE, OUTPUT_SHAPE, 
                      training = True, 
                      a_p_plus=0.0005, 
                      a_p_minus=0.005, 
                      decay=0)

n_hit_values = []
n_miss_values = []

def stimulus_from_array(s_arr):
# takes an array of spiking times for each neuron and return another array
# containing the actual spikes for each timestep
    timesteps = s_arr.max()
    result = numpy.zeros((timesteps, *s_arr.shape))
    for t in range(timesteps):
        result[t] = (s_arr == t+1)
    return result

for i in range(TRAIN_ITERATIONS):

    input_spikes = stimulus_from_array(numpy.array([1,0,0,TIMESTEPS]))
    desired_output = numpy.array([1,0])

    if numpy.random.random() < PATTERN_PROB:
        input_spikes = stimulus_from_array(numpy.array([1,0,TIMESTEPS,0,]))
        desired_output = numpy.array([0,1])

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

# simulate the first pattern
input_spikes = stimulus_from_array(numpy.array([1,0,0,TIMESTEPS]))
for t in range(TIMESTEPS):
    result = classif_layer(input_spikes[t])
print("final test 1 (expected 1 at index 0)",result)

classif_layer.reset_state()

# simulate the second pattern
input_spikes = stimulus_from_array(numpy.array([1,0,TIMESTEPS,0,]))
for t in range(TIMESTEPS):
    result = classif_layer(input_spikes[t])
print("final test 2 (expected 1 at index 1)",result)

axs2.imshow(classif_layer.weights)

plt.show()

# NOTE

# with a_p_plus=0.005 training looks much more better in the test scenario
