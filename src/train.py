import random
from dataset import spiking_dataset
from network import network

TRAIN_ITERATIONS = 1000000
CONVERGENCE_THRESHOLD = 0.01
SAVE_FREQ = 100
TIMESTEPS = network.timesteps

network.load_weights_numpy()

for i in range(TRAIN_ITERATIONS):

    (spiking_image, label) = spiking_dataset[random.randint(0,50000)]

    for t in range(TIMESTEPS):
        spiking_frame = spiking_image[t]
        network.forward(spiking_frame)

    if network.training_idx == len(network.layers)-1:
        network.layers[-1].rstdp(label)

    avg_convergence = network.layers[network.training_idx].get_avg_w_convergence()
    if network.training_idx == 4:
        print(f"iter {i} - avg_convergence for layer {network.training_idx}:", avg_convergence, "nhit/nmiss:",int(network.layers[network.training_idx].n_hit_mem.sum()),"/",int(network.layers[network.training_idx].n_miss_mem.sum()),network.layers[network.training_idx].potentials.argmax(), label.argmax())
    else:
        print(f"iter {i} - avg_convergence for layer {network.training_idx}:", avg_convergence)

    network.reset_state()

    if i % SAVE_FREQ == 0:
        network.save_weights_numpy()

    if avg_convergence < CONVERGENCE_THRESHOLD:
        next_idx = network.find_next_training_idx()
        if next_idx:
            network.set_training_layer(next_idx) 
        else:
            print("all layers trained")
            break
