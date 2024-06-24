import numpy
from layers.sconv import SConv
from layers.spool import SPool
from layers.rstdp import RSTDP
from dataset import spiking_dataset

class SpikingNetwork:
# network class, it only stores and manages all the layers

    def __init__(self, layers, timesteps, training = False):

        # automatically initialize members to arguments (self.layers = layers...)
        self.__dict__.update(locals()) 

        if training:
            self.set_training_layer(0)

    def set_training_layer(self, layer_idx):

        self.training_idx = layer_idx

        for layer in self.layers:
            layer.training = False

        print(f"activating training mode for layer {layer_idx} only")
        self.layers[self.training_idx].set_training()

    def set_training(self, do_training):
        if do_training == True:
            self.training = True
            self.set_training_layer(0)
            print(f"activating training mode for layer {layer_idx} only")
        elif do_training == False:
            print(f"turning off training mode for all layers")
            self.training = False
            for layer in self.layers:
                layer.training = False

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def find_next_training_idx(self):
        for idx in range(self.training_idx+1,len(self.layers)):
            if self.layers[idx].trainable:
                return idx
        return None

    def forward(self, spiking_frame):

        out_C1 = self.layers[0].forward(spiking_frame)
        out_P1 = self.layers[1].forward(out_C1[:,1:,1:])
        out_C2 = self.layers[2].forward(out_P1)
        out_P2 = self.layers[3].forward(out_C2)
        out_RSTDP = self.layers[4].forward(out_P2.flatten())

        return out_RSTDP

    def save_weights_numpy(self):
        for i,layer in enumerate(self.layers):
            layer.save_weights_numpy(f"../weights/layer_{i}.weights")

    def load_weights_numpy(self):
        for i,layer in enumerate(self.layers):
            layer.load_weights_numpy(f"../weights/layer_{i}.weights")

    def __call__(self, spiking_frame):
        return self.forward(spiking_frame)

TIMESTEPS = spiking_dataset.timesteps

conv1 = SConv(input_shape = spiking_dataset.data_shape[1:],
              num_kernels = 30,
              kernel_shape = (2,5,5),
              threshold = 15)

# first row and col are removed during forward and output shape changes
conv1_split_output_shape = (conv1.output_shape[0],
                            conv1.output_shape[1]-1,
                            conv1.output_shape[2]-1)

pool1 = SPool(input_shape = conv1_split_output_shape, 
              window_size = 2)

conv2 = SConv(input_shape = pool1.output_shape,
              num_kernels = 750,
              kernel_shape = (30,5,5),
              threshold = 10)

pool2 = SPool(input_shape = conv2.output_shape, 
              window_size = 2)

classif_layer = RSTDP(input_shape = numpy.prod(pool2.output_shape), 
                      output_shape = 10,
                      a_r_plus = 0.004,
                      a_r_minus = 0.003,
                      a_p_plus = 0.0005,
                      a_p_minus = 0.004,
                      init_hit_ratio = 0.9,
                      n = 100)

network = SpikingNetwork(timesteps = TIMESTEPS,
                         training = True,
                         layers=[conv1,
                                 pool1,
                                 conv2,
                                 pool2,
                                 classif_layer])

if __name__ == "__main__":

    TIMESTEPS = spiking_dataset.timesteps
    network.load_weights_numpy()
    network.set_training(False)

    # short test (test 1 image only)

    index = 0
    data = spiking_dataset[index]
    spiking_image = data[0]
    print("correct label: ", data[1])
    
    for t in range(TIMESTEPS):
        spiking_frame = spiking_image[t]
        print(network.forward(spiking_frame))

    network.reset_state()

    input("press enter to continue and test the last 10k images...")

    # test loop 
    # compute accuracy on the last 10000 images in the dataset, these images  
    # are not used for training but are used as a test set instead

    nhit = 0
    nmiss = 0
    n = 0

    for i in range(10000):
        (spiking_image, label) = spiking_dataset[i+50000]
        ret = None
        n += 1

        for t in range(TIMESTEPS):
            spiking_frame = spiking_image[t]
            ret = network(spiking_frame)

        if numpy.all(ret == label):
            nhit += 1
        else:
            nmiss += 1

        print("iter ",i," - nhit:", nhit,"nmiss:",nmiss, " - accuracy: ", nhit/n)

        network.reset_state()

# TODO

# rewrite the management of the training state, its not really clean
