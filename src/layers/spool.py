import numpy

use_slow_pool = False # can't use the same patter as the one used for conv because mask should be passed by reference
try:
    from fastpool import pool2d
except ModuleNotFoundError:
    print("can't find fastpool module, did you compile fastpool.c?")
    input("press enter to proceed with super slow python-only pooling implementation")
    use_slow_pool = True

# 1-line utils
values_from_nol_windows = lambda arr3d,ws2d: numpy.stack([[[arr3d[k,i:i+ws2d,j:j+ws2d] for j in range(0,arr3d.shape[-1],ws2d)] for i in range(0,arr3d.shape[-2],ws2d)] for k in range(arr3d.shape[-3])]) 

class SPool:

    def __init__(self, input_shape, window_size):

        self.trainable = False

        self.window_size = window_size
        self.input_shape = input_shape

        self.output_shape = (input_shape[0],                                \
                             int(numpy.ceil(input_shape[1] / window_size)), \
                             int(numpy.ceil(input_shape[2] / window_size)))

        print(f"creating SPool layer - out shape: {self.output_shape}")

        # to ensure that like in other layers every neuron can spike only once
        self.done_mask = numpy.ones(self.output_shape)

    # load and save functions are actually useless here since there are no
    # trainable weights in pool layer, however these functions are needed
    # to iterate above all layers in network class 
    def save_weights_numpy(self, dummy):
        return None
    def load_weights_numpy(self, dummy):
        return None
    def set_training(self):
        return None

    def reset_done_mask(self):
        self.done_mask = numpy.ones(self.done_mask.shape)

    def reset_state(self):
        self.reset_done_mask()

    def forward(self, spiking_frame):

        if ((self.input_shape[-1] % self.window_size != 0) or (self.input_shape[-2] % self.window_size != 0)):
            spiking_frame = numpy.pad(spiking_frame,((0,0),
                                                     (0,self.window_size-spiking_frame.shape[-2]%self.window_size),
                                                     (0,self.window_size-spiking_frame.shape[-1]%self.window_size)))

        if use_slow_pool:
            result = numpy.zeros(self.output_shape)
            windows = values_from_nol_windows(spiking_frame, self.window_size) 
            for k in range(windows.shape[0]):
                for i in range(windows.shape[1]):
                    for j in range(windows.shape[2]):
                        window = windows[k,i,j]
                        if numpy.sum(window) >= 1 and self.done_mask[k,i,j] == 1:
                            result[k,i,j] = 1
                            self.done_mask[k,i,j] = 0
            return result
        else:
            return pool2d(spiking_frame, self.done_mask, self.window_size)

    def __call__(self, spiking_frame):
        return self.forward(spiking_frame)

# TODO

# a parent class for all layers could be a good idea so useless functions like
# load and save could be omitted here

# add support for windows with shape n,m with n!=m
