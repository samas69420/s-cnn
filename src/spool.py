import numpy

# 1-line utils
values_from_nol_windows = lambda arr3d,ws2d: numpy.stack([[[arr3d[k,i:i+ws2d,j:j+ws2d] for j in range(0,arr3d.shape[-1],ws2d)] for i in range(0,arr3d.shape[-2],ws2d)] for k in range(arr3d.shape[-3])]) 

class SPool:

    def __init__(self, input_shape, window_size):
        self.window_size = window_size
        assert((input_shape[-2]) % window_size == 0)
        assert((input_shape[-1]) % window_size == 0)
        self.output_shape = (input_shape[0],                \
                             input_shape[1] // window_size, \
                             input_shape[2] // window_size)

        print(f"creating SPool layer - out shape: {self.output_shape}")

        # to ensure that like in other layers every neuron can spike only once
        self.done_mask = numpy.ones(self.output_shape)

    def reset_done_mask(self):
        self.done_mask = numpy.ones(self.done_mask.shape)

    def forward(self, spiking_frame):
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

    def __call__(self, spiking_frame):
        return self.forward(spiking_frame)

# TODO

# that triple loop in forward is just cursed
