import numpy

def slow_conv2d(array, kernel): 
# perform valid mode convolution on a 3darray with a 3dkernel 
# assuming stride 1 

    result = numpy.zeros((array.shape[-2:][0]-kernel.shape[-2:][0]+1,\
                          array.shape[-2:][1]-kernel.shape[-2:][1]+1))

    for i in range(0, array.shape[-2:][0]-kernel.shape[-2:][0]+1):
        for j in range(0, array.shape[-2:][1]-kernel.shape[-2:][1]+1):
            sliding_window = array[:,i: i+kernel.shape[-2:][0], j: j+kernel.shape[-2:][1]]
            result[i][j] = numpy.sum( sliding_window * kernel )

    return result

try:
    from fastconv import conv2d
except ModuleNotFoundError:
    print("can't find fastconv module, did you compile fastconv.c?")
    input("press enter to proceed with super slow python-only convolution implementation")
    conv2d = slow_conv2d

# 1-line utils
argmaxNd = lambda arrNd : numpy.unravel_index(arrNd.argmax(),arrNd.shape)

class SConv:
# class for a spiking convolution layer, if training is set to True
# stdp and stdp competition will be performed, otherwise only lateral inhibition

    def __init__(self, input_shape, num_kernels, kernel_shape,      \
                 threshold, wloc = 0.8, wscale = 0.05,              \
                 training = False, a_plus = 0.004 , a_minus = 0.003,
                 stdp_competition_window_size = 11 ):

        assert(len(input_shape) == 3)
        assert(input_shape[0] == kernel_shape[0])

        # automatically initialize members to arguments (self.layers = layers...)
        self.__dict__.update(locals()) 

        self.trainable = True

        # set conv layer output shape according to valid convolution mode 
        self.output_shape = (self.num_kernels,                                     \
                             self.input_shape[-2:][0]-self.kernel_shape[-2:][0]+1, \
                             self.input_shape[-2:][1]-self.kernel_shape[-2:][1]+1) # 30,23,23

        print(f"creating SConv layer - out shape: {self.output_shape}")

        # kernels initialization
        print("initializing random kernels")
        self.kernels = []
        for _ in range(self.num_kernels):
            self.kernels.append(numpy.random.normal(loc = wloc, scale = wscale, size = self.kernel_shape)) # gaussian init

        self.num_w_total = numpy.prod(self.kernel_shape) * self.num_kernels

        self.potentials = numpy.zeros(self.output_shape)
        self.lat_inhib_mask  = numpy.ones(self.output_shape)

        if training:
            self.set_training()

    def set_training(self):
        print("SConv setting training mode")
        self.training = True

        # logged_spikes(t) = all input spikes emitted up to timestep "t"
        self.logged_input_spikes = numpy.zeros(self.input_shape)

        self.stdp_comp1_mask = numpy.ones(self.output_shape)
        self.stdp_comp2_mask = numpy.ones(self.output_shape)

    def get_avg_w_convergence(self):
        avg_convergence = 0
        for kernel in self.kernels:
            avg_convergence += numpy.sum((kernel * (1-kernel)))
        avg_convergence = avg_convergence / self.num_w_total 
        return avg_convergence

    def load_weights_numpy(self, weights_file_path):
        print(f"loading weights from {weights_file_path}")
        try:
            with open(weights_file_path, "rb") as f:
                for i in range(self.num_kernels):
                   self.kernels[i] = numpy.load(f)
            print(f"loading done")
        except FileNotFoundError:
            print(f"can't find weights file \'{weights_file_path}\', continuing with random weights")

    def save_weights_numpy(self, weights_file_path):
        print(f"saving weights into {weights_file_path}")
        with open(weights_file_path, "wb") as f:
            for kernel in self.kernels:
                numpy.save(f, kernel)

    def reset_lat_inhib_mask(self):
        self.lat_inhib_mask = numpy.ones(self.lat_inhib_mask.shape)

    def reset_stdp_comp1_mask(self):
        self.stdp_comp1_mask = numpy.ones(self.stdp_comp1_mask.shape)

    def reset_stdp_comp2_mask(self):
        self.stdp_comp2_mask = numpy.ones(self.stdp_comp2_mask.shape)

    def reset_logged_spikes(self):
        self.logged_input_spikes = numpy.zeros(self.logged_input_spikes.shape)

    def reset_state(self):
        if self.training:
            self.reset_stdp_comp1_mask()
            self.reset_stdp_comp2_mask()
            self.reset_logged_spikes()
        self.reset_lat_inhib_mask()
        self.potentials = numpy.zeros(self.output_shape)

    def lateral_inhibition(self):
    # in the potentials array along the first dimesion (0) only one neuron 
    # can go above the threshold and spike, all the other neurons with the 
    # same (x,y) coordinates are inhibited

        max_indexes = self.potentials.argmax(0) # basically a matrix of spaghettos indexes
        max_values = self.potentials.max(0)
        for x,row in enumerate(max_values):
            for y,potential in enumerate(row):
                if potential > self.threshold:
                    self.lat_inhib_mask[:,x,y].fill(0)
                    self.lat_inhib_mask[max_indexes[x,y],x,y] = 1
        self.potentials *= self.lat_inhib_mask

    def w_2dindex_between_neurons(self, pre_coords, post_coords):
    # given the 2d coordinates of both pre and post synaptic neurons this function
    # computes the coordinates of the specific weight inside a convolution kernel
    # that connects the two neurons

        kernel_shape = self.kernel_shape[-2:]
        input_shape = self.input_shape[-2:]
        dirty_offset = input_shape[-1]*pre_coords[0]+pre_coords[1]-input_shape[-1]*post_coords[0]-post_coords[1]
        offset = dirty_offset-(input_shape[-1]-kernel_shape[-1])*(dirty_offset//input_shape[-1])
        return numpy.unravel_index(offset,kernel_shape)

    def w_3dindex_between_neurons(self, pre_coords3d, post_coords2d):
        coord_0 = pre_coords3d[0]
        pre_coords2d = pre_coords3d[-2:]
        w_coords2d = self.w_2dindex_between_neurons(pre_coords2d, post_coords2d)
        return (coord_0,*w_coords2d)
    
    def stdp_competition(self):
    # used only during training phase, its purpose is to select only
    # the most sensible neurons to any particular feature and let them update
    # the weights for all the neurons in the same feature map such that every
    # map responds to a unique and different feature

        # STDP competition stage 1 (only one spike max for each map)
        best_in_map_list = []
        for k in range(self.num_kernels):
            n_spikes_in_map = numpy.sum(self.calc_spikes()[k])
            if n_spikes_in_map >= 1:
                potentials_map = self.potentials[k]
                best_in_map_coords = argmaxNd(potentials_map) 
                assert(self.calc_spikes()[k][best_in_map_coords] == 1)
                self.stdp_comp1_mask[k].fill(0)
                self.stdp_comp1_mask[k][best_in_map_coords] = 1.0
                best_in_map_list.append((best_in_map_coords,potentials_map[best_in_map_coords]))

            elif n_spikes_in_map == 0: 
                best_in_map_list.append((None,0))

        self.potentials *= self.stdp_comp1_mask

        # STDP competition stage 2 (only one spike max in 11x11 spaghetto)
        working_pot = self.potentials.copy()
        valid_neurons_ind = []
        while working_pot.max() > self.threshold:
            pot_max_ind = argmaxNd(working_pot)
            valid_neurons_ind.append(pot_max_ind)
            self.stdp_comp2_mask[:,
                                 numpy.clip(pot_max_ind[1]-(self.stdp_competition_window_size//2),0,working_pot.shape[1]):numpy.clip(pot_max_ind[1]+(self.stdp_competition_window_size//2)+1,0,working_pot.shape[1]),
                                 numpy.clip(pot_max_ind[2]-(self.stdp_competition_window_size//2),0,working_pot.shape[2]):numpy.clip(pot_max_ind[2]+(self.stdp_competition_window_size//2)+1,0,working_pot.shape[2])
                                ] = 0
            working_pot *= self.stdp_comp2_mask

        for valid_neuron in valid_neurons_ind:
            self.stdp_comp2_mask[valid_neuron] = 1
    
        self.potentials *= self.stdp_comp2_mask

    def calc_spikes(self):
        threshold_mask = numpy.ma.masked_greater(self.potentials, self.threshold).mask
        spiking_frame = threshold_mask.astype(numpy.float64)             \
                        if threshold_mask.shape == self.potentials.shape \
                        else numpy.zeros(self.potentials.shape)

        # every neuron can spike only once during all the timesteps
        # to implement this mechanism and keep potentials at 0 
        # the lat_inhib_mask is reused to take in count all the neurons
        # that have already spiked
        self.lat_inhib_mask *= numpy.logical_not(spiking_frame == 1)

        return spiking_frame

    def stdp(self):
    # implementation of the simplified stdp learning logic (only sign matters)
    # it is triggered when a output neuron spikes and updates the weight of the
    # whole map the spiking neuron is part of

        spikes = self.calc_spikes() 
        spiking_maps_indexes = numpy.where(spikes.sum((1,2)))[0].tolist()
        
        for k in spiking_maps_indexes:
            spiking_neurons_indexes2d = numpy.stack(numpy.where(spikes[k] == 1)).T.tolist()

            target_kernel = self.kernels[k] # references not supported by python :(

            for neuron_indexes in spiking_neurons_indexes2d:
                input_spikes_window = self.logged_input_spikes[:, neuron_indexes[0]:neuron_indexes[0]+self.kernel_shape[1],neuron_indexes[1]:neuron_indexes[1]+self.kernel_shape[2]]

                # positive update 
                spiking_input_neurons_indexes = numpy.where(input_spikes_window == 1)
                target_kernel[spiking_input_neurons_indexes] += self.a_plus*target_kernel[spiking_input_neurons_indexes]*(1-target_kernel[spiking_input_neurons_indexes])

                # negative update
                non_spiking_input_neurons_indexes = numpy.where(input_spikes_window == 0)
                target_kernel[non_spiking_input_neurons_indexes] -= self.a_minus*target_kernel[non_spiking_input_neurons_indexes]*(1-target_kernel[non_spiking_input_neurons_indexes])

            self.kernels[k] = target_kernel
            
    def forward(self, spiking_frame):

        if self.training:
            self.logged_input_spikes += spiking_frame
            assert(self.logged_input_spikes.max() <= 1)

        # reset neurons that have spiked in previous timestep
        # without this operation next time that lateral inhibition is invoked
        # the potential of the last neurons that have spiked would still be
        # above the threshold and even higher than before so the same neurons
        # will be chosen again and again by lat. inhibition and they will keep
        # spiking for all the remaining timesteps
        self.potentials *= numpy.logical_not(self.calc_spikes())

        for k,kernel in enumerate(self.kernels): 
            self.potentials[k] += conv2d(spiking_frame, kernel)

        self.lateral_inhibition()
        if self.training:
            self.stdp_competition()
            self.stdp()

        return self.calc_spikes()

    def __call__(self, spiking_frame):
        return self.forward(spiking_frame)

# NOTES

# kernels weights initialized with normal distributions, firing threshold 
# set at 15 for the first layer
# neurons are not allowed to fire more than once, according to the paper 2 
# (https://arxiv.org/pdf/1611.01421.pdf)

# spikes can be logged sparsely or with a whole tensor shaped like the
# input, the tensor method is easy to use but scales like shit

# TODO

# remove slow loops from inhibition, competition and convolution and compile 
# them (cython, numba, C, etc), not done yet to have less dependencies
# check if a better alternative is possible for the references thing in the stdp
