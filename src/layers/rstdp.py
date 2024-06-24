import numpy

class RSTDP:
# reward modulated stdp classification layer

    def __init__(self, input_shape, output_shape, wloc=0.8, wscale=0.02,
                 a_r_plus = 0.004, a_r_minus = 0.003,
                 a_p_plus = 0.0005, a_p_minus = 0.005,
                 n = 100, init_hit_ratio = 0.9, decay = 0,
                 training = False):

        # automatically initialize members to arguments (self.wloc = wloc...)
        self.__dict__.update(locals()) 

        self.trainable = True

        print(f"creating R-STDP layer - in shape: {self.input_shape} out shape: {self.output_shape}")

        print("R-STDP - initializing random weights")
        self.weights = numpy.random.normal(size=(input_shape, output_shape), loc=self.wloc, scale=self.wscale)

        self.num_w_total = numpy.prod(self.weights.shape)

        self.potentials = numpy.zeros(output_shape)

        if training:
            self.set_training()

    def get_avg_w_convergence(self):
        avg_convergence = 0
        avg_convergence += numpy.sum((self.weights * (1-self.weights)))
        avg_convergence = avg_convergence / self.num_w_total 
        return avg_convergence

    def set_training(self):
        print("R-STDP activating training mode")
        self.training = True
        self.logged_spikes = numpy.zeros(self.input_shape)
        self.n_hit_mem = numpy.zeros(self.n)
        self.n_miss_mem = numpy.zeros(self.n)
        self.memory_index = 0
        self.n_hit_mem[:int(self.init_hit_ratio*self.n)] = 1
        self.n_miss_mem[int(self.init_hit_ratio*self.n):] = 1
        print("R-STDP initial hit / miss ratios: ",self.n_hit_mem.sum()/self.n, " / ",self.n_miss_mem.sum()/self.n)

    def load_weights_numpy(self, weights_file_path):
        print(f"loading weights from {weights_file_path}")
        try:
            with open(weights_file_path, "rb") as f:
                self.weights = numpy.load(f)
            print(f"loading done")
        except FileNotFoundError:
            print(f"can't find weights file \'{weights_file_path}\', continuing with random weights")

    def save_weights_numpy(self, weights_file_path):
        print(f"saving weights into {weights_file_path}")
        with open(weights_file_path, "wb") as f:
            numpy.save(f, self.weights)

    def rstdp(self, desired_output):
        assert(self.training == True)

        self.memory_index += 1
        max_index = self.potentials.argmax()
        desired_index = desired_output.argmax()

        spiking_input_neurons_mask = self.logged_spikes
        non_spiking_input_neurons_mask = numpy.logical_not(spiking_input_neurons_mask)*1

        if max_index == desired_index:
            self.n_hit_mem[self.memory_index % self.n] = 1
            self.n_miss_mem[self.memory_index % self.n] = 0
            n_miss = self.n_miss_mem.sum()

            # reward
            # positive update (for neurons that have spiked)
            self.weights[:,max_index] += spiking_input_neurons_mask * (n_miss/self.n) * self.a_r_plus*(self.weights[:,max_index]*(1-self.weights[:,max_index]))
            # negative update (for neurons that have not spiked)
            self.weights[:,max_index] -= non_spiking_input_neurons_mask * (n_miss/self.n) * self.a_r_minus*(self.weights[:,max_index] * (1-self.weights[:,max_index]))

        else:
            self.n_hit_mem[self.memory_index % self.n] = 0
            self.n_miss_mem[self.memory_index % self.n] = 1
            n_hit = self.n_hit_mem.sum()

            # punishment
            # negative update (for neurons that have spiked)
            self.weights[:,max_index] -= spiking_input_neurons_mask * (n_hit/self.n) * self.a_p_plus*(self.weights[:,max_index]*(1-self.weights[:,max_index]))
            # positive update (for neurons that have not spiked)
            self.weights[:,max_index] += non_spiking_input_neurons_mask * (n_hit/self.n) * self.a_p_minus*(self.weights[:,max_index] * (1-self.weights[:,max_index]))

            self.weights[:,max_index] -= self.weights[:,max_index]*self.decay

    def forward(self, input_spikes):
        if self.training:
            self.logged_spikes = numpy.logical_or(self.logged_spikes,input_spikes)*1
        self.potentials += input_spikes.dot(self.weights)
        prediction = numpy.zeros(self.potentials.shape)
        prediction[self.potentials.argmax()] = 1
        return prediction

    def reset_potentials(self):
        self.potentials.fill(0)

    def reset_logged(self):
        self.logged_spikes.fill(0)

    def reset_state(self):
        self.reset_potentials()
        if self.training: self.reset_logged()

    def __call__(self, input_spikes):
        return self.forward(input_spikes)

# NOTE

# in this implementation there is also an exponential decay, is not used in the 
# papers and it is supposed to prevent all the weights of a output class from 
# quickly converge all to 1, set decay = 0 in constructor call to disable it

# this implementation doesnt have a threshold, the classification is done only
# with the max value among the potentials

# TODO

# add support to a threshold
