import numpy
import matplotlib.pyplot as plt

"""
https://deepai.org/dataset/mnist
Pixels are organized row-wise. Pixel values are 0 to 255.
0 means background (white), 255 means foreground (black)
"""

DATASET_TIMESTEPS = 10 # in how many timesteps every image is processed

DATASET_THRESHOLD = 10 # spiking threshold for the DoG layer
                       # can also be 0
                       # 50 works bad, 10 gives results more similar to the ones in the paper

numpy.seterr(divide='ignore', invalid='ignore') # suppress warnings from t=1/conv
plt.style.use('dark_background')

def visual_print(data_element): 
# print visually an element from MNIST dataset (not the spiking version)

    img = data_element[0]
    label = data_element[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] == 0):
                print(" ", end="")
            else:
                print("@", end="")
        print()
    print("label: ",label)

def conv2d(array, kernel): 
# perform convolution assuming same input's dim for output (same mode 
# convolution) and stride 1 

    result = numpy.zeros(array.shape)

    pad_size = kernel.shape[0]//2
    padded = numpy.pad(array, pad_size) # pad with 0s for dimensions

    for i in range(0, padded.shape[0]-kernel.shape[0]+1):
        for j in range(0, padded.shape[1]-kernel.shape[1]+1):
            sliding_window = padded[i: i+kernel.shape[0], j: j+kernel.shape[1]]
            result[i][j] = numpy.sum( sliding_window * kernel )

    return result

def DoG_kernel(sigma1, sigma2 ,N = 7, M = 7): 
# creates and return a Difference of Gaussians kernel as described in the paper

    assert(N%2 != 0)
    assert(M%2 != 0)

    def gaussian(i,j, sigma):
        return (           \
                numpy.exp( \
                    - (numpy.power(i, 2) + numpy.power(j, 2)) / (2 * numpy.power(sigma, 2)) \
                ) \
                / (2 * numpy.pi * numpy.power(sigma, 2)) \
               )

    kernel = numpy.zeros([N,M])

    for i in range(N): 
        for j in range(M):
            _i = i - N//2
            _j = j - M//2
            kernel[i][j] = gaussian(_i, _j ,sigma1) - gaussian(_i, _j ,sigma2)

    return kernel

def process_image(np_img, threshold): 
# takes a 2d image and returns all spiking pixels with their spiking 
# times (not quantized), this isn't yet the final spiking image

    conv_on = conv2d(np_img, DoG_kernel(sigma1 = 1, sigma2 = 2))
    conv_off = conv2d(np_img, DoG_kernel(sigma1 = 2, sigma2 = 1))
    
    mask_on = numpy.ma.masked_greater(conv_on, threshold).mask
    spikes_on = mask_on.astype(numpy.float64)   \
               if mask_on.shape == np_img.shape \
               else numpy.zeros(np_img.shape)
                
    mask_off = numpy.ma.masked_greater(conv_off, threshold).mask 
    spikes_off = mask_off.astype(numpy.float64)   \
               if mask_off.shape ==  np_img.shape \
               else numpy.zeros(np_img.shape)

    timings_on = numpy.where(mask_on, 1 / conv_on, 0)
    timings_off = numpy.where(mask_off, 1 / conv_off, 0)

    on = numpy.stack((spikes_on,timings_on))
    off = numpy.stack((spikes_off,timings_off))
    result = numpy.stack((on,off)) 

    return result # shape: (on/off, spike/t, xvalue, yvalue)

def quantize_t(t,N):
# takes a 2d array with firing timings (0.123, 0.312 ... ) and returns 
# 2d array with timings as timesteps (0,1, ...), -1 means no spike

    timings = t.copy()
    sorted_indexes_2d = numpy.dstack(numpy.unravel_index(timings.ravel().argsort(),timings.shape))[0]
    invalid_indexes = numpy.dstack(numpy.where(timings == 0))[0]

    sorted_indexes_2d = sorted_indexes_2d.tolist()
    invalid_indexes = invalid_indexes.tolist()
    [sorted_indexes_2d.remove(ind) for ind in invalid_indexes]

    sorted_indexes_2d = numpy.array(sorted_indexes_2d)
    invalid_indexes = numpy.array(invalid_indexes)

    for i,element in enumerate(sorted_indexes_2d):
        timings[*element] = i // (sorted_indexes_2d.shape[0]/N)

    for element in invalid_indexes:
        timings[*element] = -1

    return timings

def q_timings_to_scatter(q_timings):
# takes a 2d array with quantized firing timings (1, 3 ... )
# and returns 2 arrays with every spiking neuron for each timestep

    scatterplot_x = []
    scatterplot_y = []
    for i in range(int(q_timings.max())+1):
        spiking_indexes = numpy.stack(numpy.where((q_timings == i)*1 == 1)).T
        scatterplot_x += [i]*spiking_indexes.shape[0]
        for element in spiking_indexes.tolist():
            scatterplot_y += [element[0]*q_timings.shape[-1]+element[1]]
    return (numpy.array(scatterplot_x),numpy.array(scatterplot_y))    

def tensorize_t(q_timings, N):
# takes a 2d array with quantized firing timings (1, 3 ... N-1)
# and returns a 3d array with N spiking frames, one for each timestep 

    spiking_frames = numpy.zeros((N, *q_timings.shape))
    for i in range(N):
        spiking_frames[i] = (q_timings == i)*1
    return spiking_frames

def spikes_to_spiking_image(processed_image, timesteps):
# takes an array containing spikes and conntinous timings from DoG convolutions
# and returns tensor with spikes for each timestep, this will be the full
# final spiking image

    timings_on = processed_image[0][1]
    timings_off = processed_image[1][1]
    q_timings_on = quantize_t(timings_on, timesteps)
    q_timings_off = quantize_t(timings_off, timesteps)
    tensorized_on = tensorize_t(q_timings_on, timesteps)   # shape = (timesteps,27,27)
    tensorized_off = tensorize_t(q_timings_off, timesteps) # shape = (timesteps,27,27)
    spiking_image = numpy.moveaxis(numpy.stack((tensorized_on, tensorized_off)), 0, 1)  # to have 10 x 2 x H x W, with only stack would be 2 x 10 x H x W
    return spiking_image

class Dataset():
# Dataset class is only for normal MNIST dataset, each image will be turned
# into a spiking image in the SpikingDataset class

    def __init__(self, filenames):
        try:
            self.f_train = open(filenames[0], mode='rb')
            self.f_label = open(filenames[1], mode='rb')
        except:
            print("cant open files :C")
            quit()

        self.f_train.read(4) # magic number
        self.f_label.read(4) # magic number

        self.num_of_imgs = int.from_bytes(self.f_train.read(4), "big")
        self.num_of_rows = int.from_bytes(self.f_train.read(4), "big")
        self.num_of_cols = int.from_bytes(self.f_train.read(4), "big")

        self.offset_train = 16
        self.offset_label = 8

        print(f"dataset created: {self.num_of_imgs} {self.num_of_rows}x{self.num_of_cols} images found")

    def __getitem__(self,index):
        if (index < self.num_of_imgs):
            self.f_train.seek(self.offset_train + index * self.num_of_rows * self.num_of_cols)
            self.f_label.seek(self.offset_label + index)

            label_ind = int.from_bytes(self.f_label.read(1), "big")
            label = numpy.zeros((10))
            label[label_ind] = 1

            img = []
            for _ in range(self.num_of_rows):
                row = []
                for __ in range(self.num_of_cols):
                    pixel = int.from_bytes(self.f_train.read(1), "big") 
                    row.append(pixel)
                img.append(row)

            img = numpy.array(img)
            return (img,label)
            
        else:
            raise ValueError("index error")

    def get_generator(self):
        for i in range(self.num_of_imgs):
            yield self[i]
        
    def __len__(self):
        return self.num_of_imgs

class SpikingDataset(Dataset):
# get an image from the MNIST dataset and turns it into a spiking image

    def __init__(self,dataset, threshold, timesteps):
        self.dataset = dataset
        self.threshold = threshold
        self.timesteps = timesteps 
        self.data_shape = (timesteps, 2, self.dataset.num_of_rows - 1, self.dataset.num_of_cols - 1)
    def __getitem__(self,index):
        (img,label) = self.dataset[index]
        img = img[1:,1:] # slicing to have 27x27 pics as in the paper 
        processed_image = process_image(img, self.threshold)
        return (spikes_to_spiking_image(processed_image, self.timesteps),label)
    
train_dataset = Dataset(("../dataset/train-images.idx3-ubyte","../dataset/train-labels.idx1-ubyte"))
spiking_dataset = SpikingDataset(train_dataset, threshold = DATASET_THRESHOLD, timesteps = DATASET_TIMESTEPS)

#data_generator = dataset.get_generator()
#data = next(data_generator)

if __name__ == "__main__":
# perform and show various transformations to obtain a spiking image
# from a regular greyscaled image like is done in the paper

    index = 42069                                                          # i = 213 looks the same "2" pic from paper

    spiking_data = spiking_dataset[index]                                  # content: (spiking_image              ,label)
                                                                           # shape:   ((DATASET_TIMESTEPS,2,27,27), (10))
    spiking_image = spiking_data[0] 

    data = train_dataset[index]
    visual_print(data)
    image = data[0]                                                        # shape: (28,28)
    image = image[1:,1:]                                                   # shape: (27,27)

    conv_on  = conv2d(image, DoG_kernel(sigma1 = 1, sigma2 = 2))           # shape: (27,27)
    conv_off = conv2d(image, DoG_kernel(sigma1 = 2, sigma2 = 1))           # shape: (27,27)

    processed_image = process_image(image, spiking_dataset.threshold)      # shape = (2, 2, 27, 27)

    timings_on    = processed_image[0][1]
    q_timings_on  = quantize_t(timings_on, spiking_dataset.timesteps)
    (xON,yON)     = q_timings_to_scatter(q_timings_on)

    timings_off   = processed_image[1][1]
    q_timings_off = quantize_t(timings_off, spiking_dataset.timesteps)
    (xOFF,yOFF)   = q_timings_to_scatter(q_timings_off)

    tensorized_on  = tensorize_t(q_timings_on, spiking_dataset.timesteps)  # shape = (DATASET_TIMESTEPS,27,27)
    tensorized_off = tensorize_t(q_timings_off, spiking_dataset.timesteps) # shape = (DATASET_TIMESTEPS,27,27)

    FACECOLOR = "#181135"
    fig1, axs = plt.subplots(2,3)
    fig1.set_facecolor(FACECOLOR)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].imshow(image, cmap ='gray')
    axs[0, 1].set_title("ON DoG")
    axs[0, 1].imshow(conv_on, cmap ='gray')
    axs[0, 2].set_title("ON tau")
    axs[0, 2].set_facecolor(FACECOLOR)
    axs[0, 2].scatter(xON,yON)
    axs[1, 0].set_title("Original Image")
    axs[1, 0].imshow(image, cmap ='gray')
    axs[1, 1].set_title("OFF DoG")
    axs[1, 1].imshow(conv_off, cmap ='gray')
    axs[1, 2].set_title("OFF tau")
    axs[1, 2].set_facecolor(FACECOLOR)
    axs[1, 2].scatter(xOFF,yOFF)

    # show all input spikes for each timestep
    fig2, axs = plt.subplots(2, spiking_dataset.timesteps)
    fig2.suptitle("spiking image")
    fig2.set_facecolor(FACECOLOR)
    for i in range(spiking_dataset.timesteps):
        axs[0,i].imshow(spiking_image[:,0,:,:][i]) # on
        axs[0,i].set_title(f"ON - t={i}")
        axs[1,i].imshow(spiking_image[:,1,:,:][i]) # off
        axs[1,i].set_title(f"OFF - t={i}")

    plt.show()


# TODO

# while processing each image maybe only spiking times are needed before
# tensorization and one dimension can be removed
