import numpy
import matplotlib.pyplot as plt

NUM_OF_FEATURE_TO_SHOW = 100

plt.style.use('dark_background')
FACECOLOR = "#181135"

kernels0 = []
try:
    with open("../weights/layer_0.weights", "rb") as f:
        for _ in range(30):
            kernels0.append(numpy.load(f))
except:
    print("error reading weights/layer_0.weights")
    quit()

kernels2 = []
try:
    with open("../weights/layer_2.weights", "rb") as f:
        for _ in range(750):
            kernels2.append(numpy.load(f))
except:
    print("error reading weights/layer_2.weights")
    quit()

fp1 = []
for kernel in kernels2:
    feature = numpy.zeros((kernel.shape[0],2*kernel.shape[1],2*kernel.shape[2]))
    for k in range(kernel.shape[0]):
        for i in range(kernel.shape[1]):
            for j in range(kernel.shape[2]):
                feature[k,2*i,2*j] = kernel[k, i, j]
    fp1.append(feature)

print("reconstructing ON features")

reconstructed_featuresON = []
for pooling_feature in fp1:
    result = numpy.zeros((pooling_feature.shape[0],14,14)) # TODO calc all reverse convolution shapes, here are hardcoded
    for k in range(pooling_feature.shape[0]):
        for i in range(pooling_feature.shape[1]):
            for j in range(pooling_feature.shape[2]):
                if pooling_feature[k, i, j] != 0:
                    weight = pooling_feature[k, i, j]
                    kernel = weight*kernels0[k][0] # weighted ON kernel
                    for _i in range(kernel.shape[0]):
                        for _j in range(kernel.shape[1]):
                            result[k, i+_i, j+_j] +=kernel[_i,_j]
    reconstructed_featuresON.append(result.sum(0))

print("reconstructing OFF features")

reconstructed_featuresOFF = []
for pooling_feature in fp1:
    result = numpy.zeros((pooling_feature.shape[0],14,14)) # TODO calc all reverse convolution shapes, here are hardcoded
    for k in range(pooling_feature.shape[0]):
        for i in range(pooling_feature.shape[1]):
            for j in range(pooling_feature.shape[2]):
                if pooling_feature[k, i, j] != 0:
                    weight = pooling_feature[k, i, j]
                    kernel = kernels0[k][1] # OFF kernel
                    for _i in range(kernel.shape[0]):
                        for _j in range(kernel.shape[1]):
                            result[k, i+_i-2, j+_j-2] += weight*kernel[_i,_j]
    reconstructed_featuresOFF.append(result.sum(0))

fig1, axs1 = plt.subplots(NUM_OF_FEATURE_TO_SHOW // 10,10)
fig1.set_facecolor(FACECOLOR)
fig1.suptitle("ON")
for i in range(NUM_OF_FEATURE_TO_SHOW):
    axs1[i//10][i%10].imshow(reconstructed_featuresON[i])

fig2, axs2 = plt.subplots(NUM_OF_FEATURE_TO_SHOW // 10,10)
fig2.set_facecolor(FACECOLOR)
fig2.suptitle("OFF")
for i in range(NUM_OF_FEATURE_TO_SHOW):
    axs2[i//10][i%10].imshow(reconstructed_featuresOFF[i])

plt.show()
