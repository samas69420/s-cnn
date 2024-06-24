import numpy
import matplotlib.pyplot as plt
import sys

layer = sys.argv[1]
WEIGHTS_FILE = f"../weights/layer_{layer}.weights"
NUM_KERNELS_TO_LOAD = 30 if layer == "0" else 100

assert(NUM_KERNELS_TO_LOAD % 10 == 0)

plt.style.use('dark_background')
FACECOLOR = "#181135"

loaded_kernels = []
try:
    with open(WEIGHTS_FILE, "rb") as f:
        for _ in range(NUM_KERNELS_TO_LOAD):
            loaded_kernels.append(numpy.load(f))
except FileNotFoundError:
    print(f"can't find the weights file \"{WEIGHTS_FILE}\" :(")
    print("try running the training script first")
    quit()

fig1, axs1 = plt.subplots(NUM_KERNELS_TO_LOAD // 10,10)
fig1.set_facecolor(FACECOLOR)
fig1.suptitle("ON")

# plot ON kernels
for i in range(NUM_KERNELS_TO_LOAD):
    axs1[i//10][i%10].imshow(loaded_kernels[i][0])
    axs1[i//10][i%10].set_title(f"{i}")

fig2, axs2 = plt.subplots(NUM_KERNELS_TO_LOAD // 10,10)
fig2.set_facecolor(FACECOLOR)
fig2.suptitle("OFF")

# plot OFF kernels
for i in range(NUM_KERNELS_TO_LOAD):
    axs2[i//10][i%10].imshow(loaded_kernels[i][1])
    axs2[i//10][i%10].set_title(f"{i}")


avg_convergence = 0
num_w_total = NUM_KERNELS_TO_LOAD * numpy.prod(loaded_kernels[0].shape)
for kernel in loaded_kernels:
    avg_convergence += numpy.sum((kernel * (1-kernel)))
avg_convergence = avg_convergence / num_w_total 
print("avg convergence for loaded kernels: ",avg_convergence )

plt.show()
