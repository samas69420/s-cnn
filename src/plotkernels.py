import numpy
import matplotlib.pyplot as plt

WEIGHTS_FILE = "weightsfile"
NUM_KERNELS_TO_LOAD = 30

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

fig1, axs1 = plt.subplots(3,10)
fig1.set_facecolor(FACECOLOR)
fig1.suptitle("ON")

# plot ON kernels
for i in range(30):
    axs1[i//10][i%10].imshow(loaded_kernels[i][0])
    axs1[i//10][i%10].set_title(f"{i}")

fig2, axs2 = plt.subplots(3,10)
fig2.set_facecolor(FACECOLOR)
fig2.suptitle("OFF")

# plot OFF kernels
for i in range(30):
    axs2[i//10][i%10].imshow(loaded_kernels[i][1])
    axs2[i//10][i%10].set_title(f"{i}")

plt.show()
