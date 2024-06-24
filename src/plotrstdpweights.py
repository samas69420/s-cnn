import numpy
import matplotlib.pyplot as plt
import sys

WEIGHTS_FILE = f"../weights/layer_4.weights"

plt.style.use('dark_background')
FACECOLOR = "#181135"

try:
    with open(WEIGHTS_FILE, "rb") as f:
        weights = numpy.load(f)
except FileNotFoundError:
    print(f"can't find the weights file \"{WEIGHTS_FILE}\" :(")
    print("try running the training script first")
    quit()


wimg = weights.reshape(500,-1).copy()
fig = plt.figure()
fig.set_facecolor(FACECOLOR)
ax = fig.add_subplot(1, 1, 1)
ax.imshow(wimg)

plt.show()
