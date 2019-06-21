import numpy as np
import matplotlib.pyplot as plt


def mnt2Coord(filename):
	xList = []
	yList = []
	with open(filename, "r") as file:
		for line in file:
			line = line.rstrip()
			lineSplit = line.split(" ")
			x = int(lineSplit[0])
			y = int(lineSplit[1])
			xList.append(x)
			yList.append(y)

	return xList, yList
				
if __name__ == "__main__":

	im = np.invert(plt.imread("tmp/1_1.bmp"))
	implot = plt.imshow(im, cmap=plt.cm.binary, alpha=0.6)
	pntAlpha = 0.8
	x, y = mnt2Coord("tmp/fm3.txt")
	plt.scatter(x, y, marker="x", alpha=pntAlpha)
	x, y = mnt2Coord("tmp/tao.txt")
	plt.scatter(x, y, marker="+", alpha=pntAlpha)
	plt.show()
