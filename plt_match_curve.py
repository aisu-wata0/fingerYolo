#! python3
import argparse
import os
import glob
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import scikitplot as skplt
import sklearn
import matplotlib.pyplot as plt
import matplotlib.ticker

import args


# class ROCcurve(object):
# 	def __init__(self):
# 		# List of stats for each sample
# 		self.truePosList = []
# 		self.falsePosList = []
# 		self.falseNegList = []
# 		# Sum of the lists, set on self.end()
# 		self.truePos = None
# 		self.falsePos = None
# 		self.falseNeg = None


def create_figure(pltStyle='default', lineWidth=2, dpi=300, numTicks=9, scale='linear',limits = [0.0, 1.0,  0.0, 1.0]):
	plt.style.use(pltStyle)
	fig, ax = plt.subplots()
	fig.set_dpi(dpi)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic curve')
	plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.grid(linestyle='--', which='major', alpha=0.5)
	plt.grid(linestyle=':', which='minor', alpha=0.25)

	ticks_to_useX = None
	if scale == 'log':
		ticks_to_useX = np.logspace(
			np.log10(limits[0]), np.log10(limits[1]), num=numTicks, base=10)
		ticks_to_useY = np.logspace(
			np.log10(limits[2]), np.log10(limits[3]), num=numTicks, base=10)
		# remove minor ticks, they aren't in sync
		ax.set_xticks([], minor=True)
		ax.set_yticks([], minor=True)
		# couldn't make below work
		# ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs='all', numticks=3))
	elif scale == 'linear':
		ax.xaxis.set_major_locator(matplotlib.ticker.LinearLocator(numTicks))
		ax.xaxis.set_minor_locator(matplotlib.ticker.LinearLocator(numTicks*2-1))
		ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(numTicks))
		ax.yaxis.set_minor_locator(matplotlib.ticker.LinearLocator(numTicks*2-1))

	if ticks_to_useX is not None:
		ticks_to_useX = np.round(ticks_to_useX, 2)
		ticks_to_useY = np.round(ticks_to_useY, 2)
		ax.set_xticks(ticks_to_useX)
		ax.set_yticks(ticks_to_useY)


def plt_values(y_true, y_probas, label='', linestyle='-.', lineWidth=1):
	fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_probas)
	roc_auc = sklearn.metrics.auc(fpr, tpr)
	plt.plot(fpr, tpr, linestyle=linestyle,  lw=lineWidth,
			 label=label+' area=%0.3f' % roc_auc)


def add_scores(genuineScores, impostorScores, label=''):
	impostorLabels = np.full(impostorScores.shape, 0)
	genuineLabels = np.full(genuineScores.shape, 1)
	y_probas = np.concatenate([impostorScores, genuineScores])
	y_true = np.concatenate([impostorLabels, genuineLabels])
	plt_values(y_true, y_probas, label=label)


def add_files(genuineScoresPath, impostorScoresPath, label=''):
	if args.args.verbose:
		print("\t> open file: ", genuineScoresPath)
	genuineScores = np.loadtxt(open(genuineScoresPath, 'r'))
	if args.args.verbose:
		print("\t> open file: ", impostorScoresPath)
	impostorScores = np.loadtxt(open(impostorScoresPath, 'r'))
	
	add_scores(genuineScores, impostorScores, label=label)


if __name__ == "__main__":
	# Instantiate the parser
	parser = argparse.ArgumentParser(
		description='py plt_match_curve.py -g finger-matcher/genuine_scores.txt -i finger-matcher/impostor_scores.txt')
	parser.add_argument('-v', '--verbose', action='store_true',
						help='Verbose')
	parser.add_argument('-i', '--impostor',
						help='filepath with impostor scores')
	parser.add_argument('-g', '--genuine',
						help='filepath with genuine scores')

	args = parser.parse_args()

	create_figure()
	
	add_files(args.args.genuine, args.args.impostor)

	plt.legend(loc="lower right")
	plt.show()


