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


def create_figure(pltStyle='default', lineWidth=2, dpi=300):
	plt.style.use(pltStyle)
	plt.figure(dpi=dpi)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic curve')
	plt.plot([0, 1], [0, 1], color='navy', lw=lineWidth, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])


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


