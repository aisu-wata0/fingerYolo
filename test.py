#! python3
import glob
import os
import shutil
import subprocess
import shlex
import pickle

import sys
import errno

import argparse
import logging as log
import numpy as np
import matplotlib.pyplot as plt

import compare
import Grimoire
import args


def yolo(bashCommand, validListPath, pathOut, thresh):
	bashArgThresh = ' -thresh {:0.5f} '.format(thresh)
	bashCommand = bashCommand + bashArgThresh
	pathOutfile = '/'.join([pathOut, 'results.txt'])
	print()
	print(bashCommand, flush=True)
	print()
	# halt execution if dry run
	# if args.args.dry:
	# 	return
	if(args.args.verbose):
		print('ensuring {} exists'.format(pathOut), flush=True)
	
	if not args.args.dry:
		Grimoire.ensure_dir(pathOut)
	else:
		pathOutfile = '/dev/null'
	# call yolo redirecting stdin and stdout to these files
	with open(validListPath, 'r') as fileIn, open(pathOutfile, 'w') as fileOut:
		if(args.args.verbose):
			print('Creating darknet "{}" with weights "{}"\n'.format(
				args.args.pathCfg, args.args.pathWeight), flush=True)
		# subprocess.call blocks execution until finished
		# shlex.split(bashCommand)
		if not args.args.dry:
			subprocess.call(bashCommand, stdin=fileIn, stdout=fileOut)
	# move new predicted labels to output directory
	# open file which lists images processed
	with open(validListPath, 'r') as fileIn:
		# for each processed image
		for pathImage in fileIn:
			# for each processed image path
			# find label path (change ext to txt), and copy it to output dir
			pathIn, basename = os.path.split(pathImage)
			title, _ = os.path.splitext(basename)
			pathLabelIn = '/'.join([pathIn, title + '.txt'])
			pathLabelOut = '/'.join([pathOut, title + '.txt'])

			if(args.args.verbose):
				print('cp ', pathLabelIn, ' ', pathLabelOut, flush=True)

			if not args.args.dry:
				shutil.copy2(pathLabelIn, pathLabelOut)

	print('saved predicted labels to {}'.format(pathOut), flush=True)
	print()
	return

def evaluateNet(pathsDic):
	# try to get trained weights
	try:
		weight = pathsDic['finalweight']()
	except IndexError:
		print('--- weight files not found')
		return

	print("weight file:", weight)

	## print setup
	resultTags = ["precision", "recall", "f1Score"]
	resultTagsLength = len(max(resultTags, key=len))
	stringFormat = '{0:>%d}' % (resultTagsLength+1)
	floatFormat = '{0:>%d.4}' % (resultTagsLength+1)
	##
	statsCurve = Grimoire.statsCurve()

	for thresh in args.args.thresh_range:
		thresh = round(thresh, 4)
		pathDirPred = '{}-t{:0.3f}'.format(pathsDic['outDir'], thresh)
		# if prediction not made yet, (if dir not exists or is empty)
		if not os.path.isdir(pathDirPred) or (len(list(glob.iglob('/'.join([pathDirPred, "*.txt"])))) < 1):
			# call yolo to make predictions
			# yolo command to test network and save labels to compare
			bashCommand = './darknet detector test {} {} {} -ext_output '.format(
				pathsDic['obj.data'], pathsDic['test.cfg'], weight)
			bashCommand = bashCommand + ' -dont_show ' + ' -save_labels '
			yolo(bashCommand, pathsDic['valid'], pathDirPred, thresh)

		print('python {} {} {}'.format('compare.py',
										pathDirPred, pathsDic['labelsTrue']), flush=True)
		# compare predictions with true labels
		if not args.args.dry:
			stats, mseList = compare.compareLabelsDir(pathsDic['labelsTrue'], pathDirPred)
			statsCurve.append(stats, thresh)
			stats.printPrecisionRecall()
			print("mse=", np.mean(mseList))

	statsCurve.end()
	## save curve obj to file
	pathObj = pathsDic['curve'] + '.obj'
	print("Saving statsCurve to: ", pathObj, flush=True)
	if not args.args.dry:
		with open(pathObj, 'wb') as curveFile:
			pickle.dump(statsCurve, curveFile)

	return statsCurve


def readCurveTxt(pathsDic):
	resultMean = []
	with open(pathsDic['curve'] + '.txt', 'r') as curveFile:
		line = curveFile.readline()
		line = line.rstrip()
		tresholdList = list(map(float, line.split(' ')))
		for lineNo, line in enumerate(curveFile):
			# result values
			line = line.rstrip()
			resultType = list(map(float, line.split(' ')))
			# print('append: ', resultType)
			resultMean.append(resultType)

	return resultMean, tresholdList


def test(pathsDic):
	if args.args.verbose:
		print('thresholds: ', args.args.thresh_range)

	if args.args.no_evaluate_cache or not os.path.isfile(pathsDic['curve'] + '.obj'):
		# use network and available weights to create curve
		statsCurve = evaluateNet(pathsDic)
	else:
		print("open: ", pathsDic['curve'] + '.obj', flush=True)
		with open(pathsDic['curve'] + '.obj', 'rb') as curveFile:
			statsCurve = pickle.load(curveFile)

	if statsCurve is None:
		return None


	if args.args.verbose:
		statsCurve.printPrecisionRecall()
	## plot stats curve
	# plot falseDiscoveryRate x missRate
	pathCurvePic = pathsDic['curve'] + '-precisionRecall.png'
	print("Saving curve to: ", pathCurvePic, flush=True)
	if not args.args.dry:
		statsCurve.plotPrecisionRecall()
		# save img
		if os.path.isfile(pathCurvePic):
			os.remove(pathCurvePic)
		plt.savefig(pathCurvePic)
	
	return statsCurve

