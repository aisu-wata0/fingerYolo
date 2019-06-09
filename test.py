#! python3

import glob
import os
import shutil
import subprocess
import shlex

import sys
import errno

import argparse
import logging as log
import numpy as np

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


def test(pathsDic):
	print('threshold range: ', args.args.thresh_range)
	
	# try to get trained weights
	try:
		weight = pathsDic['finalweight']()
	except IndexError:
		print('--- weight files not found')
		return
		
	print("weight file:", weight)

	resultMean = []
	## print setup
	resultTags = ["missRate", "falsePos",
               "falseDiscoveryRate", "mse"]
	resultTagLength = len(max(resultTags, key=len))
	stringFormat = '{0:>%d}' % (resultTagLength+1)
	floatFormat = '{0:>%d.4}' % (resultTagLength+1)
	##
	# 4 is the number of variables compareLabelsDir returns
	for arg in range(4):
		resultMean.append([])

	for thresh in args.args.thresh_range:
		thresh = round(thresh, 4)
		pathDirPred = '{}-t{:0.3f}'.format(pathsDic['outDir'], thresh)
		# if prediction not made yet
		if not os.path.isdir(pathDirPred):
			# call yolo to make predictions
			# yolo command to test network and save labels to compare
			bashCommand = './darknet detector test {} {} {} -ext_output '.format(
				pathsDic['obj.data'], pathsDic['test.cfg'], weight)
			bashCommand = bashCommand + ' -dont_show ' + ' -save_labels '
			yolo(bashCommand, pathsDic['valid'], pathDirPred, thresh)

		print('python {} {} {}'.format('compare.py', pathDirPred, pathsDic['labelsTrue']), flush=True)
		# compare predictions with true labels
		if not args.args.dry:
			result = compare.compareLabelsDir(pathsDic['labelsTrue'], pathDirPred)
			# missRateList, falsePosList, falseDiscoveryRateList, mseList = result
			tmpResults = []
			for res_arg, res in enumerate(result):
				resultMean[res_arg].append(np.mean(res))
				tmpResults.append(f'{np.mean(res):0.4f}')
			## print for user
			lines = ['']*2
			for val, tag in zip(tmpResults, resultTags):
				lines[0] += stringFormat.format(tag)
				lines[1] += floatFormat.format(val)
			for line in lines:
				print(line)
				#
	## print for user
	for resultListType, tag in zip(resultMean, resultTags):
		print(stringFormat.format(tag))
		line = ''
		for val in resultListType:
			line += '{:<8.4f}'.format(val) + ' '
		print(line)
	##
	# save comparison stats to curve.txt 
	with open(pathsDic['curve'] + '.txt', 'w') as curveFile:
		for resultListType in resultMean:
			for thresh in args.args.thresh_range:
				thresh = round(thresh, 4)
				curveFile.write('{:0.4f} '.format(thresh))
			curveFile.write('\n')
			for val in resultListType:
				curveFile.write('{:0.10f} '.format(val))
			curveFile.write('\n')
	# plot stats curve
	import matplotlib.pyplot as plt
	# plot falseDiscoveryRate x missRate
	if not args.args.dry:
		Grimoire.pltCurve(resultMean[2], resultMean[0], "falseDiscoveryRate", "missRate")
	# save img
	pathCurvePic = pathsDic['curve'] + '.png'
	print("Saving curve to: ", pathCurvePic, flush=True)
	if not args.args.dry:
		plt.savefig(pathCurvePic)

