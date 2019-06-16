

import subprocess
import glob
import os
import shutil
import time
import traceback
import numpy as np

import Grimoire
import test
import args
import logging as log


def files(pathNetDir):
	pathsDic = {'dir': pathNetDir}
	if not os.path.isdir(pathNetDir):
		print("-- skipping, not a directory: ", pathNetDir, flush=True)
		return None
	print('Running dir ', pathNetDir, flush=True)
	# find out name of net
	pathsDic['name'] = os.path.basename(pathNetDir)
	if not pathsDic['name']:
		pathsDic['name'] = os.path.basename(os.path.dirname(pathNetDir))

	pathsDic['boxSz'] = pathsDic['name'].split('-')[-1]
	
	pathsDic['train.cfg'] = pathNetDir + '/' + pathsDic['name'] + '-train.cfg'
	pathsDic['test.cfg'] = pathNetDir + '/' + pathsDic['name'] + '.cfg'
	pathsDic['obj.data'] = pathNetDir + '/../' + 'obj.data'

	pathsDic['weights'] = pathNetDir + '/weights'
	pathsDic['chart'] = pathNetDir + '/' + 'chart-' + pathsDic['name'] + '.png'

	pathsDic['curve'] = pathNetDir + '/' + 'curve-' + pathsDic['name']
	pathsDic['outDir'] = pathNetDir + '/' + 'preds/pred'


	pathDirScript = os.path.dirname(os.path.relpath(__file__))
	if pathDirScript == "":
		pathDirScript = '.'
	pathsDic['labelsTrue'] = pathDirScript + \
		'/labels/labels-yolo-' + pathsDic['boxSz']

	pathsDic['imgs'] = pathDirScript + '/imgs'
	
	pathsDic['finalweight'] = lambda: sorted(glob.iglob(pathsDic['weights'] + '/*.weights'))[-1]

	## Parse obj.data
	if args.args.verbose:
		print('parsing file ', pathsDic['obj.data'], flush=True)
	
	with open(pathsDic['obj.data'], 'r') as objFile:
		for line in objFile:
			line = line.rstrip('\n')
			lineSplit = line.split('=')
			argName = lineSplit[0].rstrip(' ').strip(' ')
			pathsDic[argName] = lineSplit[1].rstrip(' ').strip(' ')

	if args.args.verbose:
		print(pathsDic, flush=True)
	return pathsDic


def train(pathsDic, lastBoxSz):
	print("ensure ", pathsDic['backup'], ' exists', flush=True)
	if not args.args.dry:
		Grimoire.ensure_dir(pathsDic['backup'])

	if lastBoxSz == 0 or lastBoxSz != pathsDic['boxSz']:
		print('copying label files: ', pathsDic['labelsTrue'], ' ', pathsDic['imgs'], flush=True)
		lastBoxSz = pathsDic['boxSz']

		for labelPath in glob.iglob(pathsDic['labelsTrue'] + "/*.txt"):
			# for each processed image path
			# find label path (change ext to txt), and copy it to output dir
			pathIn, basename = os.path.split(labelPath)
			labelPathOut = pathsDic['imgs'] + '/' + basename

			if(args.args.verbose):
				print('cp ', labelPath, ' ', labelPathOut, flush=True)

			if not args.args.dry:
				shutil.copy2(labelPath, labelPathOut)

	commandArgs = command.format(pathsDic['obj.data'], pathsDic['train.cfg'])
	print("$ ", commandArgs, flush=True)
	#
	if not args.args.dry:

		subprocess.call(commandArgs)

	if os.path.isdir(pathsDic['weights']):
		# a weights directory already exists
		# use a different name, timestamped
		timestr = time.strftime("%Y%m%d-%H%M")
		pathsDic['weights'] = pathsDic['weights'] + '+' + timestr

	if os.path.isfile(pathsDic['chart']):
		# a weights directory already exists
		# use a different name, timestamped
		timestr = time.strftime("%Y%m%d-%H%M")
		pathsDic['chart'], _ = os.path.splitext(pathsDic['chart'])
		pathsDic['chart'] = pathsDic['chart'] + '+' + timestr + '.png'

	try: 
		weightFilesNo = len(os.listdir(pathsDic['backup']))
	except FileNotFoundError:
		print("-- no weight directory")
	else:
		if weightFilesNo > 0:
			print("mv ", pathsDic['backup'], ' ', pathsDic['weights'], flush=True)
			if not args.args.dry:
				shutil.move(pathsDic['backup'], pathsDic['weights'])
	if os.path.isfile('./chart.png'):
		print("mv ", './chart.png', ' ', pathsDic['chart'], flush=True)
		if not args.args.dry:
			shutil.move('./chart.png', pathsDic['chart'])
	return lastBoxSz


if __name__ == "__main__":
	import argparse
	## Instantiate the parser
	parser = argparse.ArgumentParser(
            description='Trains a darknet configuration and saves weights and evaluates with a curve graph if -tr or -t is passed\n'
            'python ' + __file__ +
           	' ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30 -tr 0.1 0.91 0.1 --dry',
            formatter_class=argparse.RawTextHelpFormatter)

	parser.add_argument('pathNetDir', type=str, nargs='*',
                     help='path where test config file is located and weights will be saved, [path]/[path].cfg; the parent directory from [path] should have these: ["test.txt", "train.txt", "obj.names", "obj.data"] ; e.g. [path]/../obj.data is a valid file')

	parser.add_argument('-v', '--verbose', action='store_true',
	help='verbose')
	parser.add_argument('-d', '--dry', action='store_true',
                     help="dry run")
	parser.add_argument('-e', '--evaluate', action='store_true',
                     help="evaluate only, skip training")
	parser.add_argument('-nec', '--no_evaluate_cache', action='store_true',
                     help="don't use saved cached evaluate values")

	group = parser.add_mutually_exclusive_group(required=False)
	group.add_argument('-t', '--thresh', type=float,
                    help='single threshold to evaluate')
	group.add_argument('-tr', '--thresh_range', nargs=3, type=float,
                    help='evaluates with multiple threshold values, step specified',
                    metavar=('START', 'STOP', 'STEP'))
	## Parse arguments
	args.args = parser.parse_args()

	if args.args.verbose:
		log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
		log.info("Verbose output.")
	else:
		log.basicConfig(format="%(levelname)s: %(message)s")

	args.args.thresholds = []
	if args.args.thresh:
		args.args.thresh_range.append(args.args.thresh)
	elif args.args.thresh_range:
		print('threshold range: ', args.args.thresh_range)
		args.args.thresh_range = np.arange(*args.args.thresh_range)

	lastBoxSz = '0'

	command = './darknet.exe detector train "{}" "{}" -map'
	# list with the stat curve of the tests
	statsCurveList = []
	# list with the paths and config dictionary for each test
	pathsDicList = []
	for pathNetDir in args.args.pathNetDir:
		pathsDic = files(pathNetDir)
		if not pathsDic:
			continue
		# skip training if evalutaion argument switch
		if not args.args.evaluate:
			lastBoxSz = train(pathsDic, lastBoxSz)
		# evaluate if thresholds were specified
		if len(args.args.thresh_range) > 0:
			try:
				statsCurve = test.test(pathsDic)
				if statsCurve is None:
					continue
				statsCurveList.append(statsCurve)
				pathsDicList.append(pathsDic)
			except Exception as err:
				print(err)
				traceback.print_exc()

	Grimoire.statsCurve.plotCurves(statsCurveList)

	pathCurvePic = './fingeryolo/' + 'curve-precisionRecallCompare.png'
	print("Saving curve to: ", pathCurvePic, flush=True)
	if not args.args.dry:
		import matplotlib.pyplot as plt
		Grimoire.statsCurve.plotCurves(statsCurveList, '-.')
		labels = []
		for idx, statsCurve in enumerate(statsCurveList):
			labels.append(pathsDicList[idx]['name'])
		plt.legend(labels)
		# save img
		if os.path.isfile(pathCurvePic):
			os.remove(pathCurvePic)
		plt.savefig(pathCurvePic)

		for idx, statsCurve in enumerate(statsCurveList):
			print("config: ", pathsDicList[idx]['name'], end=" \t")
			print("auc", statsCurve.metricAuc())
