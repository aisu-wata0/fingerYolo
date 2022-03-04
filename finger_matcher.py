#! python3
import os
import logging as log
import traceback
import subprocess
import shutil
import numpy as np

import Grimoire
import args
import template


def scores_from_predict_dir(dirPath, dirSuffix='-afis', angleTech='map'):
	dirPath = dirPath.rstrip('/')
	# afis template directory
	dirAfisPath = dirPath + dirSuffix
	# score directory location
	head, dirname = os.path.split(dirPath)
	scoresDir = os.path.dirname(head) + '/scores/'
	Grimoire.ensure_dir(scoresDir)
	filepath = scoresDir + '/' + dirname
	# score filepaths
	genuinePath = filepath + "-" + "genuine_scores.txt"
	impostorPath = filepath + "-" + "impostor_scores.txt"
	if dirSuffix != '-afis':
		genuinePath = filepath + "-" + dirSuffix +"-" + "genuine_scores.txt"
		print("genuinePath",genuinePath )
		impostorPath = filepath + "-"+  dirSuffix + "-" + "impostor_scores.txt"

	
	if os.path.isfile(genuinePath) and os.path.isfile(impostorPath):
		return genuinePath, impostorPath
		
	if not os.path.isdir(dirAfisPath):
	# if True:
		# if not, transform to afis template
		command = 'py template.py "{}" --img_dir "{}"'
		commandArgs = command.format(dirPath, args.args.img_dir)
		print("$ ", commandArgs, flush=True)
		if not args.args.dry:
			# subprocess.call(commandArgs)
			try:
				template.main([dirPath], args.args.img_dir, angleTech, dirSuffix=dirSuffix)
			except Exception as e:
				print('Exception in commandArgs: ', commandArgs)
				print(e, flush=True)
				traceback.print_exc()
	## call matcher to calculate scores
	command = 'java -jar ' + Grimoire.getDirLocation(
            __file__) + '/' + 'finger-matcher/target/finger-matcher-1.0-SNAPSHOT-jar-with-dependencies.jar "{}"'
	# command = 'mvn exec:java -D exec.mainClass=grimoire.App -Dexec.args="{}"'
	commandArgs = command.format(dirAfisPath)
	print("$ ", commandArgs, flush=True)
	if not args.args.dry:
		scoresProcessCode = subprocess.call(commandArgs)

	## copy results to elsewhere

	def copy(pathIn, pathOut):
		print('cp ', pathIn, ' ', pathOut, flush=True)
		if not args.args.dry:
			shutil.copy2(pathIn, pathOut)

	# copy files
	if scoresProcessCode == 0:
		copy("genuine_scores.txt", genuinePath)
		copy("impostor_scores.txt", impostorPath)
	else:
		print("error on score creation, didn't copy results to ", filepath)

	return genuinePath, impostorPath


if __name__ == "__main__":
	import argparse
	## Instantiate the parser
	parser = argparse.ArgumentParser(
            description='Given a directory with minutiae predictions, makes genuine and impostor acceptance tests\n'
            'python ' + __file__ + ' ./testsOpen/v3-spp2-anchor1-box-30/preds/pred-t0.425 --dry\n'
        				'scores saved in ./testsOpen/v3-spp2-anchor1-box-30/scores/pred-t0.425-{genuine/impostor}_scores.txt',
            formatter_class=argparse.RawTextHelpFormatter)

	parser.add_argument('-v', '--verbose', action='store_true',
                     help='verbose')
	parser.add_argument('pathNetDir', type=str, nargs='*',
                     help='path where test config file is located and weights will be saved, [path]/[path].cfg; the parent directory from [path] should have these: ["test.txt", "train.txt", "obj.names", "obj.data"] ; e.g. [path]/../obj.data is a valid file')

	parser.add_argument('-d', '--dry', action='store_true',
                     help="dry run")

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

	if args.args.thresh:
		args.args.thresh_range.append(args.args.thresh)
	elif args.args.thresh_range:
		print('threshold range: ', args.args.thresh_range)
		args.args.thresh_range = np.arange(*args.args.thresh_range)

	# compile package
	# command = 'mvn package'
	# print("$ ", command, flush=True)
	# if not args.args.dry:
	# 	subprocess.call(command)

	for dirPath in args.args.pathNetDir:
		scores_from_predict_dir(dirPath)
