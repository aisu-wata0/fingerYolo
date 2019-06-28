#! python3
import os
import logging as log
import subprocess
import shutil
from matplotlib import pyplot as plt
import args

import Grimoire
import finger_matcher
import plt_match_curve

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
	## Parse arguments
	args.args = parser.parse_args()

	if args.args.verbose:
		log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
		log.info("Verbose output.")
	else:
		log.basicConfig(format="%(levelname)s: %(message)s")

	plt_match_curve.create_figure()

	for dirPath in args.args.pathNetDir:
		dirPath = Grimoire.trailingSlashRm(dirPath)
		label = os.path.basename(dirPath).split("-")[1][3:]
		threshold = int(label)
		# if threshold % 100 == 0:
		if threshold < 300:
			print(threshold)
			genuineScoresPath, impostorScoresPath = finger_matcher.scores_from_predict_dir(dirPath)
			plt_match_curve.add_files(genuineScoresPath, impostorScoresPath, label)

	plt.legend(loc="lower right")
	plt.savefig("ROC_curve" + "-" + Grimoire.timestamp() + ".png")
