#! python3
import traceback

import cv2
import glob
import os
import logging as log
import json
import numpy as np

import Grimoire
import enhance
import fingerprint
import args


def minutiae2AfisDic(minutiaeList, angleList):
	"""
	From the import minutiae coordinate list and angle list outputs the afis dictionary template
	minutiaeList = [[x0,y0], [x1,y1] ... ]  ,  angleList = [a0, a1 ... ]
	Example afis template:
		{
			"width": 388,
			"height": 374,
			"minutiae": [
				{
					"x": 74,
					"y": 136,
					"direction": 1.9513027039072617,
					"type": "ending"
				},
	. . . skipped 228 lines
				{
					"x": 265,
					"y": 207,
					"direction": 1.6704649792860586,
					"type": "bifurcation"
				}
			]
		}
	"""
	afisDic = {}
	# actually useless
	afisDic["width"] = 0
	afisDic["height"] = 0
	# actually useless
	afisDic["minutiae"] = []
	for i in range(len(minutiaeList)):
		minutiae = {}
		minutiae["x"] = minutiaeList[i][0]
		minutiae["y"] = minutiaeList[i][1]
		minutiae["direction"] = angleList[i] % (2*np.pi)
		# set to any
		minutiae["type"] = "ending"
		afisDic["minutiae"].append(minutiae)
	
	return afisDic



def readYoloLabelToPoint(lines, imgShape=(640, 480)):
	minutiae = []
	width, height = imgShape
	for line_arg, line in enumerate(lines):
		numberStrings = line.split(' ')
		widthFloat = float(numberStrings[1])*width
		heightFloat = float(numberStrings[2])*height
		point = [int(round(widthFloat)), int(round(heightFloat))]
		minutiae.append(point)
	return minutiae


def yolo2centerCoord(box, shape):
	img_h, img_w = shape[0:2]
	x, y, width, height = box
	x, y = int((x)*img_w), int((y)*img_h)
	return x, y


def readYolo(boxes, imgShape):
	minutiaeList = []
	for box in boxes:
		minutiaeList = yolo2centerCoord(box, imgShape)
	return minutiaeList


def yoloFile2coord(pathLabel, imgShape):
	with open(pathLabel, 'r') as file:
		minutiaeList = readYoloLabelToPoint(file, imgShape)
	return minutiaeList


def yolo2afis(pathLabel, pathImg, imgShape, blk_sz=11):
	"""
	Compares predicted labels directory with another one
	-----
	pathDirPred : path to predicted labels directory
	pathDirTrue : path to true labels directory
	threshold : pixel distance threshold to consider point match
	-----
	return grimoire.stats(), mseList
	"""
	img = cv2.imread(pathImg, cv2.IMREAD_GRAYSCALE)
	if img is None:
		print("#### Error, couldn't open file ", pathImg)
		return [],[]
	orientationField = fingerprint.orientationField(img, blk_sz)
	minutiaeList = yoloFile2coord(pathLabel, imgShape)
	angleList = fingerprint.minutiaeOrientation(orientationField, blk_sz, minutiaeList)

	# imgDraw = fingerprint.draw_orientation_map(img, orientationField, blk_sz, thicc=1)
	# cv2.imshow(pathImg, imgDraw)
	# cv2.waitKey(0)

	return minutiaeList, angleList


if __name__ == "__main__":
	import argparse
	## Instantiate the parser
	parser = argparse.ArgumentParser(
				description='Converts a directory of predictions into afis template\n'
            'python ' + __file__ + ' ./fingeryolo/testsOpen/v3-spp2-anchor1-box-30/preds/pred-t0.600/',
				formatter_class=argparse.RawTextHelpFormatter)

	parser.add_argument('pathDir', type=str, nargs='*',
							help='path to the directory where the predictions are')

	parser.add_argument('-v', '--verbose', action='store_true',
							help='verbose')
	parser.add_argument('-d', '--dry', action='store_true',
                     help="dry run")
	parser.add_argument('--img_dir', required=False, default=Grimoire.getDirLocation(__file__) + "/" + "imgs/",
                     help="directory where the images are located")
	## Parse arguments
	args.args = parser.parse_args()
	
	if args.args.verbose:
		log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
		log.info("Verbose output.")
	else:
		log.basicConfig(format="%(levelname)s: %(message)s")


	imgShape = (640, 480)
	blk_sz = 11
	pathDirImgs = args.args.img_dir
	
	for pathDir in args.args.pathDir:
		log.info('convert directory to afis: ' + pathDir)
		pathOutDir = Grimoire.trailingSlashRm(pathDir) + '-afis'
		Grimoire.ensure_dir(pathOutDir)
		for pathLabel in glob.iglob('/'.join([pathDir, "*.txt"])):
			if '_' not in pathLabel:
				continue
			
			basename = os.path.basename(pathLabel)
			root, ext = os.path.splitext(basename)

			pathImg = '/'.join([pathDirImgs, root + '.png'])
			pathAfis = '/'.join([pathOutDir, root + '.json'])
			log.info("pathImg %s pathAfis %s", pathImg, pathAfis)
			try:
				minutiaeList, angleList = yolo2afis(pathLabel, pathImg, imgShape, blk_sz)
				dic = minutiae2AfisDic(minutiaeList, angleList)
				log.info(str(dic))
				if not args.args.dry:
					with open(pathAfis, 'w') as outFile:
						json.dump(dic, outFile)
				
			except Exception as e:
				print('Exception in yolo2afis ', pathLabel, ', ', pathImg, ', ', imgShape)
				print(e, flush=True)
				traceback.print_exc()
