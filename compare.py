
import numpy as np
import os
import glob
import sys
import traceback
import Grimoire

from numba import jit

import logging as log

def readYoloLabelToPoint(lines, width=640, height=480):
	minutiae = []
	line_arg = 0
	for line in lines:		
		numberStrings = line.split(' ')
		widthFloat = float(numberStrings[1])*width
		heightFloat = float(numberStrings[2])*height
		point = [int(round(widthFloat)), int(round(heightFloat))]
		minutiae.append(point)

		line_arg += 1

	return minutiae


@jit(nopython=True)
def distancesPoints(pointsA, pointsB):
	"""
	make distance matrix between points
	O(n^2) every point with each other
	"""
	distances_shape = (len(pointsA), len(pointsB))
	distances = np.full(distances_shape, -1)
	for i_A in range(len(pointsA)):
		for j_B in range(len(pointsB)):
			# distances[i_A, j_B] = np.sqrt(
			#     (pointsA[i_A][0]-pointsB[j_B][0]) ** 2 + (pointsA[i_A][1]-pointsB[j_B][1]) ** 2)
			## distance between two points: pointsA[i_A], pointsB[j_B]
			distances[i_A, j_B] = np.linalg.norm(pointsA[i_A] - pointsB[j_B])
	return distances


@jit(nopython=True)
def matchPts(distances, dist_arg_sort, threshold):
	"""
	return truePos, falsePos, falseNeg, mse, mse_points
	"""
	truePos = 0
	# mean squared error of true points
	mse = 0
	mse_points = 0
	# already processed points mask
	maskA = np.ones(distances.shape[0], dtype=np.bool_)
	maskB = np.ones(distances.shape[1], dtype=np.bool_)
	mask = np.ones(len(dist_arg_sort), dtype=np.bool_)

	for i_arg_dist in range(len(dist_arg_sort)):
		# skip points already matched
		if(mask[i_arg_dist] == False):
			continue
		# get what points are closest to each other
		dist_smallest_arg = dist_arg_sort[i_arg_dist]
		# how much is that smallest distance?
		dist_smallest = distances[dist_smallest_arg[0], dist_smallest_arg[1]]

		## mask points just matched
		maskIndexes = np.where(
                    (dist_arg_sort[:, 0] == dist_smallest_arg[0]))
		mask[maskIndexes] = False

		maskIndexes = np.where(
                    (dist_arg_sort[:, 1] == dist_smallest_arg[1]))
		mask[maskIndexes] = False

		mask[i_arg_dist] = True
		## mask points just matched

		# distance over threshold
		if(dist_smallest > threshold):
			# print("--- Reached threshold after ", i_arg_dist, " points")
			# print("--- distance of ", dist_smallest, " pixels")
			continue
			# break

		maskA[dist_smallest_arg[0]] = False
		maskB[dist_smallest_arg[1]] = False

		truePos += 1

		mse += dist_smallest ** 2
		mse_points += 1

	falseNeg = maskA.sum()
	falsePos = maskB.sum()
	return truePos, falsePos, falseNeg, mse, mse_points


def minutia_match(pointsTypedA, pointsTypedB, threshold):
	"""
	----------
	pointsTypedA : true points [[], [], []]
	pointsTypedB : predicted points [[], [], []]
	threshold : maximum pixel euclidean distance to consider true positive
	-------
	returns [truePos, falsePos, falseNeg, mse]
	"""
	# found points
	truePos = 0
	# not real points
	falsePos = 0
	# missed points
	falseNeg = 0
	# mean squared error of true points
	mse = 0
	mse_points = 0

	# for each minutiae type
	for type_no in range(len(pointsTypedA)):
		# get each point list; e.g. points = [[y0,x0], [y1,x1], [y2,x2], [y3,x3]]
		pointsA = np.array(pointsTypedA[type_no]).astype(np.float32)
		pointsB = np.array(pointsTypedB[type_no]).astype(np.float32)
		if pointsB.size == 0:
			falseNeg += pointsA.shape[-2]
			continue

		distances = distancesPoints(pointsA, pointsB)
		# # print("distances")
		# # print(distances)
		# print("distances sort")
		# print(np.sort(distances.flatten()))
		# sys.stdout.flush()
		dist_arg_sort = np.dstack(np.unravel_index(
			np.argsort(distances.ravel()), distances.shape))[0]

		result = matchPts(distances, dist_arg_sort, threshold)
		truePos += result[0]
		falsePos += result[1]
		falseNeg += result[2]
		mse  += result[3]
		mse_points  += result[4]

		# points = min(pointsTypedA[type_no].size//2, pointsTypedB[type_no].size//2)

	if (mse_points == 0):
		mse_points = 1

	mse = mse/(mse_points*2)

	return [truePos, falsePos, falseNeg, mse]


def minutia_matchSlow(pointsTypedA, pointsTypedB, threshold):
	"""
	----------
	pointsTypedA : true points [[], [], []]
	pointsTypedB : predicted points [[], [], []]
	threshold : maximum pixel euclidean distance to consider true positive
	-------
	returns [truePos, falsePos, falseNeg, mse]
	"""
	# found points
	truePos = 0
	# not real points
	falsePos = 0
	# missed points
	falseNeg = 0
	# mean squared error of true points
	mse = 0
	mse_points = 0

	# for each minutiae type
	for type_no in range(len(pointsTypedA)):
		# get each point list; e.g. points = [[y0,x0], [y1,x1], [y2,x2], [y3,x3]]
		pointsA = np.array(pointsTypedA[type_no])
		pointsB = np.array(pointsTypedB[type_no])
		if pointsB.size == 0:
			falseNeg += pointsA.shape[-2]
			continue


		# make distance matrix between points
		# O(n^2) every point with each other
		distances_shape = (len(pointsA), len(pointsB))
		distances = np.full(distances_shape, -1)
		for i_A in range(len(pointsA)):
			for j_B in range(len(pointsB)):
				# distances[i_A, j_B] = np.sqrt(
				#     (pointsA[i_A][0]-pointsB[j_B][0]) ** 2 + (pointsA[i_A][1]-pointsB[j_B][1]) ** 2)

				# distance between two points: pointsA[i_A], pointsB[j_B]
				distances[i_A, j_B] = np.linalg.norm(pointsA[i_A] - pointsB[j_B])

		# # print("distances", distances)
		# print("distances sort", np.sort(distances.flatten()))
		# sys.stdout.flush()

		dist_arg_sort = np.dstack(np.unravel_index(
			np.argsort(distances.ravel()), distances_shape))[0]
		mask = np.ones(len(dist_arg_sort), dtype=bool)
		maskA = np.ones(pointsA.shape[-2], dtype=bool)
		maskB = np.ones(pointsB.shape[-2], dtype=bool)
		for i_arg_dist in range(len(dist_arg_sort)):
			# skip points already matched
			if(mask[i_arg_dist] == False):
				continue
			# get what points are closest to each other
			dist_smallest_arg = dist_arg_sort[i_arg_dist]
			# how much is that smallest distance?
			dist_smallest = distances[dist_smallest_arg[0], dist_smallest_arg[1]]

			## mask points just matched
			maskIndexes = np.where(
									 (dist_arg_sort[:, 0] == dist_smallest_arg[0]))
			mask[maskIndexes] = False

			maskIndexes = np.where(
									 (dist_arg_sort[:, 1] == dist_smallest_arg[1]))
			mask[maskIndexes] = False

			mask[i_arg_dist] = True
			## mask points just matched

			# distance over threshold
			if(dist_smallest > threshold):
				# print("--- Reached threshold after ", i_arg_dist, " points")
				# print("--- distance of ", dist_smallest, " pixels")
				continue
				# break

			maskA[dist_smallest_arg[0]] = False
			maskB[dist_smallest_arg[1]] = False

			truePos += 1

			mse += dist_smallest ** 2
			mse_points += 1

		falseNeg += maskA.sum()
		falsePos += maskB.sum()

		# points = min(pointsTypedA[type_no].size//2, pointsTypedB[type_no].size//2)

	if (mse_points == 0):
		mse_points = 1

	mse = mse/(mse_points*2)
	
	return [truePos, falsePos, falseNeg, mse]


def compareLabelsDir(pathDirTrue, pathDirPred, threshold=16):
	"""
	Compares predicted labels directory with another one
	-----
	pathDirPred : path to predicted labels directory
	pathDirTrue : path to true labels directory
	threshold : pixel distance threshold to consider point match
	-----
	return grimoire.stats(), mseList
	"""
	stats = Grimoire.stats()
	mseList = []

	for pathFilePred in glob.iglob('/'.join([pathDirPred, "*.txt"])):
		if '_' not in pathFilePred:
			continue

		pathFileTrue = '/'.join([pathDirTrue, os.path.basename(pathFilePred)])

		log.info('> comparing files:')

		try:
			with open(pathFilePred, 'r') as filePred:
				log.info('read: ' + pathFilePred)
				pointsPred = readYoloLabelToPoint(filePred)

			if len(pointsPred) == 0:
				log.info('No minutiae found in ' + pathFilePred)

			with open(pathFileTrue, 'r') as fileTrue:
				log.info('read: ' + pathFileTrue)
				pointsTrue = readYoloLabelToPoint(fileTrue)

			totalTruePos = len(pointsTrue)
			totalPredPos = len(pointsPred)
			result = minutia_match([pointsTrue], [pointsPred], threshold)
			truePos, falsePos, falseNeg, mse = result
			stats.append(truePos, falsePos, falseNeg)
			mseList.append(mse)

		except Exception as e:
			print('Exception while comparing: ', pathFilePred, ' with ', pathFileTrue)
			print(e, flush=True)
			traceback.print_exc()

	stats.end()
	return stats, mseList


if __name__ == "__main__":
	import argparse
	## Instantiate the parser
	parser = argparse.ArgumentParser(
            description='Compares two directories with yolo labels\n'
            'python fingeryolo/compare.py ./fingeryolo/v3-spp2-anchor.1-box-30/results-t0.25',
            formatter_class=argparse.RawTextHelpFormatter)

	parser.add_argument('pathDirPred',
                     help='Directory with predicted labels')
	parser.add_argument('pathDirTrue', nargs='?',
                     help='Directory with true labels')

	parser.add_argument('-v', action='store_true',
                     help='verbose')
	parser.add_argument('-t', type=int, default=16,
                     help='threshold, max distance in pixels')
	## Instantiate the parser

	## Parse arguments
	args = parser.parse_args()
	
	if args.verbose:
		log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
		log.info("Verbose output.")
	else:
		log.basicConfig(format="%(levelname)s: %(message)s")

	if not args.pathDirTrue:
		head = os.path.dirname(args.pathDirPred)
		head = os.path.dirname(head)
		args.pathDirTrue = '/'.join([head, 'labels-yolo-30'])
	
	result = compareLabelsDir(args.pathDirTrue, args.pathDirPred, args.t)
	missRateList, falsePosList, falseDiscoveryRateList, mseList = result
	for arg, res in enumerate(result):
		print(arg)
		print(np.array(res).mean())

	# Grimoire.missRate()


	exit()
	## Compare a true list of points with a prediction

	pointsTrue = np.array([[0, 0], [2, 2], [4, 4], [6, 6]])
	pointsPred = np.array([[0, 0], [2, 2], [90, 90], [91, 91], [92, 92]])

	threshold = 1
	result = minutia_match([pointsTrue], [pointsPred], threshold)
	truePos, falsePos, falseNeg, mse = result
	
	print("falseNeg, falsePos, truePos, mse")
	print(result)
	print(truePos+falseNeg, ' == ', len(pointsTrue))
	

	## Yolo label file to list of points

	filepath = './fingeryolo/labels-yolo-30/1_1.txt'
	points = []
	with open(filepath, "r") as file:
		points = readYoloLabelToPoint(file)

	print(points)

	print(os.path.split('./fingeryolo/labels-yolo-30'))



