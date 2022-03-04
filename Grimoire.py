#! py -3.7

import os
import errno
import urllib
import shutil
import glob
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker
import sklearn.metrics


def printValuesLines(values, tags, floatFormat='{0:>%d.4}'):
	lines = ['']*2
	tagMaxLength = len(max(tags, key=len))
	stringFormat = '{0:>%d}' % (tagMaxLength+1)
	floatFormat = floatFormat % (tagMaxLength+1)
	for val, tag in zip(values, tags):
		lines[0] += stringFormat.format(tag)
		lines[1] += floatFormat.format(val)
	for line in lines:
		print(line)


def timestamp():
	return time.strftime("%Y%m%d-%H%M")


def missRate(truePos, falseNeg):
	"""
	return falseNeg / conditionTrue
	"""
	return falseNeg / (falseNeg + truePos)


def precisionRecall(truePos, falsePos, falseNeg):
	"""
	return precision, recall, f1Score
	"""
	truePos = np.asarray(truePos)
	falsePos = np.asarray(falsePos)
	falseNeg = np.asarray(falseNeg)
	precision = (truePos) / (truePos + falsePos)
	recall = (truePos) / (truePos + falseNeg)
	f1Score = 2 * (precision*recall) / (precision+recall)
	if np.isnan(precision).any():
		# https://stats.stackexchange.com/questions/8025/what-are-correct-values-for-precision-and-recall-when-the-denominators-equal-0
		if np.ndim(precision) == 0:
			precision = 0.0
			recall = 0.0
			f1Score = 0.0
		else:
			nanIdxs = np.where(np.isnan(precision))
			precision[nanIdxs] = 0.0
			recall[nanIdxs] = 0.0
			f1Score[nanIdxs] = 0.0
	return precision, recall, f1Score

# def precisionRecall(truePos, falsePos, falseNeg):
# 	"""
# 	return precision, recall, f1Score
# 	"""
# 	precision = 0.0
# 	recall = 0.0
# 	f1Score = 0.0
# 	try:
# 		precision = (truePos) / (truePos + falsePos)
# 	except ZeroDivisionError:
# 		pass
# 	try:
# 		recall = (truePos) / (truePos + falseNeg)
# 	except ZeroDivisionError:
# 		pass
# 	try:
# 		f1Score = 2 * (precision*recall) / (precision+recall)
# 	except ZeroDivisionError:
# 		pass
# 	return precision, recall, f1Score


def pltCurveSetup(labelX, labelY, scale='log', limits=[0.01, 1.00, 0.01, 1.00], numTicks=11, pltStyle='dark_background'):
	"""
	Creates a figure and axis
	ex: Given false positive and false negative rates, produce a DET Curve.
	import matplotlib.pyplot as plt
	pltCurveSetup('recall', 'precision', scale='linear', limits=[0.00, 1.00, 0.00, 1.00])
	for statsCurve in statsCurveList:
		plt.plot(statsCurve.recallArray, statsCurve.precisionArray, '-|')
	plt.savefig('./directory/figureName.png')
	-----------
	labelX, labelY: labels for the axis
	scale: 'log' | 'linear' | search matplotlib scales
	limits: [xMin, xMax, yMin, yMax]
	numTicks: number of ticks in the plot
	pltStyle: 'dark_background' | 'default' | search for matplotlib style sheet 
	"""
	plt.style.use(pltStyle)
	fig, ax = plt.subplots()
	# ax.text(0.0, 0.1, "LinearLocator(numticks=3)", fontsize=14, transform=ax.transAxes)
	plt.xlabel(labelX)
	plt.ylabel(labelY)
	plt.grid(linestyle='--', which='major', alpha=0.5)
	plt.grid(linestyle=':', which='minor', alpha=0.25)
	ax.set_xscale(scale)
	ax.set_yscale(scale)
	ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda y, _: '{:0.2f}'.format(y)))
	ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda y, _: '{:0.2f}'.format(y)))
	# ticks X
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

	if ticks_to_useX is not None:
		ticks_to_useX = np.round(ticks_to_useX, 2)
		ticks_to_useY = np.round(ticks_to_useY, 2)
		ax.set_xticks(ticks_to_useX)
		ax.set_yticks(ticks_to_useY)
	# ax.tick_params(direction='out', length=6, width=2, colors='r', grid_color='r', grid_alpha=0.5)
	#
	plt.axis(limits)
	# plt.show()
	return


def pltCurve(valuesX, valuesY, labelX, labelY, scale='log', limits=[0.01, 1.00, 0.01, 1.00], numTicks=11, pltStyle='dark_background'):
	"""
	plots a curve, where 
	ex: Given false positive and false negative rates, produce a DET Curve.
	import matplotlib.pyplot as plt
	pltCurve(fpsList, fnsList, "false positive rate", "false negative rate")
	plt.show()
	plt.savefig(pathCurvePic)
	-----------
	valuesX, valuesY: list point values
	labelX, labelY: labels for the axis
	scale: 'log' | 'linear' | search matplotlib scales
	limits: [xMin, xMax, yMin, yMax]
	numTicks: number of ticks in the plot
	pltStyle: 'dark_background' | 'default' | search for matplotlib style sheet 
	"""
	pltCurveSetup(labelX, labelY, scale, limits, numTicks)

	plt.plot(valuesX, valuesY, '-.')
	# plt.show()
	return


def ensure_dir(pathDir):
    """
    ----------
    pathDir : path to directory you want to exist
    -------
    """
    if not pathDir:
        raise ValueError("Invalid file path")
    try:
        os.makedirs(pathDir, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return

	
def trailingSlashRm(pathDir):
	return pathDir.rstrip('\\').rstrip('/')


def getDirLocation(file):
	"""
	----------
	file : file you want to find the path to e.g. __file__
	-------
	return directory of the file "os.path.dirname(os.path.realpath(file))"
	"""
	dirpath = os.path.dirname(file)
	if dirpath == "":
		return "."
	return dirpath


def silentremove(pathFilename):
	"""
	Removes pathFilename, doesn't care if  it  exist
	-------
	returns True if removed something, False if it didn't exist
	"""
	try:
		os.remove(pathFilename)
	except OSError as e:  # this would be "except OSError, e:" before Python 2.6
		if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
			raise  # re-raise exception if a different error occurred
		else:
			return False
	return True


def downloadFile(url, pathFile):
	"""
	Download the file from 'url' and save it locally under 'pathFile':
	"""
	with urllib.request.urlopen(url) as response, open(pathFile, 'wb') as out_file:
		try:
			shutil.copyfileobj(response, out_file)
		except:
			silentremove(pathFile)
			raise
	return


def listDirectory(path_in, filetypeExt_in, sort=True):
	"""
	list file paths from a directory 'path_in' of file extension 'filetypeExt_in'
	----
	for pathFilename, file_arg in listDirectory(path_in, filetypeExt_in):
		# do something
	"""
	file_arg = 0
	files = glob.iglob(os.path.join(path_in, "*." + filetypeExt_in))
	if sort:
		files = sorted(files)
	for pathFilename in files:

		yield pathFilename, file_arg

		file_arg += 1


class stats(object):
	def __init__(self):
		# List of stats for each sample
		self.truePosList = []
		self.falsePosList = []
		self.falseNegList = []
		# Sum of the lists, set on self.end()
		self.truePos = None
		self.falsePos = None
		self.falseNeg = None


	def append(self, truePos, falsePos, falseNeg):
		self.truePosList.append(truePos)
		self.falsePosList.append(falsePos)
		self.falseNegList.append(falseNeg)

	def lists(self):
		return [self.truePosList, self.falsePosList, self.falseNegList]

	def values(self):
		return [self.truePos, self.falsePos, self.falseNeg]

	def end(self):
		"""
		Call after all appends and before acessing the values
		you *can* call this more than once to get partial values
		"""
		self.truePos = np.sum(self.truePosList)
		self.falsePos = np.sum(self.falsePosList)
		self.falseNeg = np.sum(self.falseNegList)

	def missRateFalseRate(self):
		"""
		Calculates and returns
		return self.missRate, self.falseDiscoveryRate
		"""
		self.missRate = self.falseNeg / (self.falseNeg + self.truePos)
		self.falseDiscoveryRate = self.falsePos/(self.falsePos + self.truePos)
		return self.missRate, self.falseDiscoveryRate

	def precisionRecall(self):
		"""
		Calculates and returns
		return self.precision, self.recall, self.f1Score
		"""
		result = precisionRecall(self.truePos, self.falsePos, self.falseNeg)
		self.precision, self.recall, self.f1Score = result
		if math.isnan(self.precision):
			print("> NAN      precision", self.truePos, self.falsePos, self.falseNeg)
		else:
			print(">", self.precision ," precision", self.truePos, self.falsePos, self.falseNeg)
		return result

	def printMissRateFalseRate(self, floatFormat='{0:>%d.4}'):
		statTags = ["missRate", "falseDiscoveryRate"]
		printValuesLines([self.missRate, self.falseDiscoveryRate], statTags, floatFormat)

	def printPrecisionRecall(self, floatFormat='{0:>%d.4}'):
		if not hasattr(self, 'precision'):
			self.precisionRecall()
		statTags = ["precision", "recall", "f1Score"]
		printValuesLines([self.precision, self.recall, self.f1Score], statTags, floatFormat)


class statsCurve(object):
	def __init__(self):
		self.truePosList = []
		self.falsePosList = []
		self.falseNegList = []
		self.thresholds = []

	def lists(self):
		return [self.truePosList, self.falsePosList, self.falseNegList]

	def append(self, stats, threshold):
		for statType, statTypeList in zip(stats.values(), self.lists()):
			statTypeList.append(statType)
		self.thresholds.append(threshold)
	
	def end(self):
		"""
		Call after all appends and before acessing the values
		you *can* call this more than once to get partial values
		"""
		self.truePosArray = np.array(self.truePosList)
		self.falsePosArray = np.array(self.falsePosList)
		self.falseNegArray = np.array(self.falseNegList)

	def precisionRecall(self):
		result = precisionRecall(self.truePosArray, self.falsePosArray, self.falseNegArray)
		self.precisionArray, self.recallArray, self.f1ScoreArray = result
		self.precisionRecallCurve = [self.precisionArray, self.recallArray]
		return self.precisionRecallCurve

	def printPrecisionRecall(self):
		if not hasattr(self, 'precisionRecallCurve'):
			self.precisionRecall()
		tags = ['precision', 'recall']
		for tag, val in zip(tags, self.precisionRecallCurve):
			print(tag)
			print(val)

	def plotPrecisionRecall(self):
		if not hasattr(self, 'precisionRecallCurve'):
			self.precisionRecall()
		
		pltCurve(self.recallArray, self.precisionArray, 'recall', 'precision', scale='linear')


	@staticmethod
	def plotCurves(statsCurveList, linestyle='-|', pltStyle='default'):
		plt.clf()
		pltCurveSetup('recall', 'precision', scale='linear',
		              limits=[0.00, 1.00, 0.00, 1.00], pltStyle=pltStyle)
		for statsCurve in statsCurveList:
			plt.plot(statsCurve.recallArray, statsCurve.precisionArray, linestyle)


	def metricAuc(self):
		auc = sklearn.metrics.auc(self.recallArray, self.precisionArray)
		return auc
			
