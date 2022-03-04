#! python3
from PIL import Image
import numpy as np
from scipy import ndimage, misc, signal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm
import random
import colorsys
import cv2
import math
import sys
import time
import enhance
import traceback

from numba import jit


def draw_orientation_map(img, angles, blk_sz, thicc=1):
	'''
		img - Image array.
		angles - orientation angle at (x,y) pixel.
		blk_sz - block size of the orientation field.
		thicc - Thickness of the lines to plot.
	'''
	x_center = y_center = float(blk_sz)/2.0  # y for rowq and x for columns
	blk_no_y, blk_no_x = angles.shape
	# add color channels to ndarray
	img_draw = np.stack((img,)*3, axis=-1)
	for i in range(0, blk_no_y):
		for j in range(0, blk_no_x):
			# point is (x, y) # y top of the image is maximum y, in the loop is 0, therefore (blk_no_y-1-i)
			y, x = ((i)*blk_sz + y_center,
					  (j)*blk_sz + x_center)
			angle = angles[i][j]
			length = blk_sz  # total length of the line
			# down, left # add on y, sub on x
			starty = int(y + np.sin(angle)*length/2)
			startx = int(x - np.cos(angle)*length/2)
			# find the end point
			# up, right # sub on y, add on x
			endy = int(y - np.sin(angle)*length/2)
			endx = int(x + np.cos(angle)*length/2)
			cv2.line(img_draw, (startx, starty), (endx, endy), (255, 0, 0), thicc)

	return img_draw


def draw_singular_points_verbose(image, poincare, roi_blks, blk_sz, tolerance=2, thicc=2):
	'''
		image - Image array.
		poincare - Poincare index matrix of each block.
		blk_sz - block size of the orientation field.
		tolerance - Angle tolerance in degrees.
		thicc - Thickness of the lines to plot.
	'''
	# add color channels to ndarray
	if(len(image.shape) == 2):
		image_color = np.stack((image,)*3, axis=-1)
	else:
		image_color = image
	for i in range(1, image.shape[0]//blk_sz - 1):
		for j in range(1, image.shape[1]//blk_sz - 1):
				# if adjacent to blocks not in roi, ignore
				if(np.any(roi_blks[(i)*blk_sz, (j-1)*blk_sz] == 1) or np.any(roi_blks[(i)*blk_sz, (j+1)*blk_sz] == 1) or np.any(roi_blks[(i-1)*blk_sz, (j)*blk_sz] == 1) or np.any(roi_blks[(i+1)*blk_sz, (j)*blk_sz] == 1)):
					continue

				angle_core = 180
				color = matplotlib.cm.hot(abs(poincare[i, j]/360))
				color = (color[0]*255, color[1]*255, color[2]*255)
				if (np.isclose(poincare[i, j], angle_core, 0, tolerance)):
					cv2.circle(image_color, (int((j+0.5)*blk_sz), int((i+0.5) * blk_sz)),
								  int(blk_sz/2), color, thicc)

				angle_delta = -180
				if (np.isclose(poincare[i, j], angle_delta, 0, tolerance)):
					cv2.rectangle(image_color, (j*blk_sz, i*blk_sz),
									  ((j+1)*blk_sz, (i+1)*blk_sz), (255, 125, 0), thicc)

				whorl = 360
				if (np.isclose(poincare[i, j], whorl, 0, tolerance)):
					cv2.circle(image_color, (int((j+0.5)*blk_sz), int((i+0.5) * blk_sz)),
								  int(blk_sz/2), (0, 200, 200), thicc)
	return image_color


def draw_singular_points(image, singular_pts, poincare, blk_sz, thicc=2):
	'''
		image - Image array.
		poincare - Poincare index matrix of each block.
		blk_sz - block size of the orientation field.
		tolerance - Angle tolerance in degrees.
		thicc - Thickness of the lines to plot.
	'''
	# add color channels to ndarray
	if(len(image.shape) == 2):
		image_color = np.stack((image,)*3, axis=-1)
	else:
		image_color = image

	# cores, deltas, whorls = singular_pts/blk_sz
	divide_tuple_by_blk_sz = lambda point_blk: (point_blk[0]/blk_sz, point_blk[1]/blk_sz)
	cores, deltas, whorls = [map(divide_tuple_by_blk_sz, singular_pts_typed)
									 for singular_pts_typed in singular_pts]

	for i, j in map(lambda x: (int(x[0]), int(x[1])), cores):
		color = matplotlib.cm.hot(abs(poincare[i, j]/360))
		color = (color[0]*255, color[1]*255, color[2]*255)
		cv2.circle(image_color, (int((j+0.5)*blk_sz), int((i+0.5) * blk_sz)),
						int(blk_sz/2), color, thicc)

	for i, j in map(lambda x: (int(x[0]), int(x[1])), deltas):
		cv2.rectangle(image_color, (j*blk_sz, i*blk_sz),
						  ((j+1)*blk_sz, (i+1)*blk_sz), (0, 125, 255), thicc)

	for i, j in map(lambda x: (int(x[0]), int(x[1])), whorls):
		cv2.circle(image_color, (int((j+0.5)*blk_sz), int((i+0.5) * blk_sz)),
						int(blk_sz/2), (0, 200, 200), thicc)
	return image_color


def gradient(img, blk_sz):
	# dx = sobel_filter(img, 0)
	# dy = sobel_filter(img, 1)
	dy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
	dx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
	return gradientAux(img, blk_sz, dx, dy)

@jit(nopython=True)
def gradientAux(img, blk_sz, dx, dy):
	img_alpha_x = dx*dx - dy*dy
	img_alpha_y = 2 * dx * dy

	img_alpha_x_block = [[np.sum(img_alpha_x[idx_y: idx_y + blk_sz, idx_x: idx_x + blk_sz]) / blk_sz**2
								 for idx_x in range(0, img.shape[0], blk_sz)]
								for idx_y in range(0, img.shape[1], blk_sz)]

	img_alpha_y_block = [[np.sum(img_alpha_y[idx_y: idx_y + blk_sz, idx_x: idx_x + blk_sz]) / blk_sz**2
								 for idx_x in range(0, img.shape[0], blk_sz)]
								for idx_y in range(0, img.shape[1], blk_sz)]

	img_alpha_x_block = np.array(img_alpha_x_block)
	img_alpha_y_block = np.array(img_alpha_y_block)

	orientation_blocks = np.arctan2(img_alpha_y_block, img_alpha_x_block) / 2

	return orientation_blocks



def sobel_filter(img, axis):
	img = img.astype(np.float)

	if axis == 0:
		# x axis
		# wikipedia
		# kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
		# scipy/cv2
		kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float)
	elif axis == 1:
		# y axis
		# wikipedia
		# kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float)
		# scipy/cv2
		kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)

	convolved = signal.convolve2d(
		 img, kernel, mode='same', boundary='wrap', fillvalue=0)

	return convolved

def smooth_gradient(orientation_blocks, blk_sz):
	orientation_blocks_smooth = np.zeros(orientation_blocks.shape)
	blk_no_y, blk_no_x = orientation_blocks.shape
	# Consistency level, filter of size (2*cons_lvl + 1) x (2*cons_lvl + 1)
	cons_lvl = 1
	for i in range(cons_lvl, blk_no_y-cons_lvl):
		for j in range(cons_lvl, blk_no_x-cons_lvl):
			area_sin = area_cos = orientation_blocks[i-cons_lvl: i +
																  cons_lvl, j-cons_lvl: j+cons_lvl]

			mean_angle_sin = np.sum(np.sin(2*area_sin))
			mean_angle_cos = np.sum(np.cos(2*area_cos))
			mean_angle = np.arctan2(mean_angle_sin, mean_angle_cos) / 2
			orientation_blocks_smooth[i, j] = mean_angle

	return orientation_blocks_smooth


def reduce_points(points):
	i = 0
	length = len(points)
	next_points = []
	while i < length:
		next_points = []
		reduces = []
		point = points[i]
		j = 1
		while (i+j) < len(points):
			point_o = points[i+j]
			# vertical down vertical middle
			if((point[1] == point_o[1] and (point[0] + 1 == point_o[0]))
				# vertical middle horizontal right
				or (point[0] == point_o[0] and (point[1] + 1 == point_o[1]))
				# down left
				or (point[0] + 1 == point_o[0] and (point[1] - 1 == point_o[1]))
				# diagonal down right
				or (point[0]+1 == point_o[0] and point[1]+1 == point_o[1])):

				reduces.append(point_o)
			else:
				next_points.append(point_o)

			j += 1

		if(reduces):
			length = length - len(reduces)
			y, x = point
			for point in reduces:
				y += point[0]
				x += point[1]
			reduced_point = float(
				 y)/(len(reduces) + 1), float(x)/(len(reduces) + 1)

			new_points = points[0:i]
			new_points.append(reduced_point)
			if(next_points):
				new_points = new_points + next_points[:]
			points = new_points

		i += 1

	return points

# tolerance in blocks
def points_position(cores, deltas, tolerance=1):
	if(not deltas or not cores):
		return 'middle'

	delta = deltas[0]
	core = cores[0]
	if(np.isclose(delta[1], core[1], 0, tolerance)):
		return 'middle'

	if(delta[1] < core[1]):
		return 'left'
	else:
		return 'right'

def singular_type_classify(singular_pts):
	cores, deltas, whorls = singular_pts

	# if there is a delta and more than one core
	# get middlemost core to decide if left or right loop

	# if(len(cores) > 1 and len(deltas) == 1):
	# 	# get the middlemost core
	# 	middle_point = np.array([300/(11*2), 300/(11*2)])
	# 	distances = []
	# 	for j_core in range(len(cores)):
	# 		core = cores[j_core]

	# 		distances.append(np.linalg.norm(middle_point - np.array(core)))
	# 	ordered_arg_dist = np.argsort(distances)
		
	# 	cores = [cores[ordered_arg_dist[0]]]

	if(whorls or len(cores) == 2 or len(deltas) == 2):
		return 'whorl'
	elif (len(deltas) <= 1 and len(cores) <= 1):
		position = points_position(cores, deltas)
		if(position == 'middle'):
			return 'arch'

		if(position == 'right'):
			return 'left_loop'

		if(position == 'left'):
			return 'right_loop'
	else:
			return 'other'

def singular_pts(image, orientation_blocks, roi_blks, blk_sz, tolerance=2):
	poincare = np.zeros(orientation_blocks.shape)
	blk_no_y, blk_no_x = orientation_blocks.shape

	cores = []
	deltas = []
	whorls = []
	singular_type = None
	for i in range(1, blk_no_y-1):
		for j in range(1, blk_no_x-1):
			# if adjacent to blocks not in roi, ignore
			if(np.any(roi_blks[(i)*blk_sz, (j-1)*blk_sz] == 1) or np.any(roi_blks[(i)*blk_sz, (j+1)*blk_sz] == 1) or np.any(roi_blks[(i-1)*blk_sz, (j)*blk_sz] == 1) or np.any(roi_blks[(i+1)*blk_sz, (j)*blk_sz] == 1)):
				continue

			index = 0
			cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1),
						(1, 0), (1, -1), (0, -1), (-1, -1)]

			def get_angle(left, right):
				angle = left - right
				if abs(angle) > 180:
					angle = -1 * np.sign(angle) * (360 - abs(angle))
				return angle

			deg_angles = [np.degrees(
				 orientation_blocks[i - k][j - l]) % 180 for k, l in cells]
			# len(cells)-1 == 8
			for k in range(len(cells)-1):
				if abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
						deg_angles[k + 1] += 180
				index += get_angle(deg_angles[k], deg_angles[k + 1])

			poincare[i, j] = index

			# def get_diff(x, y):
			#    if(y > x):
			#       x, y = y, x
			#    return x-y

			# if(abs(index) > 170 and abs(index) < 190):
			#    cells = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
			#    deg_angles = [np.degrees(orientation_blocks[i + k][j + l]) % 180 for k, l in cells]
			#    index = 0
			#    for k in range(len(cells)-1):
			#       # if (abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90):
			#       #    deg_angles[k + 1] += 180
			#       index += get_diff(deg_angles[k], deg_angles[k + 1])
			#       deg_angles[k] = get_diff(deg_angles[k], deg_angles[k + 1])
			#    poincare[i, j] = index
			#    print(deg_angles)
			#    print(index)
			#    if(abs(index) > 2):
			#       print("^.^.^.^.^.^ Found something")
			#    print("     ", i, j)
			#    print("")
			#    sys.stdout.flush()

			### type classification

			angle_core = 180
			if (np.isclose(poincare[i, j], angle_core, 0, tolerance)):
				cores.append((i, j))
			angle_delta = -180
			if (np.isclose(poincare[i, j], angle_delta, 0, tolerance)):
				deltas.append((i, j))
			angle_whorl = 360
			if (np.isclose(poincare[i, j], angle_whorl, 0, tolerance)):
				whorls.append((i, j))

	cores = reduce_points(cores)
	deltas = reduce_points(deltas)
	whorls = reduce_points(whorls)
	# print(cores)
	# print(deltas)
	# print(whorls)

	singular_type = singular_type_classify([cores, deltas, whorls])
		
	# (y,x) in blk changes to (y, x) in pixels
	def blkToCoord(points_blk):
		return [(point_blk[0]*blk_sz, point_blk[1]*blk_sz) for point_blk in points_blk]

	cores = blkToCoord(cores)
	deltas = blkToCoord(deltas)
	whorls = blkToCoord(whorls)

	singular_pts = [cores, deltas, whorls]
	return poincare, singular_type, singular_pts


def minutiae(image_spook, roi_blks, blk_sz, radius=8):
	# 0, 1, 3, 4 neighbors
	minutiae_list = [[],[],[],[],[]]
	minutiae_type = np.full(image_spook.shape, -1)

	for i in range(radius, image_spook.shape[0] - radius):
		for j in range(radius, image_spook.shape[1] - radius):
			# not in RoI or is background, skip it
			if(roi_blks[i, j] == 1 or image_spook[i,j] == 0):
				continue
			eight_nei = image_spook[i-1 : i+1+1 , j -1: j +1+1]
			eight_nei_no = np.sum(eight_nei) - eight_nei[1, 1]
			minutiae_type[i,j] = eight_nei_no

	def clear_noise(image_spook, i, j, radius):

		# block = image_spook[i-radius: i + radius, j-radius: j + radius]
		
		# (lambda x: 0 if(x == 1 or x == 3))(block)
		changed = False

		if minutiae_type[i, j] == 1 or minutiae_type[i, j] == 3:
			for l in range(i-radius, i + radius):
				for m in range(j-radius, j + radius):
					# skip yourself
					if(l == i and j == m):
						continue
					if(minutiae_type[l,m] == 1
						or minutiae_type[l, m] == 3):
						changed = True
						minutiae_type[l, m] = -1
			if(changed):
				minutiae_type[i, j] = -1
		return

	for i in range(blk_sz, image_spook.shape[0] - blk_sz):
		for j in range(blk_sz, image_spook.shape[1] - blk_sz):
			if((roi_blks[i + blk_sz, j - blk_sz] == 1) or (roi_blks[i + blk_sz, j + blk_sz] == 1) or (roi_blks[i - blk_sz, j + blk_sz] == 1) or (roi_blks[i - blk_sz, j - blk_sz] == 1)):
				continue

			if(roi_blks[i, j] == 1 or minutiae_type[i, j] == -1):
				continue
			# 'isolated'
			if(minutiae_type[i,j] == 0):
				minutiae_list[0].append((i,j))

			# 'ending'
			elif(minutiae_type[i,j] == 1):
				clear_noise(minutiae_type, i, j, radius)

				# if still minutiae after clearing
				if(minutiae_type[i, j] == 1):
					minutiae_list[1].append((i, j))

			# 'edgepoint'
			elif(minutiae_type[i,j] == 2):
				pass
				# minutiae_list[2].append((i,j))

			# 'bifurcation'
			elif(minutiae_type[i,j] == 3):
				clear_noise(minutiae_type, i, j, radius)

				# if still minutiae after clearing
				if(minutiae_type[i, j] == 3):
					minutiae_list[3].append((i,j))

			# 'crossing'
			elif(minutiae_type[i,j] == 4):
				minutiae_list[4].append((i,j))
					

	return minutiae_list


def minutiae_draw(image_spook, minutiae_list, size=1, thicc=2):
	image_spook = image_spook*255

	# add color channels to ndarray
	if(len(image_spook.shape) == 2):
		image_color = np.stack((image_spook,)*3, axis=-1)
	else:
		image_color = image_spook

	resize_mult = 3
	image_color = cv2.resize(
		 image_color, (resize_mult*image_color.shape[0], resize_mult*image_color.shape[1]), interpolation=cv2.INTER_NEAREST)

	random.seed(133)
	# yellow isolated point
	# purple endpoint
	#
	# red/pink bifurcation
	# green crossing

	for i in range(len(minutiae_list)):
		minutiae_list_typed = minutiae_list[i]
		# get next 'random' color in sequence
		h, s, l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
		color = r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h, l, s)]
		
		# color = (color[0]*255, color[1]*255, color[2]*255)
		for minu in minutiae_list_typed:
			i, j = minu
			cv2.circle(image_color, (int(j*resize_mult+resize_mult/2), int(i*resize_mult+resize_mult/2)),
						  size, color, thicc)
		i += 1
	return image_color


def orientationField(img, blk_sz):
	img = enhance.contrast(img)
	img = enhance.median_filter(img, 5)
	orientation_blocks = gradient(img, blk_sz)
	orientation_blocks = smooth_gradient(
		orientation_blocks, blk_sz)
	# poincare, singular_type, singular_pts = fingerprint.singular_pts(
	#     img_roi, orientation_blocks, roi_blks, blk_sz)
	return orientation_blocks


def minutiaeOrientation(orientation_blocks, blk_sz, minutiaeList):
	angleList = []
	for minutiae in minutiaeList:
		try:
			angle = orientation_blocks[int(minutiae[1]/blk_sz), int(minutiae[0]/blk_sz)]
		except IndexError as err:
			angleList.append(0)
			print('--Exception ', err, ' in minutiaeOrientation: orientation_blocks.shape=',
			      orientation_blocks.shape, ', minutiae[1]/blk_sz=', minutiae[1]/blk_sz, '=', minutiae[1], '/',blk_sz, ', minutiae[0]/blk_sz=', minutiae[0]/blk_sz, '=', minutiae[0], '/',blk_sz, flush=True)
			traceback.print_exc()
			sys.stderr.flush()
		else:
			angleList.append(angle)
	return angleList


class Coordinate(object):
	def __init__(self, y, x):
		self.y = y
		self.x = x
		return


def gradient_window(img):
	"""
	Assumes img is a square, extract smooth angle of center pixel
	"""
	# dx = sobel_filter(img, 0)
	# dy = sobel_filter(img, 1)
	dy = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
	dx = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
	# with double the angle and squared magnitude
	angle_cos = dx*dx - dy*dy
	angle_sin = 2 * dx * dy
	# assumes img is a square
	blk_sz = min(img.shape[0], img.shape[1])
	if not (blk_sz % 2 == 1):
		blk_sz = blk_sz-1
	if not (blk_sz > 0):
		blk_sz = 3
	### average filter
	smooth_angle_sin = np.mean(angle_sin)
	smooth_angle_cos = np.mean(angle_cos)
	### gaussian filter
	# borderType = cv2.BORDER_CONSTANT
	# filterShape = (blk_sz, blk_sz)
	# if not (filterShape[0] > 0 and filterShape[0] % 2 == 1 and filterShape[1] > 0 and filterShape[1] % 2 == 1):
	# 	print(
	# 		"filterShape[0] > 0, filterShape[0] % 2 == 1, filterShape[1] > 0, filterShape[1] % 2 == 1")
	# 	print(filterShape[0] > 0, filterShape[0] % 2 == 1,
	# 	      filterShape[1] > 0, filterShape[1] % 2 == 1)
	# 	print(filterShape, flush=True)
	# # sin
	# smooth_angle_sin = cv2.GaussianBlur(angle_sin, filterShape, 0,
   #                                  borderType=borderType)
	# # center pixel
	# smooth_angle_sin = smooth_angle_sin[smooth_angle_sin.shape[0] //
   #                                 2, smooth_angle_sin.shape[1]//2]
	# # cos
	# smooth_angle_cos = cv2.GaussianBlur(angle_cos, filterShape, 0,
   #                                  borderType=borderType)
	# # center pixel
	# smooth_angle_cos = smooth_angle_cos[smooth_angle_cos.shape[0] //
   #                                 2, smooth_angle_cos.shape[1]//2]
	### arctan2
	smooth_angle = np.arctan2(smooth_angle_sin, smooth_angle_cos) / 2

	return smooth_angle % 3.14


def drawLine(img, centerCoord, angle, length, thicc=4, color=(0, 255, 0)):
	x1 = int(centerCoord[1] + (length*np.cos(angle))//2)
	y1 = int(centerCoord[0] - (length*np.sin(angle))//2)
	x2 = int(centerCoord[1] - (length*np.cos(angle))//2)
	y2 = int(centerCoord[0] + (length*np.sin(angle))//2)
	print("angle", angle, np.degrees(angle), flush=True)
	cv2.line(img, (x1, y1), (x2, y2), color, thicc)
	return


def localOrientation(img, coord, blk_sz=31):
	angle = 0
	blk_sz_half = blk_sz // 2
	# extract window around pixel coordinate
	beg = np.array([coord[0] - blk_sz_half, coord[1] - blk_sz_half+1])
	end = np.array([coord[0] + blk_sz_half, coord[1] + blk_sz_half+1])
	# check bounds
	if np.any(beg < 0):
		blk_sz_half = min(coord[0], coord[1])
	if np.any(end > img.shape):
		blk_sz_half = blk_sz_half + np.min(img.shape - end)
	# recalculate window
	beg = np.array([coord[0] - blk_sz_half, coord[1] - blk_sz_half+1])
	end = np.array([coord[0] + blk_sz_half, coord[1] + blk_sz_half+1])
	
	img = img[beg[0]:end[0], beg[1]: end[1]]
	# print("img.shape", img.shape, flush=True)
	# estimate angle of center pixel
	angle = gradient_window(img)
	## draw line for visualization
	# drawLine(img, (img.shape[0]//2, img.shape[1]//2), angle, blk_sz)
	# cv2.imshow('img', img)
	# print('img', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return angle

def minutiaeLocalOrientation(img, minutiaeList):
	angleList = []
	### enhance img before extracting angles
	# img = enhance.contrast(img)
	# img = enhance.median_filter(img, 5)
	for minutiae in minutiaeList:
		try:
			# coord = Coordinate(y=minutiae[1], x=minutiae[0])
			# numpy is (y,x)   <---  yolo is (x, y)
			coord = (minutiae[1], minutiae[0])
			angle = localOrientation(img, coord)
		except IndexError as err:
			angleList.append(0.0)
			print('--Exception ', err, ' in minutiaeLocalOrientation: coord=', coord, flush=True)
			traceback.print_exc()
			sys.stderr.flush()
		else:
			angleList.append(angle)

	return angleList
