import math

def tangentDistance(a, b, height, width, choice):
	numTangents = len(choice)
	size = height * width
	tangents = [[0.0 for j in range(size)] for k in range(len(choice))]
	tangents1 = []
	tangents2 = []
	distance = 0

	tangents1 = calculateTangents(a, tangents, numTangents, height, width, choice)
	tangents2 = normalizeTangents(tangents1, numTangents, height, width)
	distance = calculateDistance(a, b, tangents2, numTangents, height, width)

	return distance

def calculateDistance(a, b, tangents, numTangents, height, width):
	distance = 0.0
	tmp = 0.0
	k = 0
	l = 0
	size = height * width
	for l in range(size):
		tmp = a[l] - b[l]
		distance += tmp ** 2

	for k in range(numTangents):
		tmp = 0.0
		for l in range(size):
			tmp += (a[l] - b[l]) * tangents[k][l]
		distance -= tmp ** 2

	return distance

def normalizeTangents(tangents, numTangents, height, width):
	size = height * width
	result_tangents = orthonormalizePzero(tangents, numTangents, size)

	return result_tangents

def orthonormalizePzero(A, num, dim):
	ortho_threshold = 1e-9
	n = 0
	m = 0
	d = 0
	dim1 = 0
	projection = 0.0
	projection1 = 0.0
	projection2 = 0.0
	projection3 = 0.0
	projection4 = 0.0
	norm = 0.0
	tmp = 0.0

	dim1 = dim - (dim % 4)

	for n in range(num):
		for m in range(n):
			projection1 = 0.0
			projection2 = 0.0
			projection3 = 0.0
			projection4 = 0.0
			for d in range(0, dim1, 4):
				projection1 += A[n][d] * A[m][d]
				projection2 += A[n][d+1] * A[m][d+1]
				projection3 += A[n][d+2] * A[m][d+2]
				projection4 += A[n][d+3] * A[m][d+3]
			projection = projection1+projection2+projection3+projection4
			for d in range(dim1, dim):
				projection += A[n][d] * A[m][d]
			for d in range(0, dim1, 4):
				A[n][d] -= projection * A[m][d]
				A[n][d+1] -= projection * A[m][d+1]
				A[n][d+2] -= projection * A[m][d+2]
				A[n][d+3] -= projection * A[m][d+3]
			for d in range(dim1, dim):
				A[n][d] -= projection * A[m][d]

		norm = 0.0
		for d in range(dim):
			tmp = A[n][d]
			norm += tmp ** 2
		if norm < ortho_threshold:
			norm = 0.0
		else:
			norm = 1.0 / math.sqrt(norm)
		for d in range(dim):
			A[n][d] *= norm

	return A


def calculateTangents(image, input_tangents, numTangents, height, width, choice):
	templatefactor1 = 0.1667
	templatefactor2 = 0.6667
	templatefactor3 = 0.08
	maxNumTangents = 7

	index = 0
	tangentIndex = 0
	maxDim = 0
	tp = 0.0
	factorW = 0.0
	offsetW = 0.0
	factorH = 0.0
	factor = 0.0
	offsetH = 0.0
	size = height * width
	tmp = [0.0 for j in range(size)]
	x1 = [0.0 for j in range(size)]
	x2 = [0.0 for j in range(size)]
	tangents = input_tangents
	currentTangent = []


	maxDim = max(height, width)

	factorW = width * 0.5
	offsetW = 0.5 - factorW
	factorW = 1.0 / factorW

	factorH = height * 0.5
	offsetH = 0.5 - factorH
	factorH = 1.0 / factorH

	factor = min(factorH, factorW)

	for k in range(height):
		index = k*width
		x1[0] = 0.0
		x1[index] = -0.5 * image[index+1]
		for j in range(1, width-1):
			index = k*width+j
			x1[index] = 0.5 * (image[index-1] - image[index+1])
		index = (k+1)*width-1
		x1[index] = 0.5 * image[index-1]

	for j in range(width):
		tmp[j] = x1[j]
		x1[j] = templatefactor2 * x1[j] + templatefactor1 * x1[j+width]

	for k in range(1, height-1):
		for j in range(width):
			index = k * width + j
			tp = x1[index]
			x1[index] = templatefactor1 * tmp[j] + templatefactor2 * x1[index] + templatefactor1 * x1[index + width]
			tmp[j] = tp

	for j in range(width):
		index = (height - 1) * width + j
		x1[index] = templatefactor1 * tmp[j] + templatefactor2 * x1[index]


	for j in range(2, width):
		for k in range(height):
			index = k * width + j
			x1[index] += templatefactor3 * image[index-2]

	for j in range(width-2):
		for k in range(height):
			index = k * width + j
			x1[index] -= templatefactor3 * image[index+2]

	for j in range(width):
		x2[j] = -0.5 * image[j + width]
		for k in range(1, height-1):
			index = k * width + j
			x2[index] = 0.5 * (image[index-width] - image[index+width])
		index = (height-1) * width + j
		x2[index] = image[index-width] * 0.5

	for j in range(height):
		index = j * width
		tmp[j] = x2[index]
		x2[index] = templatefactor2 * x2[index] + templatefactor1 * x2[index+1]

	for k in range(1, width-1):
		for j in range(height):
			index = j * width + k
			tp = x2[index]
			x2[index] = templatefactor1 * tmp[j] + templatefactor2 * x2[index] + templatefactor1 * x2[index+1]
			tmp[j] = tp

	for j in range(height):
		index = (j+1) * width - 1
		x2[index] = templatefactor1 * tmp[j] + templatefactor2 * x2[index]


	for j in range(2, height):
		for k in range(width):
			index = j * width + k
			x2[index] += templatefactor3 * image[index-2 * width]

	for j in range(0, height-2):
		for k in range(width):
			index = j * width + k
			x2[index] -= templatefactor3 * image[index+2 * width]

	for i in range(size):
		tangents[tangentIndex][i] = x1[i]
	tangentIndex += 1

	for i in range(size):
		tangents[tangentIndex][i] = x2[i]
	tangentIndex += 1

	index = 0
	for k in range(height):
		for j in range(width):
			tangents[tangentIndex][index] = factor * ((j + offsetW) * x1[index] - (k + offsetH) * x2[index])
			index += 1
	tangentIndex += 1

	index = 0
	for k in range(height):
		for j in range(width):
			tangents[tangentIndex][index] = factor * ((k + offsetH) * x1[index] + (j + offsetW) * x2[index])
			index += 1
	tangentIndex += 1

	index = 0
	for k in range(height):
		for j in range(width):
			tangents[tangentIndex][index] = factor * ((j + offsetW) * x1[index] + (k + offsetH) * x2[index])
			index += 1
	tangentIndex += 1

	index = 0
	for k in range(height):
		for j in range(width):
			tangents[tangentIndex][index] = factor * ((k + offsetH) * x1[index] - (j + offsetW) * x2[index])
			index += 1
	tangentIndex += 1

	index = 0
	for k in range(height):
		for j in range(width):
			tangents[tangentIndex][index] = x1[index] ** 2 + x2[index] ** 2
			index += 1
	tangentIndex += 1

	return tangents

