#copy right: cited from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html

from mnist import MNIST
from math import *
from pyclust import *
templatefactor1 = 0.1667
templatefactor2 = 0.6667
templatefactor3 = 0.08
additiveBrightnessValue = 0.1
maxNumTangents = 9
ortho_singular_threshold = exp(-9)


# mndata.load_testing()

def tdIndex(y, x, width):
    return y * width + x


#


def orthonormalizePzero(A, num, dim):
    # A is tangents
    # calculates an orthonormal basis using Gram-Schmidt
    # returns 0 if basis can be found, 1 otherwise
    # try parallelization for CPU architectures with multiple FPUs

    n = m = d = dim1 = 0
    retval = 0
    projection = norm = tmp = 0.0
    projection1 = projection2 = projection3 = projection4 = 0.0

    dim1 = dim - dim % 4

    for n in xrange(num):
        A_n = A[n]
        for m in xrange(n):
            A_m = A[m]
            projection = 0.0
            projection1 = 0.0
            projection2 = 0.0
            projection3 = 0.0
            projection4 = 0.0
            for d in range(0, dim1, 4):
                projection1 += A_n[d] * A_m[d]
                projection2 += A_n[d + 1] * A_m[d + 1]
                projection3 += A_n[d + 2] * A_m[d + 2]
                projection4 += A_n[d + 3] * A_m[d + 3]
            projection = projection1 + projection2 + projection3 + projection4
            for d in xrange(dim):
                projection += A_n[d] * A_m[d]
            for d in range(0, dim1, 4):
                A_n[d] -= projection * A_m[d]
                A_n[d + 1] -= projection * A_m[d + 1]
                A_n[d + 2] -= projection * A_m[d + 2]
                A_n[d + 3] -= projection * A_m[d + 3]
            for d in xrange(dim):
                A_n[d] -= projection * A_m[d]

        # normalize
        norm = 0.0
        for d in xrange(dim):
            tmp = A_n[d]
            norm += tmp * tmp
        if norm < ortho_singular_threshold:
            norm = 0.0
        else:
            norm = 1.0 / sqrt(norm)
        for d in xrange(dim):
            A_n[d] *= norm

    # return retval
    #return tangents
    return A


# def orthonormalize (A, num, dim):
#      # A is a two demension list
#      # calculates an orthonormal basis using Gram-Schmidt
#      # returns 0 if basis can be found, 1 otherwise
#
#     n = m = d =0
#     retval = 0
#     projection = norm = tmp = 0.0
#     A_n = [0.0]
#     A_m = [0.0]
#     for n in range(1, num, 1):
#         A_n=list(A[n])
#         for m in range(1, n, 1):
#             A_m=list(A[m])
#             projection=0.0
#             for d in range(1, dim, 1):
#                 # get projection onto existing vector (scalar product)
#                 projection+=A_n[d]*A_m[d]
#             for d in range(1, dim, 1):
#                 # subtract component within existing subspace
#                 A_n[d]-=projection*A_m[d]
#
#         # normalize
#         norm=0.0
#         for (d=0 d<dim ++d)
#             tmp=A_n[d]
#             norm+=tmp*tmp
#         if (norm<ortho_singular_threshold)
#             retval=1
#         norm=1.0/sqrt(norm)
#         for (d=0 d<dim ++d)
#             A_n[d]*=norm
#
#
#     return retval



def calculateTangents(image, tangents, numTangents, height=28, width=28, choice=[1, 1, 1, 1, 1, 1, 0, 0, 0],
                      background=0.0):
    j = 0
    k = 0
    ind = 0
    tangentIndex = 0
    maxdim = height if height > width else width  # #min

    tp = 0.0
    factorW = 0.0
    offsetW = 0.0
    factorH = 0.0
    factor = 0.0
    offsetH = 0.0
    halfbg = 0.0
    # address
    tmp = [0] * maxdim
    size = height * width
    x1 = [0] * size
    x2 = [0] * size
    currentTangent = 0.0
    # maxdim =

    factorW = width * 0.5
    offsetW = 0.5 - factorW
    factorW = 1.0 / factorW

    factorH = height * 0.5
    offsetH = 0.5 - factorH
    factorH = 1.0 / factorH
    factor = factorH if factorH < factorW else factorW  # #min

    halfbg = 0.5 * background

    # x1 shift along width */
    # first use mask 1 0 -1 */
    for k in xrange(height):
        # first column */
        ind = tdIndex(k, 0, width)
        x1[ind] = halfbg - image[ind + 1] * 0.5
        # # other columns */
        for j in xrange(width - 1):
            ind = tdIndex(k, j, width)
            x1[ind] = (image[ind - 1] - image[ind + 1]) * 0.5

            # # last column */
        ind = tdIndex(k, width - 1, width)
        x1[ind] = image[ind - 1] * 0.5 - halfbg

    # # now compute 3x3 template */
    #   # first line */
    for j in xrange(width):
        tmp[j] = x1[j]
        x1[j] = templatefactor2 * x1[j] + templatefactor1 * x1[j + width]

    # # other lines */
    for k in xrange(height - 1):
        for j in xrange(width):
            ind = tdIndex(k, j, width)
            tp = x1[ind]
            x1[ind] = templatefactor1 * tmp[j] + templatefactor2 * x1[ind] + templatefactor1 * x1[ind + width]
            tmp[j] = tp

            #   # last line */
    for j in xrange(width):
        ind = tdIndex(height - 1, j, width)
        x1[ind] = templatefactor1 * tmp[j] + templatefactor2 * x1[ind]

        # # now add the remaining parts outside the 3x3 template */
        # # first two columns */
    for j in xrange(2):
        for k in xrange(height):
            ind = tdIndex(k, j, width)
            x1[ind] += templatefactor3 * background

            # # other columns */
    for j in range(2, width, 1):
        for k in xrange(height):
            ind = tdIndex(k, j, width)
            x1[ind] += templatefactor3 * image[ind - 2]

    for j in range(width - 2):
        for k in xrange(height):
            ind = tdIndex(k, j, width)
            x1[ind] -= templatefactor3 * image[ind + 2]

    # # last two columns*/
    for j in range(width - 2, width, 1):
        for k in xrange(height):
            ind = tdIndex(k, j, width)
            x1[ind] -= templatefactor3 * background

    # x2 shift along height */
    # first use mask 1 0 -1 */
    for j in range(width):
        # first line */
        x2[j] = halfbg - image[j + width] * 0.5
        # other lines */
        for k in xrange(height - 1):
            ind = tdIndex(k, j, width)
            x2[ind] = (image[ind - width] - image[ind + width]) * 0.5

        # last line */
        ind = tdIndex(height - 1, j, width)
        x2[ind] = image[ind - width] * 0.5 - halfbg

    # now compute 3x3 template */
    # first column */
    for j in range(height):
        ind = tdIndex(j, 0, width)
        tmp[j] = x2[ind]
        x2[ind] = templatefactor2 * x2[ind] + templatefactor1 * x2[ind + 1]

    # other columns */
    for k in range(width - 1):
        for j in range(height):
            ind = tdIndex(j, k, width)
            tp = x2[ind]
            x2[ind] = templatefactor1 * tmp[j] + templatefactor2 * x2[ind] + templatefactor1 * x2[ind + 1]
            tmp[j] = tp

    # last column */
    for j in range(height):
        ind = tdIndex(j, width - 1, width)
        x2[ind] = templatefactor1 * tmp[j] + templatefactor2 * x2[ind]

    # now add the remaining parts outside the 3x3 template */
    for j in xrange(2):
        for k in range(width):
            ind = tdIndex(j, k, width)
            x2[ind] += templatefactor3 * background

    for j in range(2, height, 1):
        for k in range(width):
            ind = tdIndex(j, k, width)
            x2[ind] += templatefactor3 * image[ind - 2 * width]

    for j in xrange(height - 2):
        for k in range(width):
            ind = tdIndex(j, k, width)
            x2[ind] -= templatefactor3 * image[ind + 2 * width]

    for j in range(height - 2, height, 1):
        for k in range(width):
            ind = tdIndex(j, k, width)
            x2[ind] -= templatefactor3 * background

    tangentIndex = 0

    if choice[0] > 0:  # horizontal shift
        currentTangent = tangents[tangentIndex]
        for ind in xrange(size):
                currentTangent[ind] = x1[ind]
        tangentIndex += 1

    if (choice[1] > 0):  # vertical shift
        currentTangent = tangents[tangentIndex]
        for ind in xrange(size): currentTangent[ind] = x2[ind]
        tangentIndex += 1

    if (choice[2] > 0):  # hyperbolic    1
        #        Vapnik book says this is "diagonal deformation" (error),
        #        this is the "axis deformation"
        currentTangent = tangents[tangentIndex]
        ind = 0
        for k in xrange(height):
            for j in xrange(width):
                currentTangent[ind] = ((j + offsetW) * x1[ind] - (k + offsetH) * x2[ind]) * factor
                ind += 1

        tangentIndex += 1

    if (choice[3] > 0):  # hyperbolic    2, (description = inverse of hyperbolic 1)
        currentTangent = tangents[tangentIndex]
        ind = 0
        for k in xrange(height):
            for j in xrange(width):
                currentTangent[ind] = ((k + offsetH) * x1[ind] + (j + offsetW) * x2[ind]) * factor
                ind += 1

        tangentIndex += 1

    if (choice[4] > 0):  # scaling
        currentTangent = tangents[tangentIndex]
        ind = 0
        for k in xrange(height):
            for j in xrange(width):
                currentTangent[ind] = ((j + offsetW) * x1[ind] + (k + offsetH) * x2[ind]) * factor
                ind += 1

        tangentIndex += 1

    if (choice[5] > 0):  # rotation
        currentTangent = tangents[tangentIndex]
        ind = 0
        for k in xrange(height):
            for j in xrange(width):
                currentTangent[ind] = ((k + offsetH) * x1[ind] - (j + offsetW) * x2[ind]) * factor
                ind += 1

        tangentIndex += 1

    if (choice[6] > 0):  # line thickness
        currentTangent = tangents[tangentIndex]
        ind = 0
        for k in xrange(height):
            for j in xrange(width):
                currentTangent[ind] = x1[ind] * x1[ind] + x2[ind] * x2[ind]
                ind += 1

        tangentIndex += 1

    if (choice[7] > 0):  # additive brightness
        currentTangent = tangents[tangentIndex]
        for ind in xrange(size):
            currentTangent[ind] = additiveBrightnessValue
        tangentIndex += 1

    if (choice[8] > 0):  # multiplicative brightness
        currentTangent = tangents[tangentIndex]
        for ind in xrange(size):
            currentTangent[ind] = image[ind]
        tangentIndex += 1

    del tmp
    del x1
    del x2

    assert tangentIndex == numTangents
    # return tangentIndex
    return numTangents, tangents


def tangentDistance(imageOne, imageTwo, height=28, width=28, choice=[1, 1, 1, 1, 1, 1, 0, 0, 0], background=0.0):
    # first part of main function
    i = 0
    numTangents = 0
    numTangentsRemaining = 0
    dist = []

    size = width * height
    print maxNumTangents
    for i in xrange(maxNumTangents):
        if choice[i] > 0:
            print "yes"
            numTangents += 1

    # tangents is a list with numTangents's length
    tangents = [0] * numTangents

    print len(tangents)
    # assert(tangents)

    # tangents have many one-arrays, with length of numTangents
    for i in xrange(numTangents):
        tangents[i] = [0.0] * size
        # assert(tangents[i])

    # determine the tangents of the first image
    numTangents, tangents = calculateTangents(imageOne, tangents, numTangents, height, width, choice, background)

    # find the orthonormal tangent subspace
    numTangentsRemaining, tangents = normalizeTangents(tangents, numTangents, height, width)

    # determine the distance to the closest point in the subspace
    dist=calculateDistance(imageOne, imageTwo, tangents, numTangentsRemaining, height, width)

    print dist
    for i in xrange(len(tangents)):
        print len(tangents[i])
        for j in xrange(10):
            print tangents[i][j]

    del tangents

    return dist

def calculateDistance(imageOne, imageTwo, tangents, numTangents, height,width):
    #first part of the main function
    dist=tmp=0.0
    tangents_k = [0.0]
    k=l=0

    size=height*width

    # first calculate squared Euclidean distance
    for l in xrange(size):
        tmp=imageOne[l]-imageTwo[l]
        dist+=tmp*tmp
    

    # then subtract the part within the subspace
    for k in xrange(numTangents):
        tangents_k=tangents[k]
        tmp=0.0
        for l in xrange(size):
            tmp+=(imageOne[l]-imageTwo[l])*tangents_k[l]
        dist-=tmp*tmp

    return dist

def twoSidedTangentDistance(imageOne, imageTwo, height=28, width=28, choice=[1,1,1,1,1,1,0,0,0], background=0.0):
    i=numTangents=numTangentsRemaining=0
    tangents=dist=0.0

    size=width*height

    for i in xrange(maxNumTangents):
        if(choice[i]>0):
            numTangents+=1
    

    tangents=[0.0]*(2*numTangents)
    #assert(tangents>0)
    for i in xrange(2*numTangents):
        tangents[i]=[0.0]*size
        # assert(tangents[i]>0)
    

    # determine the tangents of the images
    numTangents,tangents1=calculateTangents(imageOne, tangents[:numTangents], numTangents, height, width, choice, background)
    numTangents, tangents2 = calculateTangents(imageTwo, tangents[numTangents:], numTangents, height, width, choice, background)
    tangents=tangents1+tangents2
    # find the orthonormal tangent subspace
    numTangentsRemaining, tangents = normalizeTangents(tangents, 2*numTangents, height, width)

    # determine the distance to the closest point in the subspace
    dist=calculateDistance(imageOne, imageTwo, tangents, numTangentsRemaining, height, width)


    del tangents
    print dist
    return dist


def normalizeTangents(tangents, numTangents, height, width):
    size = height * width

    tangents = orthonormalizePzero(tangents, numTangents, size)

    # here we always return the original number of tangents dimensions
    # "lost" in the normalization are set to zero vectors this is not
    # what was intended originally, but it works and is kept for
    # backward compatibility
    return numTangents, tangents


if __name__ == '__main__':
    mndata = MNIST('/Users/apple/git/python-mnist/data')
    mndata.load_training()
    print len(mndata.train_images[0])
    tangentDistance(mndata.train_images[0], mndata.train_images[1])
    twoSidedTangentDistance(mndata.train_images[0], mndata.train_images[1])

