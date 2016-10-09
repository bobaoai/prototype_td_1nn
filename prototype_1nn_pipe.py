from tangent_distance import *
# from prototype_1nn_pipe import*
import Pycluster
from mnist import MNIST
import pandas, numpy, scipy
from scipy import stats
from datetime import datetime
import numpy as np
import random
import ctypes
tss = ctypes.CDLL("/Users/apple/Desktop/2016Quarter/CSE250B/hw1/tangent-distance-GPL/ts.so")
str(datetime.now())


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    # randomly initialize an array of k medoid indices
    M = np.sort(np.random.choice(n, k))

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in xrange(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C
#
# def get_full_distance_matrix(points):
#     distances = []
#     for j in range(0, len(points)):
#         print str(datetime.now())
#         print j
#         arr1 = (c_double * len(points[j]))([j])
#         for i in range(j+1,len(points)):
#             p2 = points[i]
#             # print j,i
#             # arr1 = (c_double * len(list1))(*list1)
#             arr2 = (c_double * len(list2))(*list2)
#             distance = tss.distance
#             distance.restype = c_double
#             a= distance(arr1,arr2)
#             distances.append(twoSidedTangentDistance(p1,p2))
#             # tangentDistance(p2,p1)
#     return distances

def load_data():
    mndata = MNIST('/Users/apple/git/python-mnist/data')
    mndata.load_training()
    # mndata.train_images
    # mndata.train_labels
    return mndata

def prototype_selection():
    mndata = MNIST('/Users/apple/git/python-mnist/data')
    mndata.load_training()
    label = mndata.train_labels[:5000]
    training_data=pandas.DataFrame(mndata.train_images[:5000])
    print training_data.shape
    col=[str(i) for i in range(1,training_data.shape[1]+1,1)]
    training_data.columns = col
    training_data['L'] = pandas.Series(label, index=training_data.index)

    print training_data.shape
    mn_class = {}
    for i in xrange(11):
        mn_class[str(i)]=training_data[training_data.L==i]
    id_list = []
    for i in range(0,10,1):
        print str(datetime.now())
        print i
        points = mn_class[str(i)][mn_class[str(i)].columns[:training_data.shape[1]-1]]
    #     print points.head()
        print points.shape
        points = points.values.tolist()
    #     print len(points)
    #     print len(points[0])
        dis_ma = get_full_distance_matrix(points)
        print str(datetime.now())
        print "end distance"
        clusterid, nfound = kMedoids(D=dis_ma, k=10, tmax=1000)
        id_list.append(clusterid)



if __name__ == '__main__':
    str(datetime.now())
    # prototype_selection()
    from ctypes import *
    # arr = (ctypes.c_int * len(pyarr))(*pyarr)
    mndata = MNIST('/Users/apple/git/python-mnist/data')
    mndata.load_training()
    list1 = mndata.train_images[0]
    list2 = mndata.train_images[3]
    list3 = mndata.train_images[22222]
    print "label1",mndata.train_labels[0]
    print "label2", mndata.train_labels[3]
    print "label3",mndata.train_labels[22222]
    tss = ctypes.CDLL("/Users/apple/Desktop/2016Quarter/CSE250B/hw1/tangent-distance-GPL/ts.so")
    arr1 = (c_double * len(list1))(*list1)
    arr2 = (c_double * len(list2))(*list2)
    arr3 = (c_double * len(list3))(*list3)
    onedistance = tss.onesidedistance

    onedistance.restype = c_double
    distance = tss.distance
    distance.restype = c_double
    a= distance(arr1,arr2)
    b= distance(arr1,arr3)
    c= distance(arr2,arr3)
    print a,b,c
    a= onedistance(arr1,arr2)
    b= onedistance(arr1,arr3)
    c= onedistance(arr2,arr3)
    print a,b,c
    # print hh

