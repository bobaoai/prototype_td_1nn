from tangent_distance import *
from mnist import MNIST

def get_full_distance_matrix(points):
    distances = []
    for j in range(0, len(points)):
        p1 = points[j]
        for i in range(j+1,len(points)):
            p2 = points[i]
            print j,i
            tangentDistance(p1,p2)
            tangentDistance(p2,p1)
    return distances

def load_data():
    mndata = MNIST('/Users/apple/git/python-mnist/data')
    mndata.load_training()
    # mndata.train_images
    # mndata.train_labels
    return mndata

def prototype_selection():
    mndata = load_data()


if __name__ == '__main__':
    prototype_selection()
