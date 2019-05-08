import csv
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_off(fileName):
    retPts = []
    with open(fileName, 'r') as f:
        offReader = csv.reader(f, delimiter=' ')
        # get first two lines
        for i, row in enumerate(offReader):
            if i > 1:
                # have an additional 1 in order to use homogeuns xforms
                retPts.append((float(row[0]), float(row[1]), float(row[2]), 1))

    return retPts


def read_asc(fileName):
    retPts = []
    with open(fileName, 'r') as f:

        for i, line in enumerate(f):
            row = [val.strip() for val in line.strip().split()]

            retPts.append((float(row[0].strip()), float(
                row[1].strip()), float(row[2].strip()), 1))

    return retPts

# functions for finding point pairs


def find_best_point_pairs_mark_01(pts1, pts2):
    pts1_mean = np.mean(pts1[:, :3], axis=0)
    pts2_mean = np.mean(pts2[:, :3], axis=0)

    distances_pts1, indexesPts1 = np.unique(np.linalg.norm(
        pts1-pts2_mean, axis=1), return_index=True)
    distances_pts2, indexesPts2 = np.unique(np.linalg.norm(
        pts2-pts1_mean, axis=1), return_index=True)

    distancesIndexes = np.stack((indexesPts1, indexesPts2), axis=1)
    reordered = np.argsort(distancesIndexes, axis=0)
    return distancesIndexes[reordered[:, 0], 1]

# calculate the homogoenus transform


def get_current_transform(p_pts, q_pts):
    p_mean = np.mean(p_pts[:, :3], axis=0)
    q_mean = np.mean(q_pts[:, :3], axis=0)

    p_prime = p_pts[:, :3] - p_mean
    q_prime = q_pts[:, :3] - q_mean

    W = np.dot(q_prime.T, p_prime)

    U, S, V_T = np.linalg.svd(W)

    rotation = np.dot(V_T.T, U.T)
    translation = p_mean - np.dot(rotation, q_mean.T)
    ht = np.identity(4)
    ht[:3, :3] = rotation
    ht[:3, 3] = translation

    return ht


# pre processing
def trim_outliers(pts):
    meanPts = np.mean(pts, axis=0)
    distances = np.sum((pts-meanPts)**2, axis=1)

    # trim 90th percential
    return np.logical_xor(distances < np.percentile(distances, 90), distances > np.percentile(distances, 10))


p_pts = np.unique(
    np.array(read_asc('/media/Data/icpData/ICP-dental/W10.asc')), axis=0)
q_pts = np.unique(
    np.array(read_asc('/media/Data/icpData/ICP-dental/W9.asc')), axis=0)
final_p_pts = p_pts.copy()
final_q_pts = q_pts.copy()
# first make sure we have the same number of pts to begin with
p_pts = p_pts[trim_outliers(p_pts[:, :3]), :]
q_pts = q_pts[trim_outliers(q_pts[:, :3]), :]

maxPts = min(p_pts.shape[0], q_pts.shape[0])
print(max(p_pts.shape[0], q_pts.shape[0]), maxPts)
p_pts = p_pts[:maxPts, :].copy()
q_pts = q_pts[:maxPts, :].copy()


startTime = time.time()
'''
# target
P_file = './P_new.off'

# start
Q_file = './Q_new.off'

p_pts = np.array(read_off(P_file))
q_pts = np.array(read_off(Q_file))
'''

newQ_pts = np.ones_like(q_pts)
oldQ_pts = np.copy(q_pts)
indicies = find_best_point_pairs_mark_01(p_pts[:, :3], oldQ_pts[:, :3])

for i in range(11):
    indicies = find_best_point_pairs_mark_01(p_pts[:, :3], oldQ_pts[:, :3])
    ht = get_current_transform(p_pts.copy(), oldQ_pts[indicies, :].copy())

    oldQ_pts = ht.dot(oldQ_pts.T).T

    print(np.sum((p_pts[:, :3]-oldQ_pts[indicies, :3])**2))

f_ht = get_current_transform(oldQ_pts.copy(), q_pts.copy())
finalOut = f_ht.dot(final_q_pts.T).T
np.savetxt('test.off', finalOut, delimiter=' ',
           fmt='%f', header='COFF\n19300 0 0', comments='')
# np.savetxt('README.md', f_ht, fmt='%f')
print(f_ht)
print(time.time()-startTime)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(finalOut[:, 0], finalOut[:, 1],
           finalOut[:, 2], s=2, c='r', marker='.')
ax.scatter(final_p_pts[:, 0], final_p_pts[:, 1],
           final_p_pts[:, 2], s=2, c='b', marker='.')
plt.show()


''' 
1. how to find the "closest" points?????
1a. kdtree
1b. extract features find -> find regions with similar features then call those closest.....
1c. Do PCA then make very thing line up?
1d. clustering, not sure what to do here yet


2. Define metrics for defining what is a "GOOD" result
2a. apply metrics to all of the above methods and compare
2b. Do we want to talk about speed?

3. Make nice plots and put in presentations and be done.
4. We could do a demo
'''
