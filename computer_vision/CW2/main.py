import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import copy
import os
import time

frame1 = cv.imread('CO316 Coursework 2-2019_Data/exp_1496.png')
frame2 = cv.imread('CO316 Coursework 2-2019_Data/exp_1522.png')

frame1_plt = plt.imread('CO316 Coursework 2-2019_Data/exp_1496.png')
frame2_plt = plt.imread('CO316 Coursework 2-2019_Data/exp_1522.png')


def to_blue(img):
    new_img = copy.deepcopy(img)
    new_img[:, :, 1] = 0
    new_img[:, :, 2] = 0
    return new_img


def to_red(img):
    new_img = copy.deepcopy(img)
    new_img[:, :, 0] = 0
    new_img[:, :, 1] = 0
    return new_img


frame1 = cv.resize(frame1, (640, 360))
frame2 = cv.resize(frame2, (640, 360))

frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
frame1_kp, frame1_des = sift.detectAndCompute(frame1_gray, None)
frame2_kp, frame2_des = sift.detectAndCompute(frame2_gray, None)

bf = cv.BFMatcher()

matches = bf.match(frame1_des, frame2_des)
matches = sorted(matches, key=lambda x: x.distance)

frame1_kp_ind = [match.queryIdx for match in matches[:20]]
frame2_kp_ind = [match.trainIdx for match in matches[:20]]

img1 = cv.drawKeypoints(frame1, [frame1_kp[ind] for ind in frame1_kp_ind], frame1_gray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2 = cv.drawKeypoints(frame2, [frame2_kp[ind] for ind in frame2_kp_ind], frame2_gray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_with_matches = cv.drawMatches(frame1, frame1_kp, frame2, frame2_kp, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

for match in matches[:20]:
    print(np.subtract(frame2_kp[match.trainIdx].pt, frame1_kp[match.queryIdx].pt))

cv.imshow('Matches', img_with_matches)
cv.imshow('Frame 1 Keypoints', img1)
cv.imshow('Frame 2 Keypoints', img2)

cv.imwrite('figures/frame1_kp.png', img1)
cv.imwrite('figures/frame2_kp.png', img2)
cv.imwrite('figures/match_frames.png', img_with_matches)


plt.figure(1)
plt.imshow(frame1_gray, cmap='Blues', alpha=1)
plt.imshow(frame2_gray, cmap='Reds', alpha=0.5)

for match in matches[:20]:

    plt.plot([frame1_kp[match.queryIdx].pt[0], frame2_kp[match.trainIdx].pt[0]],
             [frame1_kp[match.queryIdx].pt[1], frame2_kp[match.trainIdx].pt[1]],
             linewidth=1, c='r', marker='+', mec='b')

mean_translation = np.mean([np.subtract(frame2_kp[match.trainIdx].pt, frame1_kp[match.queryIdx].pt) for match in matches[:20]], axis=0)

print(mean_translation)

# Translation matrix
M = np.float32([[1, 0, mean_translation[0]], [0, 1, mean_translation[1]]])
(rows, cols) = frame1_gray.shape[:2]

translated_frame1 = cv.warpAffine(frame1_gray, M, (cols, rows))

plt.figure(2)
plt.imshow(translated_frame1, cmap='Blues', alpha=1)
plt.imshow(frame2_gray, cmap='Reds', alpha=0.5)

for match in matches[:20]:

    plt.plot(frame1_kp[match.queryIdx].pt[0] + mean_translation[0],
             frame1_kp[match.queryIdx].pt[1] + mean_translation[1], marker='+', c='r')
    plt.plot(frame2_kp[match.trainIdx].pt[0], frame2_kp[match.trainIdx].pt[1], marker='+', c='b')

plt.show()
#
# def update(val=0):
#     stereo.setBlockSize(cv.getTrackbarPos('window_size', 'disparity'))
#     stereo.setUniquenessRatio(cv.getTrackbarPos('uniquenessRatio', 'disparity'))
#     stereo.setSpeckleWindowSize(cv.getTrackbarPos('speckleWindowSize', 'disparity'))
#     stereo.setSpeckleRange(cv.getTrackbarPos('speckleRange', 'disparity'))
#     stereo.setDisp12MaxDiff(cv.getTrackbarPos('disp12MaxDiff', 'disparity'))
#
#     disp = stereo.compute(frame1_gray, frame2_gray).astype(np.float32) / 16.0
#
#     cv.imshow('left', frame1_gray)
#     cv.imshow('disparity', (disp - min_disp) / num_disp)
#
#
#
# window_size = 5
# min_disp = 0
# num_disp = 16
# blockSize = window_size
# uniquenessRatio = 1
# speckleRange = 3
# speckleWindowSize = 3
# disp12MaxDiff = 200
# P1 = 600
# P2 = 2400
# cv.namedWindow('disparity')
# cv.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
# cv.createTrackbar('window_size', 'disparity', window_size, 21, update)
# cv.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
# cv.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
# cv.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)
# stereo = cv.StereoSGBM_create(
#     minDisparity=min_disp,
#     numDisparities=num_disp,
#     blockSize=window_size,
#     uniquenessRatio=uniquenessRatio,
#     speckleRange=speckleRange,
#     speckleWindowSize=speckleWindowSize,
#     disp12MaxDiff=disp12MaxDiff,
#     P1=P1,
#     P2=P2
# )
# update()

good_frame1_kp = np.int32([frame1_kp[match.queryIdx].pt for match in matches[:20]])
good_frame2_kp = np.int32([frame2_kp[match.trainIdx].pt for match in matches[:20]])

#
# good = []
# good_frame1_kp = []
# good_frame2_kp = []
# # ratio test as per Lowe's paper
# for m in matches:
#     print(m.distance)
#     if m.distance < 0.8:
#         good.append(m)
#         good_frame2_kp.append(frame2_kp[m.trainIdx].pt)
#         good_frame1_kp.append(frame1_kp[m.queryIdx].pt)
#
# good_frame1_kp = np.int32(good_frame1_kp)
# good_frame2_kp = np.int32(good_frame2_kp)


F, mask = cv.findFundamentalMat(good_frame1_kp, good_frame2_kp, cv.FM_LMEDS)

good_frame1_kp = good_frame1_kp[mask.ravel() == 1]
good_frame2_kp = good_frame2_kp[mask.ravel() == 1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(good_frame2_kp.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(frame1_gray, frame2_gray,lines1,good_frame1_kp, good_frame2_kp)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(good_frame1_kp.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(frame2_gray,frame1_gray,lines2, good_frame2_kp, good_frame1_kp)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()



# print(len(frame1_kp))
# print(frame2_kp[805].pt)


# goodfeaturestotrack()
# calculateopticalflowpyrlk()
# stereoSGBM_create()
