import cv2
import numpy as np
import math
import copy

import os

## MATRICES

base_folder = '../1-Rectification/Image_Settings/SAME_CAMERA_SETTINGS/ROTATION'
setting_folder = os.path.join(base_folder,'setting_1')

#LEFT IMAGE DATA AND PARAMETERS
filenameImgLeft = os.path.join(setting_folder, 'left.jpg')
imgLeft = cv2.imread(filenameImgLeft, cv2.IMREAD_COLOR)

intrinsics_filename = '../1-Rectification/Parameters/FacetimeBigImages/Intrinsics.xml'
cv_file = cv2.FileStorage(intrinsics_filename, cv2.FILE_STORAGE_READ)
instrinsic_matrix = cv_file.getNode("intrinsics").mat()
intrinsicsLeft = np.array(instrinsic_matrix,dtype=np.float64)

distortion_filename = '../1-Rectification/Parameters/FacetimeBigImages/Distortion.xml'
cv_file = cv2.FileStorage(distortion_filename, cv2.FILE_STORAGE_READ)
distortion_coeffs = cv_file.getNode("distortion_coeffs").mat()
distortionCoeffsLeft = np.array(distortion_coeffs,dtype=np.float64)


#RIGHT IMAGE DATA AND PARAMETERS
filenameImgRight = os.path.join(setting_folder, 'right.jpg')
imgRight = cv2.imread(filenameImgRight, cv2.IMREAD_COLOR)

intrinsics_filename = '../1-Rectification/Parameters/FacetimeBigImages/Intrinsics.xml'
cv_file = cv2.FileStorage(intrinsics_filename, cv2.FILE_STORAGE_READ)
instrinsic_matrix = cv_file.getNode("intrinsics").mat()
intrinsicsRight = np.array(instrinsic_matrix,dtype=np.float64)

distortion_filename = '../1-Rectification/Parameters/FacetimeBigImages/Distortion.xml'
cv_file = cv2.FileStorage(distortion_filename, cv2.FILE_STORAGE_READ)
distortion_coeffs = cv_file.getNode("distortion_coeffs").mat()
distortionCoeffsRight = np.array(distortion_coeffs,dtype=np.float64)

####


filenames = [filenameImgLeft,filenameImgRight] # Image name

square_height = 3 #in cm
square_width = 3 #in cm

n_boards = 2

board_w = 8
board_h = 6

board_total  = board_w * board_h


cv2.namedWindow('Snapshot')
cv2.namedWindow('Raw Video')


image_points = [[],[]]
object_points = []

intrinsic_matrix = [np.zeros((3,3), dtype=np.float32 ) for i in range(2)]
distortion_coeffs = [np.zeros((4,1), dtype=np.float32 ) for i in range(2)]

successes = 0
frame = 0


new_object_points = []
for j in range(board_total):

    new_object_point = [0.0]*3

    new_object_point[0] = (float(j//board_w)) * square_height
    new_object_point[1] = (float(j%board_w)) * square_width
    new_object_point[2] = 0.0

    new_object_points.append(copy.deepcopy(new_object_point))

object_points.append(np.array(copy.deepcopy(new_object_points), dtype = np.float32))


while(successes < n_boards):    
    img =cv2.imread(filenames[successes],cv2.IMREAD_COLOR)
    rows = img.shape[0]
    cols = img.shape[1]

    raw_img = np.copy(img)
    cv2.imshow('Raw Video',raw_img)          


    found, corners = cv2.findChessboardCorners(img, (board_w,board_h), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if found:            

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        corner_count = corners2.shape[0]

        img = cv2.drawChessboardCorners(img, (board_w,board_h), corners2, found)


        if corner_count == board_total:               
            snapshot = np.copy(img)             

            cv2.imshow('Snapshot',snapshot)
            step = successes*board_total

            new_image_points = []
            for j in range(board_total):
                new_image_points.append(copy.deepcopy(corners2[j]))

            image_points[successes].append(np.array(copy.deepcopy(new_image_points), dtype = np.float32))

            successes = successes + 1

            print("%d successful Snapshots out of %d collected"%(successes,n_boards))

        else:
            snapshot = np.copy(gray)
            cv2.imshow('Snapshot',snapshot)
    else:
        snapshot = np.copy(gray)                   
        cv2.imshow('Snapshot',snapshot)



    if cv2.waitKey(2000) & 0xFF == ord('q'):
        cv2.destroyAllWindows() 
        exit()

cv2.waitKey(2000)
cv2.destroyAllWindows()

print("*** Calbrating the camera now...\n")


# print(object_points)


retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(object_points, image_points[0], image_points[1], intrinsicsLeft, distortionCoeffsLeft, intrinsicsRight, distortionCoeffsRight, (cols, rows))


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c, _ = img1.shape
    i = 0
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        if  i % board_w == 0:
            color = tuple(np.random.randint(0,255,3).tolist())
            # color = tuple([255,255,255])
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,5)
            img1 = cv2.circle(img1,tuple(pt1[0]),12,color,-1)
            img2 = cv2.circle(img2,tuple(pt2[0]),12,color,-1)
        i=i+1
    return img1,img2

# lines1 = cv2.computeCorrespondEpilines(image_points[1][0], 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(imgLeft,imgRight,lines1,image_points[0][0],image_points[1][0])



# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image

lines2 = cv2.computeCorrespondEpilines(image_points[0][0], 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(imgRight,imgLeft,lines2,image_points[1][0],image_points[0][0])


R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(intrinsicsLeft, distCoeffs1, intrinsicsRight, distCoeffs2, (cols, rows), R, T)
map1, map2 = cv2.initUndistortRectifyMap(intrinsicsRight, distCoeffs2, R2, P2, (cols, rows), cv2.CV_32FC1)
leftOutput = cv2.remap(img3, map1, map2, cv2.INTER_LANCZOS4)

cv2.imwrite('req3_epi_setting_B_right.jpg',img3)
cv2.imwrite('req3_epi_setting_B_left.jpg',img4)
cv2.imwrite('req3_epi_setting_B_rectified_right.jpg',leftOutput)

