import cv2
import numpy as np
import math
import copy

import os

## MATRICES

base_folder = 'Image_Settings/SAME_CAMERA_SETTINGS/ROTATION'
setting_folder = os.path.join(base_folder,'setting_3')

#LEFT IMAGE DATA AND PARAMETERS
filenameImgLeft = os.path.join(setting_folder, 'left.jpg')
imgLeft = cv2.imread(filenameImgLeft, cv2.IMREAD_COLOR)

intrinsics_filename = 'Parameters/FacetimeBigImages/Intrinsics.xml'
cv_file = cv2.FileStorage(intrinsics_filename, cv2.FILE_STORAGE_READ)
instrinsic_matrix = cv_file.getNode("intrinsics").mat()
intrinsicsLeft = np.array(instrinsic_matrix,dtype=np.float64)

distortion_filename = 'Parameters/FacetimeBigImages/Distortion.xml'
cv_file = cv2.FileStorage(distortion_filename, cv2.FILE_STORAGE_READ)
distortion_coeffs = cv_file.getNode("distortion_coeffs").mat()
distortionCoeffsLeft = np.array(distortion_coeffs,dtype=np.float64)


#RIGHT IMAGE DATA AND PARAMETERS
filenameImgRight = os.path.join(setting_folder, 'right.jpg')
imgRight = cv2.imread(filenameImgRight, cv2.IMREAD_COLOR)

intrinsics_filename = 'Parameters/FacetimeBigImages/Intrinsics.xml'
cv_file = cv2.FileStorage(intrinsics_filename, cv2.FILE_STORAGE_READ)
instrinsic_matrix = cv_file.getNode("intrinsics").mat()
intrinsicsRight = np.array(instrinsic_matrix,dtype=np.float64)

distortion_filename = 'Parameters/FacetimeBigImages/Distortion.xml'
cv_file = cv2.FileStorage(distortion_filename, cv2.FILE_STORAGE_READ)
distortion_coeffs = cv_file.getNode("distortion_coeffs").mat()
distortionCoeffsRight = np.array(distortion_coeffs,dtype=np.float64)

####


filenames = [filenameImgLeft,filenameImgRight] # Image name
intrinsics_filename = ['Iphone/Intrinsics.xml','FacetimeBigImages/Intrinsics.xml'] # which camera to get intrinsics from

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


R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(intrinsicsLeft, distCoeffs1, intrinsicsRight, distCoeffs2, (cols, rows), R, T)


map1, map2 = cv2.initUndistortRectifyMap( intrinsicsLeft , distCoeffs1, R1, P1, (cols, rows), cv2.CV_32FC1)

leftOutput = cv2.remap(imgLeft, map1, map2, cv2.INTER_LANCZOS4)


map1, map2 = cv2.initUndistortRectifyMap(intrinsicsRight, distCoeffs2, R2, P2, (cols, rows), cv2.CV_32FC1)

rightOutput = cv2.remap(imgRight, map1, map2, cv2.INTER_LANCZOS4)


cv2.imshow('janelaL', leftOutput)
cv2.imshow('janelaR', rightOutput)
cv2.waitKey(0)

cv2.imwrite(os.path.join(base_folder,'janelaL.jpg'), leftOutput)
cv2.imwrite(os.path.join(base_folder,'janelaR.jpg'), rightOutput)

fLeft = (cameraMatrix1[0,0] + cameraMatrix1[1,1])/2.
fRight = (cameraMatrix2[0,0] + cameraMatrix2[1,1])/2.

focal = (fLeft + fRight)/2.
baseline = np.linalg.norm(T)

parameter_filename = os.path.join(base_folder, 'Parameters.xml')

cv_file_parameters = cv2.FileStorage(parameter_filename, cv2.FILE_STORAGE_WRITE)
cv_file_parameters.write("f", focal)
cv_file_parameters.write("b", baseline)
cv_file_parameters.release()