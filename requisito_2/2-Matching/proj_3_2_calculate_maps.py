import numpy as np
import cv2
import random
import os



def normalize_array(Arr,max_value):
    return ((Arr - np.amin(Arr) )/(np.amax(Arr) - np.amin(Arr) ))*(max_value)

baseFolder = '../1-Rectification/Image_Settings/DIFFERENT_CAMERAS_SETTINGS/TOO_FAR_SETTING/setting_1'
input_folder = os.path.join(baseFolder,'Rectified')
output_folder = os.path.join(baseFolder,'Depth_15')


cv_file = cv2.FileStorage(os.path.join(input_folder,'Parameters.xml'), cv2.FILE_STORAGE_READ)
b = cv_file.getNode("b").real()
f = cv_file.getNode("f").real()


imgLeft = cv2.imread(os.path.join(input_folder,'janelaL.jpg'),cv2.IMREAD_COLOR)
imgRight = cv2.imread(os.path.join(input_folder,'janelaR.jpg'),cv2.IMREAD_COLOR)

rowsLeft = imgLeft.shape[0]
colsLeft = imgLeft.shape[1]

rowsRight = imgRight.shape[0]
colsRight = imgRight.shape[1]

cv_file = cv2.FileStorage(os.path.join(output_folder,'points.xml'), cv2.FILE_STORAGE_READ)
imgLeftPositions = cv_file.getNode("imgLeftPositions").mat()
imgRightPositions = cv_file.getNode("imgRightPositions").mat()

dimgLeftPositions = np.array(imgLeftPositions,dtype=np.int32)
dimgRightPositions = np.array(imgRightPositions,dtype=np.int32)

imgLeftPositions = np.array(imgLeftPositions,dtype=np.float32)
imgRightPositions = np.array(imgRightPositions,dtype=np.float32)


XLpXR = imgLeftPositions[:,1] + imgRightPositions[:,1]
XLmXR = imgLeftPositions[:,1] - imgRightPositions[:,1]
YLpYR = imgLeftPositions[:,0] + imgRightPositions[:,0]


X = (b/2)*(XLpXR/XLmXR)
Y = (b/2)*(YLpYR/XLmXR)
Z = (b*f)/(XLmXR)


norm_Z = normalize_array(np.log10(Z),255)
norm_XLmXR = normalize_array(XLmXR,255)


X = X.astype(np.int32)
Y = Y.astype(np.int32)


depth2 = np.full([rowsLeft,colsLeft,3],[0,0,255] ,dtype = np.uint8)
disparity = np.full([rowsLeft,colsLeft,3], [0,0,255] ,dtype = np.uint8)

for i in range(X.shape[0]):


    depth2[dimgLeftPositions[i,0],dimgLeftPositions[i,1],0] = norm_Z[i]
    depth2[dimgLeftPositions[i,0],dimgLeftPositions[i,1],1] = norm_Z[i]
    depth2[dimgLeftPositions[i,0],dimgLeftPositions[i,1],2] = norm_Z[i]    

    disparity[dimgLeftPositions[i,0],dimgLeftPositions[i,1],0] = norm_XLmXR[i]
    disparity[dimgLeftPositions[i,0],dimgLeftPositions[i,1],1] = norm_XLmXR[i]
    disparity[dimgLeftPositions[i,0],dimgLeftPositions[i,1],2] = norm_XLmXR[i]    


print("Z mais proximo: %.1fcm"%(np.amin(Z)))
print("Z mais distante: %.1fcm"%(np.amax(Z)))

cv2.imwrite(os.path.join(output_folder,'Depth.png'),depth2)
cv2.imwrite(os.path.join(output_folder,'Disparity.png'),disparity)
