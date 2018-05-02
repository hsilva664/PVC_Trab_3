import cv2
import numpy as np
import os
import random

# baseFolder = '../requisito_2/1-Rectification/Image_Settings/DIFFERENT_CAMERAS_SETTINGS/TOO_FAR_SETTING/setting_1'
baseFolder = '../requisito_2/1-Rectification/Image_Settings/SAME_CAMERA_SETTINGS/ROTATION/setting_1'
rectified_folder = os.path.join(baseFolder,'Rectified')
depth_folder = os.path.join(baseFolder,'Depth_15')

image_name = os.path.join(rectified_folder,'janelaL.jpg')
image_color_flag = cv2.IMREAD_COLOR


cv_file = cv2.FileStorage(os.path.join(rectified_folder,'Parameters.xml'), cv2.FILE_STORAGE_READ)
b = cv_file.getNode("b").real()
f = cv_file.getNode("f").real()

cv_file = cv2.FileStorage(os.path.join(depth_folder,'points.xml'), cv2.FILE_STORAGE_READ)
imgLeftPositions = cv_file.getNode("imgLeftPositions").mat()
imgRightPositions = cv_file.getNode("imgRightPositions").mat()

raw_first_click = False

raw_initial_point = (0,0)
raw_final_point = (0,0)

initial_point_format_ij = None
initial_point_index = None
final_point_format_ij = None
final_point_index = None

raw_draw_line = False

raw_calculate_distance = False

def getClosestPoint(point): #format (i,j) and as numpy array
    minimum_distance_vector = np.linalg.norm(imgLeftPositions - point, axis=1)
    minimum_distance = np.amin(minimum_distance_vector)
    raw_i= np.where( minimum_distance_vector == minimum_distance )
    rand = random.randint(0,raw_i[0].shape[0]-1)
    return raw_i[0][rand]

    

def get_left_click(event, x, y, flags, param):    
    global raw_first_click, raw_initial_point, raw_final_point, raw_draw_line, raw_calculate_distance
    global initial_point_format_ij, initial_point_index, final_point_format_ij, final_point_index
    if event == cv2.EVENT_LBUTTONUP:
        if param['window'] == 'Raw Window':
            print("Clique em Raw Window\nLinha: %d\nColuna: %d"%(y,x))            
            raw_first_click = not raw_first_click
            if raw_first_click:
                raw_initial_point = (x,y)

                initial_point_format_ij = np.array([raw_initial_point[1],raw_initial_point[0]],dtype=np.float32)
                initial_point_index = getClosestPoint(initial_point_format_ij)

                closest_initial_point = imgLeftPositions[initial_point_index,:]
                raw_initial_point = (int(closest_initial_point[1]), int(closest_initial_point[0]) )

                raw_draw_line = False
                raw_calculate_distance = False
            else:
                raw_final_point = (x,y)

                final_point_format_ij = np.array([raw_final_point[1],raw_final_point[0]],dtype=np.float32)                
                final_point_index = getClosestPoint(final_point_format_ij)

                closest_final_point = imgLeftPositions[final_point_index,:]
                raw_final_point = (int(closest_final_point[1]), int(closest_final_point[0]))

                raw_draw_line = True
                raw_calculate_distance = True
        

            if raw_calculate_distance:                

                initial_point_3D = np.array([X[initial_point_index],Y[initial_point_index],Z[initial_point_index]], dtype=np.float32)
                final_point_3D = np.array([X[final_point_index],Y[final_point_index],Z[final_point_index]], dtype=np.float32)

                raw_distance = np.linalg.norm(initial_point_3D - final_point_3D)

                print("Distancia em centimetros: %.2f"%(raw_distance))

img = cv2.imread(image_name, image_color_flag)

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

cv2.namedWindow('Raw Window')
cv2.setMouseCallback('Raw Window', get_left_click, param = {'window': 'Raw Window'})


while True:
    if raw_draw_line:
        raw_img = np.copy(img)
        cv2.line(raw_img, raw_initial_point, raw_final_point, (0,0,255) )
        cv2.imshow('Raw Window',raw_img)
    else:
        cv2.imshow('Raw Window',img)    

    key = cv2.waitKey(100)

    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows() 
        exit()
