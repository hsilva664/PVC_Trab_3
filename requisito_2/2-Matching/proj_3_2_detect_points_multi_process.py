import numpy as np
import cv2
import time
import multiprocessing
import os
import random

# Load an color image in grayscale

baseFolder = '../1-Rectification/Image_Settings/DIFFERENT_CAMERAS_SETTINGS/TOO_FAR_SETTING/setting_1'
input_folder = os.path.join(baseFolder,'Rectified')
output_folder = os.path.join(baseFolder,'Depth_15')

imgLeft = cv2.imread(os.path.join(input_folder,'janelaL.jpg'),cv2.IMREAD_COLOR)
imgRight = cv2.imread(os.path.join(input_folder,'janelaR.jpg'),cv2.IMREAD_COLOR)

start_time = time.time()

num_processes = 8

window_size = 15

rowsLeft = imgLeft.shape[0]
colsLeft = imgLeft.shape[1]

rowsRight = imgRight.shape[0]
colsRight = imgRight.shape[1]

window_halfsize = (window_size-1) //2


def processLoop(amin_i,amax_i, return_dict):


    processImgLeftPositions = []
    processImgRightPositions = []
    for i in range(amin_i, amax_i):
        for j in range(min_j, max_j):

            if np.array_equal(imgLeft[i,j,:], [0,0,0]):
                continue

            search_margin_pct = 0.5

            j_right_margin = int((j + search_margin_pct*colsRight) if (j + search_margin_pct*colsRight) < colsRight else colsRight)
            j_left_margin = int((j - search_margin_pct*colsRight) if (j - search_margin_pct*colsRight) > 0 else 0)

            template = imgLeft[(i - window_halfsize):(i + window_halfsize + 1), (j - window_halfsize):(j + window_halfsize + 1), :]

            croppedRight = imgRight[(i - window_halfsize):(i + window_halfsize) + 1, j_left_margin:j_right_margin , :]

            result = cv2.matchTemplate(croppedRight, template, cv2.TM_SQDIFF)

            maximum = np.amax(result)
            t_result = np.copy(result)
            continueLoop = False
            while True:
                minimum = np.amin(t_result)
                if minimum == maximum:
                    continueLoop = True
                    break
                r_i = i
                raw_i, raw_j = np.where( t_result == minimum )
                raw_i = raw_i[0]
                raw_j = raw_j[0]
                r_j = raw_j + window_halfsize + j_left_margin
                if r_j >= j:
                    t_result[raw_i][raw_j] = maximum
                else:
                    break

            if continueLoop:
                continue
                
            minimum_compare_factor = 1.2 * minimum

            far_bool = np.ones(result.shape[1],dtype=np.bool)        
            min_far = (raw_j - window_halfsize) if (raw_j - window_halfsize) > 0 else 0
            max_far = (raw_j + window_halfsize + 1) if (raw_j + window_halfsize + 1) < (result.shape[1]) else (result.shape[1])
            far_bool[ min_far:max_far ] = False

            far_result = result[:,far_bool]
            number_matches = len(far_result[(far_result <= minimum_compare_factor)])

            max_number_matches = 0 

            if (number_matches > max_number_matches):
                continue
            else:
                processImgLeftPositions.append( (i,j) )
                processImgRightPositions.append( (r_i,r_j) ) 

            
    return_dict[multiprocessing.current_process().name] = { 'left': processImgLeftPositions,
                                                            'right': processImgRightPositions}



if __name__ == '__main__':

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    min_i = window_halfsize
    max_i = rowsLeft - window_halfsize

    min_j = window_halfsize
    max_j = colsLeft - window_halfsize

    process_i_min = min_i
    thred_inc = (max_i - min_i)//num_processes
    process_arr = []
    for ii in range(num_processes):
        process_i_max = process_i_min + thred_inc if (process_i_min + thred_inc) < max_i else max_i
        n_process = multiprocessing.Process(target=processLoop,args=(process_i_min, process_i_max,return_dict))
        process_arr.append(n_process)
        n_process.start()
        process_i_min = process_i_max

    for single_process in process_arr:
        single_process.join()

    imgLeftPositions = []
    imgRightPositions = []

    for inner_dict in return_dict.values():
        imgLeftPositions = imgLeftPositions + inner_dict['left']
        imgRightPositions = imgRightPositions + inner_dict['right']


    duration = time.time() - start_time

    print(duration)

    imgLeftPositions = np.array(imgLeftPositions)
    imgRightPositions = np.array(imgRightPositions)

    cv_file_points = cv2.FileStorage(os.path.join(output_folder,'points.xml'), cv2.FILE_STORAGE_WRITE)

    cv_file_points.write("imgLeftPositions", imgLeftPositions)
    cv_file_points.write("imgRightPositions", imgRightPositions)

    cv_file_points.release()    

    cL = np.copy(imgLeft)
    cR = np.copy(imgRight)
    for ii in range(len(imgLeftPositions)):
        if ii%1000 == 0:
            L_i = imgLeftPositions[ii][0]
            L_j = imgLeftPositions[ii][1]
            R_i = imgRightPositions[ii][0]
            R_j = imgRightPositions[ii][1]
            B = random.randint(0, 255)
            G = random.randint(0, 255)
            R = random.randint(0, 255)
            cv2.circle(cL, (L_j,L_i), 7, (B,G,R), 4)
            cv2.circle(cR, (R_j,R_i), 7, (B,G,R), 4)


    cv2.imwrite(os.path.join(output_folder,'detected_left.png'),cL)
    cv2.imwrite(os.path.join(output_folder,'detected_right.png'),cR)

