import numpy as np
import cv2

# Load an color image in grayscale
imgLeft = cv2.imread('../imgs-estereo/aloeL.png',cv2.IMREAD_COLOR)
imgRight = cv2.imread('../imgs-estereo/aloeR.png',cv2.IMREAD_COLOR)

window_size = 9

rowsLeft = imgLeft.shape[0]
colsLeft = imgLeft.shape[1]

rowsRight = imgRight.shape[0]
colsRight = imgRight.shape[1]

imgLeftPositions = []
imgRightPositions = []

window_halfsize = (window_size-1) //2

min_i = window_halfsize
max_i = rowsLeft - window_halfsize

min_j = window_halfsize
max_j = colsLeft - window_halfsize

for i in range(min_i, max_i):
    imgRowLeftPositions = []
    imgRowRightPositions = []
    for j in range(min_j, max_j):

        j_right_margin = int((j + 0.1*colsRight) if (j + 0.1*colsRight) < colsRight else colsRight)
        j_left_margin = int((j - 0.1*colsRight) if (j - 0.1*colsRight) > 0 else 0)

        template = imgLeft[(i - window_halfsize):(i + window_halfsize + 1), (j - window_halfsize):(j + window_halfsize + 1), :]

        croppedRight = imgRight[(i - window_halfsize):(i + window_halfsize) + 1, j_left_margin:j_right_margin , :]

        result = cv2.matchTemplate(croppedRight, template, cv2.TM_SQDIFF)

        maximum = np.amax(result)
        t_result = np.copy(result)
        while True:
            minimum = np.amin(t_result)
            r_i = i
            raw_i, raw_j = np.where( t_result == minimum )
            raw_i = raw_i[0]
            raw_j = raw_j[0]
            r_j = raw_j + window_halfsize
            if r_j > j:
                t_result[raw_i][raw_j] = maximum
            else:
                break
            
        minimum_compare_factor = 1.2 * minimum

        far_bool = np.ones(result.shape[1],dtype=np.bool)        
        min_far = (raw_j - window_halfsize) if (raw_j - window_halfsize) > 0 else 0
        max_far = (raw_j + window_halfsize + 1) if (raw_j + window_halfsize + 1) < (result.shape[1]) else (result.shape[1])
        far_bool[ min_far:max_far ] = False

        far_result = result[:,far_bool]
        number_matches = len(far_result[(far_result <= minimum_compare_factor)])

        max_number_matches = 0 

        if (number_matches > max_number_matches):
            # print("Left: %d %d\n"%(i, j))
            # print("Right: %d %d\n"%(r_i, r_j))            
            # temp_left = np.copy(imgLeft)
            # temp_right = np.copy(imgRight)
            # cv2.circle(temp_left, (j,i), 7, (0,0,255), 4)
            # cv2.circle(temp_right, (r_j,r_i), 7, (0,0,255), 4)
            # cv2.imshow('Left',temp_left)
            # cv2.imshow('Right',temp_right)
            # cv2.waitKey(0)
            continue
        else:            
            # print("Left: %d %d\n"%(i, j))
            # print("Right: %d %d\n"%(r_i, r_j))            
            # temp_left = np.copy(imgLeft)
            # temp_right = np.copy(imgRight)
            # cv2.circle(temp_left, (j,i), 7, (0,0,0), 4)
            # cv2.circle(temp_right, (r_j,r_i), 7, (0,0,0), 4)
            # cv2.imshow('Left',temp_left)
            # cv2.imshow('Right',temp_right)
            # cv2.waitKey(0)            
            imgRowLeftPositions.append( (i,j) )
            imgRowRightPositions.append( (r_i,r_j) )


#CANCELED -> SEE IN DISTANCE FROM IMAGES TOO DIFFERENT FROM NEAR NEIGHBORHOOD
    # error_range_size = 10
    # min_diff_ratio = 0.8
    # max_diff_ratio = 1.3
    # ii = 0
    # rmedian = []
    # lmedian = []
    # wrong_left_list = []
    # wrong_right_list = []
    # print(len(imgRowLeftPositions))
    # while ii < len(imgRowLeftPositions):
    #     min_ii = ii - error_range_size if ii - error_range_size > 0 else 0
    #     max_ii = ii + error_range_size if ii + error_range_size < (len(imgRowLeftPositions) - 1) else (len(imgRowLeftPositions) - 1)

    #     iip1 = ii + 1 if ii + 1 < (len(imgRowLeftPositions) - 1) else (len(imgRowLeftPositions) - 1)

    #     left_list = [k for k in range(min_ii,ii)]
    #     right_list = [k for k in range(iip1,max_ii+1)]

    #     point_distance = abs(imgRowLeftPositions[ii][1] - imgRowRightPositions[ii][1])

    #     if len(left_list):
    #         left_list_lelements = np.array([imgRowLeftPositions[k][1] for k in left_list])
    #         left_list_relements = np.array([imgRowRightPositions[k][1] for k in left_list])

    #         left_list_dif = left_list_lelements - left_list_relements

    #         left_list_difmedian = abs(np.median(left_list_dif))
            
    #         left_ratio = float(point_distance)/left_list_difmedian

    #         stranger_from_left = left_ratio > max_diff_ratio or left_ratio < min_diff_ratio

    #     else:
    #         left_list_difmedian = 0
    #         stranger_from_left = False


    #     if len(right_list):
    #         right_list_lelements = np.array([imgRowLeftPositions[k][1] for k in right_list])
    #         right_list_relements = np.array([imgRowRightPositions[k][1] for k in right_list])

    #         right_list_dif = np.abs(right_list_lelements - right_list_relements)

    #         right_list_difmedian = np.median(right_list_dif)
            
    #         right_ratio = float(point_distance)/right_list_difmedian

    #         stranger_from_right = right_ratio > max_diff_ratio or right_ratio < min_diff_ratio            

    #     else:
    #         right_list_difmedian = 0
    #         stranger_from_right = False


    #     if stranger_from_left and stranger_from_right:
    #         wrong_left_list.append(imgRowLeftPositions[ii])
    #         wrong_right_list.append(imgRowRightPositions[ii])
    #         del imgRowLeftPositions[ii]
    #         del imgRowRightPositions[ii]            
    #         ii = ii - 1

    #     else:
    #         lmedian.append(left_list_difmedian)
    #         rmedian.append(right_list_difmedian)
            
    #     ii = ii + 1

    # print(len(imgRowLeftPositions))
    # imgLeftPositions = imgLeftPositions + imgRowLeftPositions
    # imgRightPositions = imgRightPositions + imgRowRightPositions

    # for aa in range(len(wrong_left_list)):
    #     if aa % 10 == 0:
    #         a_i = wrong_left_list[aa][0]
    #         a_j = wrong_left_list[aa][1]
    #         a_r_i = wrong_right_list[aa][0]
    #         a_r_j = wrong_right_list[aa][1]        
    #         print("Left: %d %d\n"%(a_i, a_j))
    #         print("Right: %d %d\n"%(a_r_i, a_r_j)) 
    #         # print(lmedian[aa])           
    #         # print(rmedian[aa])
    #         temp_left = np.copy(imgLeft)
    #         temp_right = np.copy(imgRight)
    #         cv2.circle(temp_left, (a_j,a_i), 7, (0,255,255), 4)
    #         cv2.circle(temp_right, (a_r_j,a_r_i), 7, (0,255,255), 4)
    #         cv2.imshow('Left',temp_left)
    #         cv2.imshow('Right',temp_right)
    #         cv2.waitKey(0)