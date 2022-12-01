import os, sys
import random as rd
import pandas as pd
import numpy as np

n = [250, 1000, 10000]
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def is_inside_square(upper_left_point, size, point):
    """
    Check if a point is inside a square.
    :param upper_left_point: a point on the upper left corner of the square
    :param size: the size of the square
    :param point: a point
    :return: True if the point is inside the square, False otherwise
    """
    if point[0] < upper_left_point[0] or point[0] > upper_left_point[0] + size:
        return False
    if point[1] < upper_left_point[1] or point[1] > upper_left_point[1] + size:
        return False
    return True

def generate_concept_A(num_points, positive_noise = 0.97, negative_noise = 0.01):
    '''
    Generate n data points with 2 features lies within a rectangle of size [-6, 6] x [-4, 4]
    and y = [-1, 1] with noise and uniform distribution
    '''
    squares = [[-4, 3], [-2, -1], [2, 1]]
    size = 3
    
    point_lst = []
    labels = []
    #  first generate x1 and x2
    for i in range(num_points):
        x1 = rd.uniform(-6, 6)
        x2 = rd.uniform(-4, 4)
        
        # check if (x1, x2) lies in any of the squares with side = 3
        if is_inside_square(squares[0], size, [x1, x2]) or is_inside_square(squares[1], size, [x1, x2]) or is_inside_square(squares[2], size, [x1, x2]):
            # draw a random number from [0, 1]
            r = rd.uniform(0, 1)
            # if r is less than positive_noise, then label = 1
            point_lst.append([x1, x2])
            labels.append(1 if r < positive_noise else -1)
        else:
            # draw a random number from [0, 1]
            r = rd.uniform(0, 1)
            point_lst.append([x1, x2])
            # if r is less than negative_noise, then label = 1
            labels.append(1 if r < negative_noise else -1)
            
    return point_lst, labels

def generate_concept_B(num_points, positive_noise = 0.97, negative_noise = 0.01):
    '''
    Generate n data points with 2 features lies within a rectangle of size [-6, 6] x [-4, 4]
    and y = [-1, 1] with noise and uniform distribution
    '''
    squares = [[-4, 3], [-1, -2], [2, 0]]
    size = 1
    
    point_lst = []
    labels = []
    #  first generate x1 and x2
    for i in range(num_points):
        x1 = rd.uniform(-6, 6)
        x2 = rd.uniform(-4, 4)
        
        # check if (x1, x2) lies in any of the squares with side = 3
        if is_inside_square(squares[0], size, [x1, x2]) or is_inside_square(squares[1], size, [x1, x2]) or is_inside_square(squares[2], size, [x1, x2]):
            # draw a random number from [0, 1]
            r = rd.uniform(0, 1)
            # if r is less than positive_noise, then label = 1
            point_lst.append([x1, x2])
            labels.append(1 if r < positive_noise else -1)
        else:
            # draw a random number from [0, 1]
            r = rd.uniform(0, 1)
            point_lst.append([x1, x2])
            # if r is less than negative_noise, then label = 1
            labels.append(1 if r < negative_noise else -1)
            
    return point_lst, labels

# for different n, generate data points and labels for both concept A and B
for i in n:
    point_lst_A, labels_A = generate_concept_A(i)
    point_lst_B, labels_B = generate_concept_B(i)
    
    # save concept A data points and labels to csv file
    df_A = pd.DataFrame(point_lst_A, columns = ['x1', 'x2'])
    df_A['labels'] = labels_A
    
    # save concept B data points and labels to csv file
    df_B = pd.DataFrame(point_lst_B, columns = ['x1', 'x2'])
    df_B['labels'] = labels_B
    
    # save to csv file
    df_A.to_csv(f'{ROOT_DIR}/data/concept_A_' + str(i) + '.csv', index = False)
    df_B.to_csv(f'{ROOT_DIR}/data/concept_B_' + str(i) + '.csv', index = False)
    
    # generate data with no noise
    point_lst_A, labels_A = generate_concept_A(i, positive_noise=1, negative_noise=0)
    point_lst_B, labels_B = generate_concept_B(i, positive_noise=1, negative_noise=0)
    
    #turn point_list to numpy array
    X = np.array(point_lst_A)
    # save concept A data points and labels to csv file
    df_A = pd.DataFrame(point_lst_A, columns = ['x1', 'x2'])
    df_A['labels'] = labels_A
    
    # save concept B data points and labels to csv file
    df_B = pd.DataFrame(point_lst_B, columns = ['x1', 'x2'])
    df_B['labels'] = labels_B
    
    # save to csv file
    df_A.to_csv(f'{ROOT_DIR}/data/no_noise/concept_A_' + str(i) + '.csv', index = False)
    df_B.to_csv(f'{ROOT_DIR}/data/no_noise/concept_B_' + str(i) + '.csv', index = False)