import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image
# import scipy.io


# Load the .npy file containing the feature vectors
features_path = 'C:/Users/anura/OneDrive/Documents/all In one/concatenated_features_medical_images_144.npy'
features_vector = np.load(features_path)
D = features_vector
# // increment is denominator of precision 
for increment in range(1,10):

    ## changes to efficient 
    def Distance_measurement(D, Q):
        # Convert D and Q to signed integers once
         # Ensure D and Q are compatible for Euclidean distance calculation
        D = D.astype(np.float32)  # Cast to float for precision
        Q = Q.astype(np.float32)
    
    # Calculate Euclidean distances for each feature vector in D relative to Q
        s2 = np.linalg.norm(D - Q, axis=1)  # Vectorized Euclidean distance across rows
    
    # Sort the distances and get the indices of the closest images
        sorted_indices = np.argsort(s2)
    
        # Return the top 10 closest matches (indices + 1 for 1-based index)
        return sorted_indices[:increment] + 1



    # To store the 2D matrix of results
    matrix = []
    # Query image features
    for i in range(1, 697):
        Q = D[i - 1, :]
        result = Distance_measurement(D, Q)
        matrix.append(result)  # Store the result in the matrix


    # Define the specific ranges
    ranges = [
        (1, 59),
        (60, 84),
        (85, 116),
        (117, 156),
        (157, 187),
        (188, 229),
        (230, 272),
        (273, 297),
        (298, 317),
        (318, 353),
        (354, 397),
        (398, 470),
        (471, 527),
        (528, 596),
        (597, 659),
        (660, 696)
    ]

    def makes1(matrix,n):
    # Convert list to a 2D NumPy array for easy manipulation
        matrix = np.array(matrix)
        # print(matrix)
        for lower_bound, upper_bound in ranges:
        # Replace values within the specified range with 1, others with 0
            matrix[lower_bound - 1:upper_bound-1, :] = np.where(
            (matrix[lower_bound - 1:upper_bound-1, :] >= lower_bound) & 
            (matrix[lower_bound - 1:upper_bound-1, :] <= upper_bound), 1, 0)
        return matrix

    # Print the final binary result matrix
    # print("Binary Result Matrix (1 for values between 1-59 and 60-84 and 21 to 30 as so on, 0 for others):")

    # D{1,1}=1:59 = 59;
    # D{1,2}=60:84 = 25;
    # D{1,3}=85:116 = 32;
    # D{1,4}=117:156 = 40;
    # D{1,5}=157:187 = 31;
    # D{1,6}=188:229 = 42;
    # D{1,7}=230:272 = 43;
    # D{1,8}=273:297 = 25;
    # D{1,9}=298:317 = 20;
    # D{1,10}=318:353 = 36;
    # D{1,11}=354:397 = 44;
    # D{1,12}=398:470 = 73;
    # D{1,13}=471:527 = 57;
    # D{1,14}=528:596 = 69;
    # D{1,15}=597:659 = 63;
    # D{1,16}=660:696 = 37;

    

    matrix = makes1(matrix,17)
    # print(matrix)

    sumOfList = []
    def row_sums(matrix):
        # Calculate the sum of each row
        return np.sum(matrix, axis=1)

    # print(row_sums(matrix))
    sumOfList.append(row_sums(matrix))
    # print("Row Sums List:")
    # print(sumOfList)

    # Calculate p(i|Q) for each value in P
    pOfI = []
    def findP(P,Q):
        #  Calculate the precision for each query
        pOfI.append((P/Q))

    for i in sumOfList:
        findP(i,increment)
    # print("P(Iq) = \n")
    # print(pOfI)


    # Function to compute Average Retrieval Precision (ARP)
    def ARP(DB, P):
        # Flatten the list of precision values if P is a list of lists
        P_flattened = [item for sublist in P for item in sublist]
        Psum = np.sum(P_flattened)
        return (1 / DB) * Psum   # Compute ARP

    print(f"ARP = {ARP(6996, pOfI)} for {increment}")

    # Calculate recall (R) for each value in P
    ROfI = []
    def findR(P,Q):
        # Calculate the recall for each query
        ROfI.append((P/Q))

    for i in sumOfList:
        # # low,high = range(i)
        # range_obj = range(increment)
        # low = range_obj[0]  # First element in the range
        # high = range_obj[-1]  # Last element in the range
        # findR(i,(high-low)+1)
        findR(i,59)
    # print(f"R(Iq) = \n {ROfI}")

    # for i in sumOfList:
    #     # Get `low` and `high` based on `i` as the start and end of a range of length `i`
    #     range_obj = range(i)
    #     low = range_obj[0]  # First element in the range
    #     high = range_obj[-1]  # Last element in the range

    # # Call the `findR` function with the calculated range
    # findR(i, (high - low) + 1)

    # print(f"R(Iq) = \n {ROfI}")


    # Function to compute Average Retrieval Recall (ARR)
    def ARR(revImageInDB,R):
        R_flattened = [item for sublist in R for item in sublist]
        Rsum = np.sum(R_flattened)
        return (1 / revImageInDB) * Rsum  # Compute ARR

    print(f"ARR = {ARR(696, ROfI)} for {increment}")
