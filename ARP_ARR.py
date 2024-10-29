import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io

# Load the .mat file
mat = scipy.io.loadmat('C:/Users/anura/OneDrive/Documents/MATLAB/features.mat')

# Assuming the variable storing the data is called 'FV1' in the .mat file
D = mat['FV1']

# // increment is denominator of precision 
for increment in range(1,11):

    ## changes to efficient 
    def Distance_measurement(D, Q):
        # Convert D and Q to signed integers once
        D_signed = D.astype(np.int32)
        Q_signed = Q.astype(np.int32)
    
        # Vectorized calculation: broadcasting across all rows
        s1 = np.abs(D_signed - Q_signed)
        s11 = np.abs(1 + D_signed + Q_signed)
    
        # Compute p and sum across rows efficiently
        p = np.divide(s1, s11)
        s2 = np.sum(p, axis=1)

        # Sort the distances and get the indices of the closest images
        sorted_indices = np.argsort(s2)
    
        # Return the top 10 closest matches (indices + 1 for 1-based index)
        return sorted_indices[:increment] + 1



    # To store the 2D matrix of results
    matrix = []
    # Query image features
    for i in range(1, 401):
        Q = D[i - 1, :]
        result = Distance_measurement(D, Q)
        matrix.append(result)  # Store the result in the matrix


    def makes1(matrix,n):
    # Convert list to a 2D NumPy array for easy manipulation
        matrix = np.array(matrix)
        # print(matrix)
        for i in range(1, n):  # Adjust the range if you have more row
            lower_bound = (i - 1) * 10 + 1
            upper_bound = i * 10
        # For rows within the range of (i-1)*10 to i*10, replace with 1s or 0s
            matrix[lower_bound - 1:upper_bound, :] = np.where(
                (matrix[lower_bound - 1:upper_bound, :] >= lower_bound) &
                (matrix[lower_bound - 1:upper_bound, :] <= upper_bound), 1, 0)
        return matrix

    # Print the final binary result matrix
    # print("Binary Result Matrix (1 for values between 1-10 and 11-20 and 21 to 30 as so on, 0 for others):")
    matrix = makes1(matrix,41)
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

    print(f"ARP = {ARP(400, pOfI)} for {increment}")

    # Calculate recall (R) for each value in P
    ROfI = []
    def findR(P,Q):

        # Calculate the recall for each query
        ROfI.append((P/Q))

    for i in sumOfList:
        findR(i,10)
    # print(f"R(Iq) = \n {ROfI}")

    # Function to compute Average Retrieval Recall (ARR)
    def ARR(revImageInDB,R):
        R_flattened = [item for sublist in R for item in sublist]
        Rsum = np.sum(R_flattened)
        return (1 / revImageInDB) * Rsum  # Compute ARR

    print(f"ARR = {ARR(400, ROfI)} for {increment}")
