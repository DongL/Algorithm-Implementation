# Problem: Find maximum subarray 


#################### Brute-force #####################
def brute_force(A):

    # initialize containers
    arr_size = len(A)
    max_sum = A[0]
    sum = dict()
    
    # scan all the possible combinations and record the max sum with its subarray
    for i in range(arr_size): 
        sum[i] = 0
        
        # scan all the possible combinations
        for j in range(i, arr_size, 1):
            sum[i] += A[j]

            # record the max sum with its subarray
            if max_sum < sum[i]:
                max_sum = sum[i] 
                subarray_left = i
                subarray_right = j
                                
    return subarray_left, subarray_right, max_sum


# Self check    
A = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
brute_force(A)

#################### Divide-and-conquer #####################
import numpy as np

def find_max_crossing_subarray(A, lo, mi, hi):
    sum = 0
    left_sum = A[lo]
    max_left = lo
    for i in range(mi, lo, -1):
        sum += A[i] 
        if left_sum < sum:
            left_sum = sum
            max_left = i
    
    sum = 0
    right_sum = A[mi+1]
    max_right = hi
    for j in range(mi+1, hi, 1):
        sum += A[j]
        if right_sum < sum:
            right_sum = sum
            max_right = j
    
    return (max_left, max_right, left_sum + right_sum)
        

def find_max_subarray(A, lo, hi):  
    if hi == lo:    
        return (lo, hi, A[lo])  
    
    
    else:
        mi = int(np.floor((lo + hi)/2))
        (left_lo, left_hi, left_sum) = find_max_subarray(A, lo, mi)    
        (right_lo, right_hi, right_sum) = find_max_subarray(A, mi + 1, hi)
        (cross_lo, cross_hi, cross_sum) = find_max_crossing_subarray(A, lo, mi, hi)
    
    if left_sum <= right_sum and cross_sum <= right_sum:
        return (right_lo, right_hi, right_sum)
    elif right_sum <= left_sum and cross_sum <= left_sum:
        return (left_lo, left_hi, left_sum)
    else:
        return (cross_lo, cross_hi, cross_sum)   


# Self check    
find_max_subarray(A, 0, len(A) - 1)    