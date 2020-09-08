#################### bubble_sort #####################
# O(n^2) time | O(1) space
# Best O(n); Average, worst O(n^2) time  
def bubble_sort(A):
    for i in range(len(A) - 1):
        for j in range(len(A) - 1, i, -1):
            if A[j] < A[j - 1]:
                A[j], A[j - 1] = A[j - 1], A[j]
                

def bubbleSort2(array):
	for j in range(len(array)):
		for i in range(len(array) - j - 1):
            if array[i] > array[i + 1]:
				array[i], array[i + 1] = array[i + 1], array[i]
	return array		                


#################### merge_sort #####################
def merge_sort(A):

    if len(A) > 1:
        mi = int(len(A)/2)
        left = A[:mi]
        right = A[mi:]
        
        merge_sort(left)
        merge_sort(right)   
        
        left.append(float("inf"))
        right.append(float("inf"))

        i = 0 
        j = 0
        k = 0
        
        while i <= len(left) and j <= len(right) and k < len(A):
            if left[i] <= right[j]:
                A[k] = left[i]
                i += 1
            else:
                A[k] = right[j]
                j += 1
            k += 1                


#################### insertion_sort #####################
# O(n^2) time | O(1) space
# Best O(n); Average, worst O(n^2) time  
def insertion_sort(A):
    for i in range(1,len(A)): 
                 
        currentvalue = A[i]     
        pos = i             
        
        while pos>0 and A[pos-1]>currentvalue:
            
            A[pos]=A[pos-1]
            pos -= 1

        A[pos]=currentvalue     

        
def insertionSort2(array):
    # Write your code here.
    for i in range(0, len(array)):
		insertionSwap(array, i, i+1)
	return array
			
def insertionSwap2(array, i, j):
	for k in range(len(array[:j])):
		if array[i] < array[k]:
			array[i], array[k] = array[k], array[i]


        
        
#################### findMedian #####################        
        
def findMedian(A):
    assert len(A) > 0
    
    # sort
    insertion_sort(A)
    
    # get median
    mi = int((len(A) + 1)/2) - 1
    return A[mi] 

def findMedianOfMedians(A):
    if len(A) <= 5:
        return findMedian(A)
    
    # split into n/5 groups
    grp = {}

    n = int(len(A)/5)
    mod = len(A) % 5
    runs = n + (1 if mod > 0 else 0)
    
    for i in range(runs):
        grp[i] = []

    
    j = 0
    for i in range(len(A)):
        grp[j].append(A[i])
        
        if (i + 1) % 5 == 0 and i + 1 != len(A):
            j += 1
    
    # find the median for each group
    medians = [findMedian(v) for v in grp.values() if len(v) != 0]
    
    # recursively find the median of medians
    medianOfMedians = SELECT(medians, int((len(medians) + 1)/2) - 1)
    
    return medianOfMedians 


def SELECT(A, i):
    
    if i < 0 or i > len(A):
        raise ValueError(f'i must be less than {len(A)} and greater than 0')

    # base case
    if len(A) == 1:
        return A[0]

    # partition
    x = findMedianOfMedians(A)
    
    left = [el for el in A if el <= x]
    right = [el for el in A if el > x]
    p = len(left)
    
    # find ith element
    if p == i:
        return x
    elif i < p:
        return SELECT(left, i)  
    else:
        return SELECT(right, i - p)        


# Self check
A = list(range(10))
print(A)
SELECT(A,5)            
