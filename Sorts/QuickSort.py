def qs(arr, lo, hi):
    '''
        Return:
                None
    '''
    if (lo >= hi):
        return

    pivotIdx = partition(arr, lo, hi)
    # On one side
    qs(arr, lo, pivotIdx - 1)
    # On the other side
    qs(arr, pivotIdx + 1, hi)

def partition(arr, lo, hi):
    '''
        Return: int (Pivot index)
    '''
    pivot = arr[hi]
    idx = lo - 1
    # Walk from low to high, not including hi (bcz pivot)
    # Weak sort on sub-array:
    for i in range(lo, hi, 1):
        if arr[i] <= pivot:
            idx += 1    # When first elem found, moves into starting point of array
            temp = arr[i]
            arr[i] = arr[idx]
            arr[idx] = temp
            # Everything below the pivot has been moved to the beginning of the array
    # If we find nothing below us, increment by 1
    idx += 1
    arr[hi] = arr[idx]
    arr[idx] = pivot
    
    return idx

def quick_sort(nums):
    '''
    Input : 
            nums - a List
    Return: 
            None (QuickSort modifies in-place)
    '''
    qs(nums, 0, len(nums) - 1)

# Quick sort happens in-place
'''
    First element of tuple - Input
    Second element of tuple - Expected values
'''
test_cases = [
  ([1, 2, 4, 3, 5, 6], [1, 2, 3, 4, 5, 6]),
  ([6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]),
  ([1, 2, 3, 2, 1], [1, 1, 2, 2, 3]),
  ([8, 7, 6, 4, 5], [4, 5, 6, 7, 8]),
  ([9, 3, 7, 4, 69, 420, 42], [3, 4, 7, 9, 42, 69, 420]),
  ([], []),
  ([1], [1])
]

for nums, expected in test_cases:
  quick_sort(nums)
  assert nums == expected, f"Failed test case: quick_sort({nums}) should return {expected}, but got {nums}"

print("All test cases passed!")