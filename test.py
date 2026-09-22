arr = [11,7,15,2]
k = 9

def two_sum(arr, k):
    num_map={}
    for  i in range(len(arr)):
        complement = k-arr[i]
        if complement in num_map:
            return [num_map[complement], i]
        num_map[arr[i]] = i
    return None

print(two_sum(arr, k))