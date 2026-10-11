nums="12435645679"

def largest_odd(nums):
  result=[]

  for i in range(len(nums)):
    if int(nums[i]) % 2 != 0:
      result.append(nums[i])


  if result:
    return max(result)




  


  return -1
    
print(largest_odd(nums))
    