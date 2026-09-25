nums="1243564567"

def largest_odd(nums):
  result=[]

  for num in nums:
    if num%3 == 0:
      result.append(num)


  result.sort()
  result.reverse()
  for i in range(len(result)):
    return result[i]


  return False
    
print(largest_odd(nums))
    