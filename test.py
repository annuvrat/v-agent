s= "listen"
l= "silent"

def is_anagram(s, l):
    counts={}
    if len(s)!= len(l):
        return False

    for char in s:  
        if char not in counts:
            counts[char]=1
        else:
            counts[char]+=1


    for char in l:
        if char not in counts:
            return False


        counts[char] -=1  

    for count in counts.values():
        if count!=0:
         return False

    return True



print(is_anagram(s,l))