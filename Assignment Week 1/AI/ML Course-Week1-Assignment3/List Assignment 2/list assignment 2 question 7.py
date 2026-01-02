# Python Program to Find the Number Occurring Odd Number of Times in a List. A list is given in 
# which all elements except one element occurs an even number of times. The problem is to find 
# the element that occurs an odd number of times. 

list = [1,2,2,3,3,3,4,5,5,6,6,6,6,6] 
print("The numbers that occur odd number of times: ")
k = 1
for i in range(len(list)):
    item=list[i]
    if (i+1 != len(list)):
        item1=list[i+1]
    else:
        item1=""
    if(item == item1):
        k=k+1
    else:
        if(k%2 != 0):
            print(item, end = "  ")
        k=1