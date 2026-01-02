# Python Program to Find Average of a List. The program takes the elements of the list one by one 
# and displays the average of the elements of the list.

list = []
n = int(input("Enter The Number of Elements of list: "))
print("Enter The Elements: ")
for i in range(n):
    value = int(input())
    list.append(value)
list.sort()
print(list)
sum_of_elements = sum(list)
len_of_list = len(list)
avg = sum_of_elements/len_of_list

print("Average of given numbers in list is: ", avg)