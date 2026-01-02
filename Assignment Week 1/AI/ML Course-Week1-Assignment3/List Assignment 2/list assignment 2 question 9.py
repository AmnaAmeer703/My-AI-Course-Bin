# Python Program to Swap the First and Last Element in a List. Python Program to Swap the First 
# and Last Element in a List

n = int(input("Enter the length of list: "))
list = []
for i in range(n):
    value = int(input("Enter the elements: "))
    list.append(value)
print("The orignal list is: ")
print(list)

list[0], list[-1], = list[-1], list[0]
print("The new list after swaping first and last element: ")
print(list)