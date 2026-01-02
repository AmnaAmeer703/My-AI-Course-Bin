# This is a Python Program to find the largest number in a list. The program takes a list and prints 
# the largest number in the list.

list = []
n = int(input("Please Enter The Length Of The List: "))
print("Enter The Elements: ")

for i in range(n):
    value = int(input())
    list.append(value)
max_value = max(list)
print("The List is: ", list)
print("The largest number in the given List is: ", max_value)