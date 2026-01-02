# The program takes a list and prints the largest number in the list. The program takes a list and 
# prints the second largest number in the list.

list = []
n = int(input("Please Enter The Length Of The List: "))
print("Enter The Elements: ")

for i in range(n):
    value = int(input())
    list.append(value)
max_value = max(list)
print("list is: ", list)
print("The largest number in the given List is: ", max_value)
list.sort()
print("The second largest number in the given list is: ", list[-2])