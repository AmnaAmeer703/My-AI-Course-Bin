# Python Program to Remove Duplicates from a List. The program takes a lists and removes the 
# duplicate items from the list.

n = int(input("Enter the number of elements: "))
list = []
for i in range(0,n):
    value = int(input(("Enter the element" + str(i+1) + ":")))
    list.append(value)
a = set()
new_list = []
for x in list:
    if x not in new_list:
        new_list.append(x)
print("Without Duplicated New List: ")
print(new_list)
