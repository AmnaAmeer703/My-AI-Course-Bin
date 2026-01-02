# Python Program to Count Occurrences of Element in List. The program takes a number and 
# searches the number of times the particular number occurs in a list.

list = []
n = int(input("Enter The Number of Elements: "))

for i in range(1,n+1):
    value = int(input("Enter The Element: "))
    list.append(value)
list.sort()
print("List: ", list)
count = 0
m = int(input("Enter the number to be counted: "))
for j in list:
    if(j == m):
        count = count + 1
print("Number",m,"occurs",count,"times")