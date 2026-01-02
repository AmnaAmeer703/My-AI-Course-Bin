# Python Program to Print Largest Even and Largest Odd Number in a List. The program takes in a 
# list and prints the largest even and largest off number in it.

n = int(input("Enter The Number of Elements to be in the list: "))
list = []
for i in range(0,n):
    a = int(input("Enter The Element: "))
    list.append(a)
c = []
d = []
for i in list:
    if (i%2 == 0):
        c.append(i)
    else:
        d.append(i)
c.sort()
d.sort()
count1=0
count2=0
for i in c:
    count1 = count1 + 1
for j in d:
    count2 = count2 + 1
list.sort()
print("The List is: ", list)
print("Largest Even Number is: ", c[count1-1])
print("Largest Odd Number is: ", d[count2-1])
