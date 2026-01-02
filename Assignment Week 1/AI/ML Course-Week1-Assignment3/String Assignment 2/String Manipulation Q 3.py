# This is a Python Program to display which letters are in the two strings but not in both

string1 = input("Please Enter The First Word: ")
string2 = input("Please Enter The Second Word: ")

set1 = set(string1)
set2 = set(string2)

a = (set1)^(set2)
print("The Letters which in two strings but not in both are: ")
for i in a:
    print(i, end = " ")