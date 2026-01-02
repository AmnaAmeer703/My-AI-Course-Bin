# Python Program to Find the Union of Two Lists. The program takes two lists and finds the unions 
# of the two lists.

list1 = []
n = int(input('Enter size of list 1: '))
for x in range(n):
    value_1 = int(input('Enter any number:'))
    list1.append(value_1)
 
list2 = []
m = int(input('Enter size of list 2:'))
for x in range(m):
    value_2 = int(input('Enter any number :'))
    list2.append(value_2)
 
union = list(set().union(list1,list2))
 
print('The Union of two lists is: ',union)