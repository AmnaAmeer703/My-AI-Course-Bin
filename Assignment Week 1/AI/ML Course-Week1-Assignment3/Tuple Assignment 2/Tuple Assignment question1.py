# Python Program to Create a List of Tuples with the First Element as the Number and Second 
# Element as the Square of the Number. The program takes a range and creates a list of tuples 
# within that range with the first element as the number and the second element as the square of 
# the number.

x = int(input("Enter The First Number: "))
y = int(input("Enter The Second Number: "))

output = [(x,x**2) for x in range(x,y+1)]
print(output)