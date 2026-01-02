# Python Program to Create Dictionary that Contains Number. The program takes a number from 
#the user and generates a dictionary that contains numbers (between 1 and n) in the form 
#(x,x*x).

n = int(input("Please Enter The Number: "))
d = {}

for x in range(1,n+1):
    d[x] = x*x

print("The generated dictionary is: ", d)