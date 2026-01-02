# Python Program to Print an Identity Matrix [The program takes a number n and prints an identity matrix of the desired size.

number = int(input("Enter The Number: "))

for i in range(0,number):
    for j in range(0,number):
        if i == j:
            print("1", sep = " ",end = " ")
        else:
            print("0", sep = " ",end=" ")
    print()