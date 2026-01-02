# Take input from user and print prime numbers till that input number

n = int(input("Enter The Number: "))
print("Prime Numbers Upto Your Enter Number are: ")

for x in range(2,n):
    for y in range(2,x):
        if x % y == 0:
            break
    else:
        print(x, end = " ")