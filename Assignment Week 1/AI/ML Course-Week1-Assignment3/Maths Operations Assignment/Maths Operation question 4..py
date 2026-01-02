# Python Program to Find the LCM of Two Numbers [The program takes two numbers and prints the LCM of two numbers

x = int(input("Enter The First Number: "))
y = int(input("Enter The Second Number: "))

z = x*y

for i in range(1,z+1):
    if i%x == 0 and i%y == 0:
        print("The LCM Of Two Given Numbers are: ", i)
        break