# Python Program to Find the LCM of Two Numbers [The program takes two numbers and prints the LCM of two numbers

x = int(input("Enter The First Number: "))
y = int(input("Enter The Second Number: "))

if x > y:
    smaller = y
else:
    smaller = x
for i in range(1,smaller+1):
    if(x % i == 0) and (y % i ==0):
        hcf = i
lcm = (x*y)/hcf
print("The LCM of The Two Numbers are: ", lcm)