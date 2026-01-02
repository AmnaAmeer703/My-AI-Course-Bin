# Python Program to Check If Two Numbers are Amicable Numbers or Not[The program takes two numbers and checks if they are amicable numbers.] Amicable numbers are pairs of different numbers where the sum of the proper divisors (divisors excluding the number itself) of one number equals the other number, and vice versa. The smallest example is 220 and 284.

x = int(input("Enter The First Number: "))
y = int(input("Enter The Second Number: "))

sum = 0
sum1 = 0
for i in range(1,x):
    if (x % i == 0):
        sum = sum+i
for j in range(1,y):
    if (y % j == 0):
        sum1 = sum1+j
if (sum == y and sum1 == x):
    print("Given Numer",x,"and",y,"are Amicable Pair")
else:
    print("Given Number",x,"and",y,"are Not Amicable Pair")