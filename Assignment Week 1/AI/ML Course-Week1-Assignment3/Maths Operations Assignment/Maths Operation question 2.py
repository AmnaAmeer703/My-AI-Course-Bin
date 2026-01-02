# Python Program to Find Quotient and Remainder of Two Numbers[The program takes two numbers and prints the quotient and remainder.

x = int(input("Enter The Value Of Dividend: "))
y = int(input("Enter The Value Of Divisor: "))

quotient = x // y
remainder = x % y

print("Quotient is: ", quotient)
print("Remainder is: ", remainder)