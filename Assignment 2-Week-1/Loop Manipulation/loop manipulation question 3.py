# Take input from user, and print odd number till that input number

user_input = int(input("Enter the number: "))

print("All Odd Numbers Upto Your Enter Number are: ")

for i in range(1,user_input+1,2):
        print(i,end = " ")