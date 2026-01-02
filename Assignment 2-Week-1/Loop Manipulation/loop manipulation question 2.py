# Take input from user, and print even number till that input number

user_input = int(input("Enter The Number: "))

print("All Even Number Upto Your Enter Number are: ")

for i in range(2,user_input+2,2):
        print(i,end= " ")