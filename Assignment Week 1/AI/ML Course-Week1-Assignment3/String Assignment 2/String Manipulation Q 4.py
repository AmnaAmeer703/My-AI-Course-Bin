# Python Program to Find the Larger String without using Built-in Functions.The program takes in two strings and display the larger string without using built-in function.

message1 = input("Please Enter The First Message: ")
message2 = input("Please Enter The Second Message: ")

count1 = 0
count2 = 0

for i in message1:
    count1+= 1
for j in message2:
    count2+= 1
if count1<count2:
    print("The largest Message is Message 2: ", message2)
else:
    print("The largest Message is Message 1: ", message1)