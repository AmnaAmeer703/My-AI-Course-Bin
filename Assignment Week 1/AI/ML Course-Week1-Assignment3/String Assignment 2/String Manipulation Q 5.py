# Python Program to Count Number of Uppercase and Lowercase Letters in a String[The program takes a string and counts the number of lowercase letters and uppercase letters in the string.

Message = input("Please Enter The Message: ")

uppercase = 0
lowercase = 0

for i in Message:
    if i.isupper():
        uppercase += 1
    elif i.islower():
        lowercase += 1
print("Count Of Uppercase ia Message are: ", uppercase)
print("Count Of Lowercase in a Message are: ", lowercase)