# Python Program to Create a New String Made up of First and Last 2 Characters. The program takes a string and forms a new string made of the first 2 characters and last 2 characters from a given string.

String = input("Please Enter The Message: ")

New_String = String[:2] + String[-2:]

print("The New String is: ", New_String)