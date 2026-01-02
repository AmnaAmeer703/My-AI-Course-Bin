# Python Program to Check if the Substring is Present in the Given String. [The program takes a string and checks if a substring is present in the given string

String = input("Please Enter The Message: ")
sub_string = input("Please Enter The Message: ")

if sub_string.lower() in String.lower():
    print("Yes Substring is present in a String")
else:
    print("No Substring is Not Present in a String")