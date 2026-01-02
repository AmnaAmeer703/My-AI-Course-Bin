# Write a program to create a new string made of an input string's first, second and last character

input_string = input("Enter The Input String: ")

# string first character
first_char = input_string[0]

# length of string
length = len(input_string)

# string middle index
middle_index = int(length/2)
output = first_char + input_string[middle_index]
# #string last character
last_char = input_string[-1]

New_string = output + last_char

print("The New Converted String is: ",New_string)

