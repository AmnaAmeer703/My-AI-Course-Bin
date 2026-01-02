# Python Program to Check if Two Strings are Anagram. An anagram in Python is a pair of strings that have the same characters, but in a different order. It involves rearranging the letters of one string to form the other.

string1 = input("Please Enter First Word: ")
string2 = input("Please Enter Secod Word: ")

if len(string1) == len(string2):
    sorted_sring1 = sorted(string1)
    sorted_sring2 = sorted(string2)
    if sorted_sring1 == sorted_sring2:
        print("It is Anagram")
    else:
        print("It is not Anagram")