# Python Program to check if a string is a aprogram or not [The Program take a string and checks if it is a program or not]
def check(string):
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    for character in  alphabets:
        if character not in string.lower():
            return False
        return True


string = input("Please Enter The Sentence: ")
if (check(string) == True):
    print("The Enter String is Pangram")
else:
    print("The Enter String is not Pangram")

