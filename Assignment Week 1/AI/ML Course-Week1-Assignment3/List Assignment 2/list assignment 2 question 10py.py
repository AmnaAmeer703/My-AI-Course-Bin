# Python Program to Return the Length of the Longest Word from the List of Words. The program 
# takes a list of words and returns the word with the longest length.

list =[]
n= int(input("Enter the number of elements in list: "))
for x in range(0,n):
    word=input("Enter word" + str(x+1) + ":")
    list.append(word)
max1=len(list[0])
result = list[0]
for i in list:
    if(len(i)>max1):
       max1=len(i)
       result = i
print("The word with the longest length is:")
print(result)