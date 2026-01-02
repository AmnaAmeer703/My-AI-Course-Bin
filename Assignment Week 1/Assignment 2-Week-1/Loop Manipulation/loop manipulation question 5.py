# Print multiplication table of a given number

X = int(input("Enter the number: "))
print("This is the multiplication table: ")
i = 1
while i <= 10:
    print(X,"x",i,"=",X*i)
    i = i + 1