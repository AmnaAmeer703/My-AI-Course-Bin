# Total Marks and Percentage
# Input marks of 5 Subject Print
# Total Marks
# Percentage
# Average

a = int(input("Enter The Marks Of Maths: "))
b = int(input("Enter The Marks Of Stats: "))
c = int(input("Enter The Of Computer Science: "))
d = int(input("Enter The Marks Of English: "))
e = int(input("Enter The Marks Of Python Language: "))

Total_Marks = 500
Sum = a+b+c+d+e
Average = (a+b+c+d+e)/5
Percentage = (Sum/Total_Marks)*100

print("Total Marks Are: ", Total_Marks)
print("Average:", Average)
print("Percentage Is: ", Percentage)