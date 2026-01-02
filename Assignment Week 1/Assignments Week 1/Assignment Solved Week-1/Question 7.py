# Question 7: Distribute Items Equally - You have n candies and k students.
# Write a program to find:
# how many candies each student gets
# how many are left

# let total number of candies denote by 'n' and total number of students denote by 'k'

n = int(input("Enter Total Number Of Candies: "))
k = int(input("Enter Total Number Of Students: "))

candies_each_student_get = n // k

candies_left_behind = n % k

print("Number Of Candies Each Student Get: ", candies_each_student_get)
print("Candies Left Behind: ", candies_left_behind)

