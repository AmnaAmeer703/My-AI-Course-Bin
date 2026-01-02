# Python Program to Remove All Tuples in a List Outside the Given Range. The program removes all tuples in a list of tuples with the USN outside the given range. 
#Problem Solution 
# 1. Take in the lower and upper roll number from the user. 
# 2. Then append the prefixes of the USN’s to the roll numbers. 
# 3. Using list comprehension, find out which USN’s lie in the given range. 
# 4. Print the list containing the tuples. 
# 5. Exit.

y = [("amna","a123"),("zainab","a124"),("javeria","a125"),("shama","a126")]
lower_roll_no = int(input("Enter The Lower Roll Number: "))
upper_roll_no = int(input("Enter The Upper Roll Number: "))

low = "a123" + str(lower_roll_no)
upper = "a12" + str(upper_roll_no)

output = [x for x in y if x[1]>low and x[1]<upper]
print(output)

