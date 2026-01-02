# Question 10: Salary Calculator. Input basic salary. Calculate
# HRA = 20% of basic
# DA = 15% of basic
# Total Salary = Basic + HRA + DA

Basic_Salary = int(input("Enter The Basic Salary: "))
HRA = Basic_Salary * 20/100
DA = Basic_Salary * 15/100

Total_Salary = Basic_Salary + HRA + DA
print("Total Salary Is: ", Total_Salary)