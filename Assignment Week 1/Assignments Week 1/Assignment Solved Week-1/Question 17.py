# Question 17: Convert Minutes to Hours and Minutes.
# Input number of minutes and convert to hours and remaining minutes

Minutes = int(input("Enter The Minutes: "))
Hour = Minutes // 60
Min = Minutes % 60
print("Hour:",Hour,"Minutes:",Min)