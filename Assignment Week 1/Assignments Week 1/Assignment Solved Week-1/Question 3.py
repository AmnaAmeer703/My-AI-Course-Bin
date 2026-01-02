
# Question 3 : Claculate Compund interest Formula (CI = P * (1 + R/100)**T-P)
# P = Principal, T = time, R = Rate

P = int(input("Enter The Value Of Principal: "))
T = int(input("Enter The Value Of Time: "))
R = int(input("Enter The Value Of Rate: "))

CI = P * (1 + R/100)**T-P

print("The Compound Interest is: ", CI)