# Python Program to Find All Perfect Squares in the Given Range.[ The program takes a range and creates a list of all numbers in the range which are perfect squares and the sum of the digits is 
# less than 10.] To find perfect squares within a range, identify the smallest and largest integers 
# whose squares fall within that range, then list the squares of those integers.  
# Example: 
# Range: 1 to 100 
# Smallest integer: 1 (1 * 1 = 1) 
# Largest integer: 10 (10 * 10 = 100) 
# Perfect Squares: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100

start = int(input("Enter The Start Of The Range: "))
end= int(input("Enter The End Of The Range: "))

while start <= end:
    for i in range(1,start):
        if i*i == start:
            print(start, end=",")
    start += 1