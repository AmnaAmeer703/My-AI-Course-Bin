# Calculate Body Max Index
# Input weight(kg) and height(m),then calculate BMI = weight/(height**2)

# weight = w, height = h

w = float(input("Enter The weight: "))
h = float(input("Enter The Height: "))
BMI = w/(h**2)

print("The BMI is: ", BMI)