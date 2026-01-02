# Question 12: Currency Converter (USD to PKR)
# Input amount in USD. Convert using a fixed Exchange rate

# The Current Fixed Exchange rate of 1 PKR to USD is 0.0035 (resource: Western Union Website current rate)

PKR = 0.0035
Amount_In_USD = float(input("Please Enter Amount In USD: "))

Converted_Amount = Amount_In_USD * PKR

print("Yout Converted Amount in PKR is: ", Converted_Amount)

