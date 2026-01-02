# Question 8: Calculate profit and Loss
# Input cost price and selling price. display either:
# Profit and amount,
# Loss or ammount,
# No profit no loss

Cost_price = float(input("Enter Cost Price: "))
Selling_price = float(input("Enter Selling price: "))

if Selling_price > Cost_price:
    Profit = Selling_price - Cost_price
    print("Profit and amount: ", Profit)
elif Cost_price > Selling_price:
    Loss = Cost_price - Selling_price
    print("Loss or amount: ", Loss)
else:
    print("No Profit No Loss")