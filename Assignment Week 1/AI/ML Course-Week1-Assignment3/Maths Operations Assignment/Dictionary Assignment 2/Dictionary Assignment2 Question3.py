# Python Program to Find the Sum of All the Items in a Dictionary The program takes a dictionary 
# and prints the sum of all the items in the dictionary.

d = {"a":12,"b":20,"c":30,"d":40,"e":50,"f":60,"g":70,"h":80,"i":90,"j":100}

sum = 0
for i in d.values():
    sum +=  i
print(sum)