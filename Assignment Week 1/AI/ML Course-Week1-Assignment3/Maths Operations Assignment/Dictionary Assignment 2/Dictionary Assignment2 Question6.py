# Python Program to Concatenate Two Dictionaries. The program takes two dictionaries and 
# concatenates them into one dictionary.

d1 = {"a":10,"b":20,"c":30,"d":40,"e":50}
d2 = {"f":60,"g":70,"h":80,"i":90,"j":100}
# To concatenate d1 and d2
d1.update(d2)
print(d1)