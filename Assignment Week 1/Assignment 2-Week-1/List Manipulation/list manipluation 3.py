# Remove empty string from the list of string

Names = ["Amna","Zainab","","Fatima","","Javeria","Shama","","Fauzia"]

for i in Names:
    if i == "":
        Names.remove(i)

print(Names)