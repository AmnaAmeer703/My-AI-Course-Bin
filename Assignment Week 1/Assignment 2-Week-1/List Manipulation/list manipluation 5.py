# Replaces list item with new value if found

Name_list = ["Amna","Zainab","Shama","Javeria","Fauzia","Sarah","Farwa","Khadija"]
 
# Replace the name "Sarah" if name "Fatima" found


for name in range(len(Name_list)):
    if Name_list[name] == "Sarah":
        Name_list[name] = "Fatima"

print(Name_list)
