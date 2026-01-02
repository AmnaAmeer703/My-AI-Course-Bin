# Delete a list of keys from a dictionary

Students = {1:"Amna",2:"Zainab",3:"Shama",4:"Javeria",5:"Fauzia",6:"Fatima",7:"Sarah",8:"Khadija",9:"Minahil",10:"Nazia",11:"Usman",12:"Salman"}

keys_to_delete = [8,9,12]

for keys in keys_to_delete:
    if keys in Students:
        del Students[keys]

print(Students)