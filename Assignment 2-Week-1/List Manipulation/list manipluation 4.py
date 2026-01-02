# Add new item to list after a specified item

Names = ["Amna","Zainab","Shama","Fatima","Sarah","Javeria"]

# if want to add another name "Fauzia" in a list after the name "Sarah"
specified_name = "Sarah"
index_of_specified_name = Names.index(specified_name)
insertion_index = index_of_specified_name + 1

New_Name = "Fauzia"

Names.insert(insertion_index,New_Name)

print(Names)
