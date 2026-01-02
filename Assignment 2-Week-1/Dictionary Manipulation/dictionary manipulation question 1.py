# Check if a value exists in a dictionary

Student_Dictionary = {"Student1":"Amna","Student2":"Fatima","Student3":"Javeria","Student4":"Zainab","Student5":"Shama"}

# Check if "Sarah" exit in a dictionary
Student_Name = "Sarah"

if Student_Name in Student_Dictionary.values():
    print("The required student exist in a Student Dictionary")
else:
    print("The required student does not exist in a Student Dictionary")