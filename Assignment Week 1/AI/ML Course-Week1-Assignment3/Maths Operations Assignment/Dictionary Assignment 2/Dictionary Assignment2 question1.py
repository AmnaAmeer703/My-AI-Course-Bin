# Python Program to Check if a Key Exists in a Dictionary or Not[This is a Python Program to check 
#if a given key exists in a dictionary or not

d = {"Amna":30,"Zainab":5,"Shama":60,"Javeria":40,"Fauzia":55,"Fatima":10}
user_id = input("Please Enter The Key You Want to Check: ")

if user_id in d:
    print("Key is present")
    print("The Value is: ", d[user_id])
else:
    print("Key is Not present")