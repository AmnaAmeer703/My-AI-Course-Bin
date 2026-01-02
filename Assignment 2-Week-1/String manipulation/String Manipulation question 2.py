# Write a aprogram to count occurance of all characters with in a string given:

string = "Nexskillbeproductive"
character_count = {}

for i in string:
    if i in character_count:
        character_count[i] = character_count[i] + 1
    else:
        character_count[i] = 1
print("Number Of Occuracne Of Each Character in a string is: ",character_count)