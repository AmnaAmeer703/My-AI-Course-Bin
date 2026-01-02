# Get the key of a minimum value from the following dictionary

Student_marks = {"Amna":80,"Zainab":95,"Shama":80,"Javeria":79,"Fauzia":60,"Sarah":45}

student_who_get_minimum_marks = min(Student_marks,key=Student_marks.get)
print("The Student who get minimum marks is:",student_who_get_minimum_marks)
