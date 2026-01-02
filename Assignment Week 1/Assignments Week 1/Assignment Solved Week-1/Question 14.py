# Question 14: Percentage Of Correct Answers
# Input total questions and correct answers and calculate the percentage score

Total_Question = int(input("Enter The Number Of Total Question: "))
Correct_Answers = int(input("Enter The Number Of Correct answer: "))

Percentage_Score = (Correct_Answers/Total_Question)*100

print("The Percentage Score Is: ",Percentage_Score)