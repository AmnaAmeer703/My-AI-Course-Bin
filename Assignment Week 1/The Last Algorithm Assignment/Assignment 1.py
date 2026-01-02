# Write a program, to list all words, with vowel in it

Title = "The Last Algorithm"
# We select a random sentence from the AI Fiction Story "The Last Algrothim" and make a program on it

# Selected random Sentence "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it "Athena-9"—the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."

sentence = input("Enter The Sentence: ")
vowels = "aeiouAEIOU"

print("All The Words With Vowels in it: ")
for word in sentence.split(" "):
    if word[0] in vowels:
        print(word, end=" ")