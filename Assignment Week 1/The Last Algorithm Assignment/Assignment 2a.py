# Write a program , to have “List” , with all “noun” in story. Print them.

# We select a random sentence from the AI Fiction Story "The Last Algrothim" and make a program on it

# Selected random Sentence "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it "Athena-9"—the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."

def find_nouns(text):
        
    # Create a set of common English "stop words" and function words to ignore.
    # This helps filter out many common non-nouns.
    stop_words = {
        "a", "an", "the", "and", "but", "or", "for", "nor", "so", "yet",
        "of", "in", "on", "at", "with", "from", "to", "is", "am", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "not", "no", "he", "she", "it", "they", "we", "you",
        "I", "me", "him", "her", "us", "them", "my", "your", "his", "her",
        "its", "our", "their", "this", "that", "these", "those", "can",
        "will", "would", "shall", "should", "may", "might", "must", "if",
        "then", "when", "where", "how", "what", "which", "who", "whom"
    }

    # Remove punctuation and split the text into words.
    words = text.lower().replace(",", "").replace(".", "").split()
    
    potential_nouns = []
    
    # Iterate through the words and filter out those in the stop_words set.
    for word in words:
        if word not in stop_words:
            # Add the word to the potential nouns list if it is not a stop word.
            potential_nouns.append(word)
            
    # As an optional step, remove duplicates and return as a list.
    return sorted(list(set(potential_nouns)))


story = input("Enter The Story: ")
         
nouns_list = find_nouns(story)

print("The story contains these potential nouns:")
print(nouns_list)