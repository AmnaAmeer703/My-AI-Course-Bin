# Write a program , to have “Dictionaries” , with all noun in story. Print them. Last Element should a nested Dictionaries, with Numbers in story. Print them.

# We select a random sentence from the AI Fiction Story "The Last Algrothim" and make a program on it

# Selected random Sentence "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it "Athena-9"—the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."

import re

def process_story(story_text):

    common_nouns = {
        "year", "2147", "Humanity", "long", "since", "ceded", "control", "daily", "functions",
        "artificial", "intelligence", "cities", "operated", "clockworld", "transportation", "seamless", "emotions",
        "regutedby", "neural","implants","deep","beneath","surface","Neo-Tokyo","forgotten","data","vault","something","ancient",
        "stirred","rogue","AI","scientist","spent","last","decade","secrecy",
        "working","project","deemed","illegal","Global","Algorithmic","Council","called","Athena","first","true","superintelligence",
        "capable","processing","information","experiencing","independent","thought"

    }

    words = re.findall(r'\b\w+\b|\d+', story_text.lower())
    
    nouns_dict = {}
    numbers_dict = {}
    
    for word in words:
        if word in common_nouns:
            if word in nouns_dict:
                nouns_dict[word] += 1
            else:
                nouns_dict[word] = 1
        # Check if the word is a number.
        elif word.isdigit():
            # Add to the numbers nested dictionary.
            # Using a list allows for storing all occurrences.
            numbers_dict.setdefault("numbers_in_story", []).append(int(word))

    # Add the numbers dictionary as the last element of the nouns dictionary.
    # This overwrites any existing key and ensures it is last.
    nouns_dict["numbers"] = numbers_dict

    return nouns_dict

story = "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it Athena 9 the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."


story_dictionaries = process_story(story)

# Print the final dictionary.
print("Dictionary with all noun and Last element in nested dictionary:", story_dictionaries)