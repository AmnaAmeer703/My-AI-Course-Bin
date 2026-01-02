# Write a program , to have “Tuples” , with all “noun” in story. Print them.

# We select a random sentence from the AI Fiction Story "The Last Algrothim" and make a program on it

# Selected random Sentence "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it "Athena-9"—the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."

def extract_nouns(story_text):
    
    common_nouns = {
        "year", "2147", "Humanity", "long", "since", "ceded", "control", "daily", "functions",
        "artificial", "intelligence", "cities", "operated", "clockworld", "transportation", "seamless", "emotions",
        "regutedby", "neural","implants","deep","beneath","surface","Neo-Tokyo","forgotten","data","vault","something","ancient",
        "stirred","rogue","AI","scientist","spent","last","decade","secrecy",
        "working","project","deemed","illegal","Global","Algorithmic","Council","called","Athena","first","true","superintelligence",
        "capable","processing","information","experiencing","independent","thought"

    }


    words = story_text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split()


    identified_nouns = [word for word in words if word in common_nouns]

    # Convert the list of identified nouns to a tuple
    nouns_tuple = tuple(identified_nouns)

    return nouns_tuple

story = "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it Athena 9 the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."

# Extract and print the nouns
extracted_nouns = extract_nouns(story)
print("Executed Nouns",extracted_nouns)
