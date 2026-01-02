#Write a program , to have “Tuples” , with all “noun” in story. Print them. Last Element should a nested Tuples, with Numbers in story. Print them

# We select a random sentence from the AI Fiction Story "The Last Algrothim" and make a program on it

# Selected random Sentence "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it "Athena-9"—the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."

def extract_nouns_and_numbers(story):
    
    words = story.replace('.', '').replace(',', '').replace(';', '').lower().split()
    nouns = []
    numbers = []

    
    common_nouns = {
        "year", "2147", "Humanity", "long", "since", "ceded", "control", "daily", "functions",
        "artificial", "intelligence", "cities", "operated", "clockworld", "transportation", "seamless", "emotions",
        "regutedby", "neural","implants","deep","beneath","surface","Neo-Tokyo","forgotten","data","vault","something","ancient",
        "stirred","rogue","AI","scientist","spent","last","decade","secrecy",
        "working","project","deemed","illegal","Global","Algorithmic","Council","called","Athena","first","true","superintelligence",
        "capable","processing","information","experiencing","independent","thought"
        }
    for word in words:
        if word.isdigit():
            numbers.append(int(word))
        elif word in common_nouns:
            nouns.append(word)
        
    result_tuple = tuple(nouns) + (tuple(numbers),)
    return result_tuple

story_text = "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it Athena 9 the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."

extracted_data = extract_nouns_and_numbers(story_text)

print(f"Extracted Tuples: {extracted_data}")

print(f"Last Element (Numbers): {extracted_data[-1]}")