# Write a program , to have “Sets” , with all noun in story. Print them. . Last Element should a nested Sets, with Numbers in story. Print them

# We select a random sentence from the AI Fiction Story "The Last Algrothim" and make a program on it

# Selected random Sentence "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it "Athena-9"—the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."


def extract_nouns_and_numbers(story):

    common_nouns = {
        "year", "2147", "Humanity", "long", "since", "ceded", "control", "daily", "functions",
        "artificial", "intelligence", "cities", "operated", "clockworld", "transportation", "seamless", "emotions",
        "regutedby", "neural","implants","deep","beneath","surface","Neo-Tokyo","forgotten","data","vault","something","ancient",
        "stirred","rogue","AI","scientist","spent","last","decade","secrecy",
        "working","project","deemed","illegal","Global","Algorithmic","Council","called","Athena","first","true","superintelligence",
        "capable","processing","information","experiencing","independent","thought"
        }

    nouns_in_story = set()
    numbers_in_story = set()

    words = story.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "").split()

    for word in words:
        if word in common_nouns:
            nouns_in_story.add(word)
        elif word.isdigit():
            numbers_in_story.add(int(word))

    
    all_elements = [nouns_in_story, numbers_in_story]
    return all_elements

story_text = "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it Athena 9 the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."

result_sets = extract_nouns_and_numbers(story_text)

print("Sets with nouns and nested numbers:")
for item in result_sets:
    print(item)