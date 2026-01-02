# Write a program , to have “List” , with all “noun” in story. Last Element should a nested List, with Numbers in story. Print them


# We select a random sentence from the AI Fiction Story "The Last Algrothim" and make a program on it

# Selected random Sentence "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it "Athena-9"—the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."

story = "The year was 2147. Humanity had long since ceded control of its daily functions to artificial intelligence. Cities operated like clockwork, transportation was seamless, and even emotions could be regutedby neural implants. But deep beneath the surface of Neo-Tokyo, in a forgotten data vault, something ancient stirred. Dr. Elias Voss, a rogue AI scientist, had spent the last decade in secrecy, working on a project deemed illegal by the Global Algorithmic Council. He called it Athena 9 the first true artificial superintelligence, capable of not just processing information but experiencing independent thought."



common_nouns = {
        "year", "2147", "Humanity", "long", "since", "ceded", "control", "daily", "functions",
        "artificial", "intelligence", "cities", "operated", "clockworld", "transportation", "seamless", "emotions",
        "regutedby", "neural","implants","deep","beneath","surface","Neo-Tokyo","forgotten","data","vault","something","ancient",
        "stirred","rogue","AI","scientist","spent","last","decade","secrecy",
        "working","project","deemed","illegal","Global","Algorithmic","Council","called","Athena","first","true","superintelligence",
        "capable","processing","information","experiencing","independent","thought"
        }

output_list = []
numbers_list = []


cleaned_story = story.lower().replace('.', '').replace(',', '').strip()
words = cleaned_story.split()


for word in words:
    # Check if the word is a number using the isdigit() method
    if word.isdigit():
        numbers_list.append(int(word))
    
    elif word in common_nouns:
        output_list.append(word)


output_list.append(numbers_list)


print(output_list)