# Python Program to Print All Permutations of a String in Lexicographic Order without Recursion. The problem is the display all permutations of a string in lexicographic or dictionary order.

def next_permutation(s):
    s = list(s)
    n = len(s)
    i = n - 2
    while i >= 0 and s[i] >= s[i+1]:
        i -= 1
    if i == -1:
        return False
    j = n - 1
    while s[j] <= s[i]:
        j -= 1
    s[i], s[j] = s[j], s[i]
    s = s[:i + 1] + s[i + 1:][::-1]
    return ''.join(s)
def permutation_in_laxicographic_order(string):
    string = ''.join(sorted(string))
    print(string)
    while True:
        string = next_permutation(string)
        if not string:
            break
        print(string)
string = input("Enter The string: ")
print("permutation in laxicographic order:")
permutation_in_laxicographic_order(string)