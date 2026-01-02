# Turn every item in a alist into its square

list = [10,20,30,40,50,60,70,80,90,100]

square_of_list = []

for i in list:
    square_of_list.append(i * i)
print("Orignal list is: ", list)
print("The Square of list is: ", square_of_list)