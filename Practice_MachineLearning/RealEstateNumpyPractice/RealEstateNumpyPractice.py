import numpy as np

brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, previous_sold_date = np.genfromtxt("\NumpyPractice\RealEstate-USA.csv", delimiter=",", usecols=(0,1,2,3,4,5,6,7,8,9,10,11), unpack=True,dtype=None,skip_header=1)

print(brokered_by)
print(status)
print(price)
print(bed)
print(bath)
print(acre_lot)
print(street)
print(city)
print(house_size)
print(previous_sold_date)

print("Real Estate Price mean: " , np.mean(price))
print("Real Estate Price average: " , np.average(price))
print("Real Estate Price std: " , np.std(price))
print("Real Estate Price median: " , np.median(price))
print("Real Estate Price percentile - 25: " , np.percentile(price,25))
print("Real Estate percentile  - 75: " , np.percentile(price,75))
print("Real Estate Price percentile  - 3: " , np.percentile(price,3))
print("Real Estate Price min : " , np.min(price))
print("Real Estate Price max : " , np.max(price))

print("Real Estate Price square: " , np.square(price))
print("Real Estate Price sqrt: " , np.sqrt(price))
print("Real Estate Price pow: " , np.power(price,price))
print("Real Estate Price abs: " , np.abs(price))

addition = bed + bath
subtraction = bed - bath
multiplication = bed * bath
division = bed / bath

print(" Real Estate - bed and bath - Addition:", addition)
print(" Real Estate - bed and bath - Subtraction:", subtraction)
print(" Real Estate - bed and bath - Multiplication:", multiplication)
print(" Real Estate - bed and bath - Division:", division)

pricePie = (price/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("Real Estate Price - div - pie  - Sine values:", sine_values)
print("Real Estate Price - div - pie Cosine values:", cosine_values)
print("Real Estate Price - div - pie Tangent values:", tangent_values)

print("Real Estate Price - div - pie  - Exponential values:", np.exp(pricePie))


log_array = np.log(pricePie)
log10_array = np.log10(pricePie)

print("Real Estate Price - div - pie  - Natural logarithm values:", log_array)
print("Real Estate Price - div - pie  = Base-10 logarithm values:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(pricePie)
print("Real Estate Price - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(pricePie)
print("Real Estate Price - div - pie   - Hyperbolic Cosine values:", cosh_values)


tanh_values = np.tanh(pricePie)
print("Real Estate Price - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(pricePie)
print("Real Estate Price - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(pricePie)
print("Real Estate Price - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)

D2 = np.array([price,
                  house_size])
print(D2)

print ("Real Estate Price and House Size - 2 dimentional arrary - " ,D2)

# check the dimension of array1
print("Real Estate Price and House Size - 2 dimentional arrary - dimension" , D2.ndim) 
# Output: 2

# return total number of elements in array1
print("Real Estate Price and House Size - 2 dimentional arrary - total number of elements" ,D2.size)
# Output: 400

# return a tuple that gives size of array in each dimension
print("Real Estate Price and House Size - 2 dimentional arrary - gives size of array in each dimension" ,D2.shape)

print("Real Estate Price and House Size - 2 dimentional arrary - data type" ,D2.dtype) 

# Splicing array
D2Price_HouseSizeSlice=  D2[:1,:5]
print("Real Estate Price and House Size - 2 dimentional arrary - Splicing array - D2[:1,:5] " , D2Price_HouseSizeSlice)
D2Price_HouseSizeSlice2=  D2[:1, 4:15:4]
print("Real Estate Price and House Size - 2 dimentional arrary - Splicing array - D2[:1, 4:15:4] " , D2Price_HouseSizeSlice2)


# Indexing array
D2Price_HouseSizetSliceItemOnly=  D2Price_HouseSizeSlice[0,1]
print("Zameen.com Long Plus Lat - 2 dimentional arrary - Index array - D2LongLatSlice[1,5] " , D2Price_HouseSizetSliceItemOnly)
D2Price_HouseSizeSlice2ItemOnly=  D2Price_HouseSizeSlice2[0, 2]
print("Zameen.com Long Plus Lat - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " , D2Price_HouseSizeSlice2ItemOnly)


#You should use the builtin function nditer, if you don't need to have the indexes values.
for elem in np.nditer(D2):
    print(elem)

#EDIT: If you need indexes (as a tuple for 2D table), then:
for index, elem in np.ndenumerate(D2):
    print(index, elem)



# 2 x 149 ========>>>>> 1  x 400 - reshape
D2Reshape = np.reshape(D2, (1, 400))
print("Real Estate Price and House Size - 2 dimentional arrary - np.reshape(D2, (1, 400)) : " , D2Reshape)
print("Real Estate Price and House Size - 2 dimentional arrary - np.reshape(D2, (1, 400)) : Size " , D2Reshape.size)
print("Real Estate Price and House Size - 2 dimentional arrary - np.reshape(D2, (1, 400)) : ndim " , D2Reshape.ndim)
print("Real Estate Price and House Size - 2 dimentional arrary - np.reshape(D2, (1, 400)) : shape " , D2Reshape.shape)
print("Real Estate Price and House Size - 2 dimentional arrary - np.reshape(D2, (1, 400)) : ndim " , D2Reshape.ndim)




print()