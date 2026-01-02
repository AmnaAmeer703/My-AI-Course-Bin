import numpy as np

address,city,country,keys,latitude,longitude,name,postalCode,province,websites = np.genfromtxt("NumpyPractice\RealEstateNumpyPractice\FastFoodRestaurants.csv", delimiter=",", usecols=(0,1,2,3,4,5,6,7,8,9),dtype=('U100','U100','U100','U100','f8','f8','U100','f8','U100','U100'),
                                                                                               encoding='utf-8',unpack=True,skip_header=1,invalid_raise=False)

print(address)
print(city)
print(country)
print(keys)
print(latitude)
print(longitude)
print(name)
print(postalCode)
print(province)
print(websites)



addition = latitude + longitude
subtraction = latitude - longitude
multiplication = latitude * longitude
division = latitude / longitude

print(" Fast Food Restaurants - latitude and longitude - Addition:", addition)
print(" Fast Food Restaurants - latitude and longitude - Subtraction:", subtraction)
print(" Fast Food Restaurants - latitude and longitude - Multiplication:", multiplication)
print(" Fast Food Restaurants - latitude and longitude - Division:", division)

latitudePie = (latitude/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(latitudePie)
cosine_values = np.cos(latitudePie)
tangent_values = np.tan(latitudePie)

print("Fast Food Restaurants - latitude - div - pie  - Sine values:", sine_values)
print("Fast Food Restaurants - latitude - div - pie Cosine values:", cosine_values)
print("Fast Food Restaurants - latitude - div - pie Tangent values:", tangent_values)

print("Fast Food Restaurants - latitude - div - pie  - Exponential values:", np.exp(latitudePie))


log_array = np.log(latitudePie)
log10_array = np.log10(latitudePie)

print("Fast Food Restaurants - latitude - div - pie  - Natural logarithm values:", log_array)
print("Fast Food Restaurants - latitude - div - pie  = Base-10 logarithm values:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(latitudePie)
print("Fast Food Restaurants - latitude - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(latitudePie)
print("Fast Food Restaurants - latitude - div - pie   - Hyperbolic Cosine values:", cosh_values)


tanh_values = np.tanh(latitudePie)
print("Fast Food Restaurants - latitude - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(latitudePie)
print("Fast Food Restaurants - latitude - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(latitudePie)
print("Fast Food Restaurants - latitude - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)

D2 = np.array([latitude,
                  longitude])
print(D2)

print ("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - " ,D2)

# check the dimension of array1
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - dimension" , D2.ndim) 
# Output: 2

# return total number of elements in array1
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - total number of elements" ,D2.size)
# Output: 400

# return a tuple that gives size of array in each dimension
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - gives size of array in each dimension" ,D2.shape)

print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - data type" ,D2.dtype) 

# Splicing array
D2latitide_longitudeSlice=  D2[:1,:5]
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - Splicing array - D2[:1,:5] " , D2latitide_longitudeSlice)
D2latitude_longitudeSlice2=  D2[:1, 4:15:4]
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - Splicing array - D2[:1, 4:15:4] " , D2latitude_longitudeSlice2)


# Indexing array
D2latitude_longitudeSliceItemOnly=  D2latitide_longitudeSlice[0,1]
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - Index array - D2LongLatSlice[1,5] " , D2latitude_longitudeSliceItemOnly)
D2latitude_longitudeSlice2ItemOnly=  D2latitude_longitudeSlice2[0, 2]
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " , D2latitude_longitudeSlice2ItemOnly)


for elem in np.nditer(D2):
    print(elem)

for index, elem in np.ndenumerate(D2):
    print(index, elem)



D2Reshape = np.reshape(D2, (1, 400))
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - np.reshape(D2, (1, 400)) : " , D2Reshape)
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - np.reshape(D2, (1, 400)) : Size " , D2Reshape.size)
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - np.reshape(D2, (1, 400)) : ndim " , D2Reshape.ndim)
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - np.reshape(D2, (1, 400)) : shape " , D2Reshape.shape)
print("Fast Food Restaurants - latitude and longitude - 2 dimentional arrary - np.reshape(D2, (1, 400)) : ndim " , D2Reshape.ndim)




print()