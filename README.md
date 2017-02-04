# LSH
A simple implementation of locality sensitive hashing in python

# What is locality sensitive hashing?
Locality sensitive hashing is a method for quickly finding (approximate) nearest neighbors. This implementation follows the approach of 
generating random hyperplanes to partition the dimension space in neighborhoods and uses that to hash the input space into buckets.
To read more about LSH and this specific implementation, see https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection

To train the model: 
```
#assumes that data is a num_observations by num_features numpy matrix
lsh_model = LSH(data)
num_of_random_vectors = 15
lsh_model.train(num_of_random_vectors)

#find the 5 nearest neighbors of data[1] while searching in 10 buckets 
lsh_model.query(data[1,:], 5, 10)
```