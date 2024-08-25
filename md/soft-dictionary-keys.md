---
title: '"Soft" Dictionary Keys'
---

[Transformer Neural Networks][1] utilize the key-concept of an Attention Mechanism to perform "lookups" on the data it has seen. In this post I want to detail the idea of "soft" keys, and for me it was easier to get to the crux of how Transformers work with this understanding. I first came across this idea from a [Lucas Beyer talk][2].

## Python Dictionaries

Most programming languages implement a dictionary (or associative map) as a primitive data structure and define them as associations between the abstract idea of keys and values.

In python keys are defined as any hashable object. For example,

```python
m = {
	 "dog": 10,
	 "cat": 2,
	 "tiger": 5,
	 8: 12
}
```

Here, we have four keys, `"dog"`, `"cat"`, `"tiger"`, `8` and they are mapped to values. The first three keys are Strings and the fourth key is a Number (integer in this case). All the values here are Numbers as well.

Internally, python calls the in-built [`hash`][3]  [method][4] to hash the keys into a well-known or fixed representation,

```python
>>> hash(10)
10
>>> hash('abc')
4001473844447581453
```

The key point here is that keys are converted into a well-defined representation. In the case of python the representation is eventually a [fixed size integer][5].

## Exact Dictionary Lookups

For the python example, dictionary look ups work by supplying a query and returning the value for a key that _exactly_ matches the query if there is one,

```python
>>> m["dog"]
10
>>> m["lion"]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'lion'
```

In this case, the query `"dog"` has an exact match while `"lion"` does not. This exact match is useful in real world programming because real world objects are usually discrete. For example, as a web developer building a website that sets the color of the font based on the day of the week, we don't need more than 7 exact keys to represent the days of the week. There is a discreteness to the entire process.

## Matrix Lookup

We can also rewrite this look up using matrix multiplication for succinctness (and move into how this relates to transformers and machine learning). The succinctness here is just a reorganization of the `m["dog"]` lookup, there is nothing else happening here other than a notational change and the introduction of matrix multiplication.

For the example above, the matrix notation representing the dictionary would be,
```maths
keys = [
	1 0 0 0
	0 1 0 0
	0 0 1 0
	0 0 0 1
]

values = [
	10
	2
	5
	12
]
```


Here, we represent and interpret each row of the `keys` matrix to one of the four keys `"dog"`, `"cat"`, `"tiger"`, `8`, (in this order). The `values` matrix is a column vector for the corresponding values from the example above.

In this example we have fixed set of `keys` and the possible `queries` come from this set. Ie, the "domains" of these two objects are the same.

To look up `dog` we first setup a matrix, `query`, using the row interpretation of the `keys` matrix as,
```maths
dog_query = [
	1 0 0 0
]

8_query = [
	0 0 0 1
]
```

Here, the `dog_query` is a row vector with the first column set to 1 and the others 0. The 1 in the first column means we want the first key `dog`. If we wanted to pick `tiger`, we would set the last column to 1.

The idea with the `query` matrix is that a row in that matrix is a binary selector of a particular column in the `key` matrix, expressed as a matrix multiplication. Each column thus is interpreted as the key we want to lookup. Note how the column set to `1` is the index that matches the corresponding row in the `keys` matrix. Ie, the first column matches with the first row, the second column with the second row and so on. The column count must match the row count of the keys matrix.

To perform the actual look up we simply multiply the `query`, `key` and `value` matrix using the rules of matrix multiplication,
```maths
k: dog_query * keys -> [1 0 0 0]

output: k * values -> [10]
```

We can also batch the lookups and perform the equivalent of a python for loop, for example,
```python
>>> m = {
...      "dog": 10,
...      "cat": 2,
...      "tiger": 5,
...      8: 12
... }
>>> queries = ["dog", 8]
>>> output = []
>>> for query in queries:
...     output.append(m[query])
...
>>> print(output)
[10, 12]
```

using the matrix lookup as,

```maths
queries = [
	1 0 0 0
	0 0 0 1
]

k: queries * keys -> [
	1 0 0 0
	0 0 0 1
]

output: k * values -> [
	10
	12
]
```

From the `output`, we can simply look up the row corresponding to the `queries` to get the output we wanted.

## "Soft" keys

The matrix lookup above is doing exactly what the python dictionary lookup would do except in a different notation assuming you setup the problem in accordance to the rules of matrix multiplication.

The important observation here is that the keys matrix is a collection of null vectors with one of the columns set to 1. The interpretation we make with this setup is that the keys are independent of each other. In the real world we make two assumptions about objects (keys in our case). We assume that,

	1. objects are discrete entities, to make the problem easier to interpret and deal with.
	2. objects are treated an independent entites, even though they could be related, and they are handled in a manner that remove that relatedness.

In the `m` dict above, we introduce an exact match for lookup because we want objects to map to exactly one value. If objects are related, like `dog` and `cat`, we remove any relationship that the real world has (ie they are both animals). We do this because for our hypothetical use-case it might not matter, and we can ignore it having understood that we can ignore this relationship.

However if we did want to introduce a relationship between `dog` and `cat`, we have a few options,

1. We look at our use case and figure out a way to connect `dog`, `cat`, `tiger` and our new key `animal`.
2. We ignore the relationship of `animal` with the existing keys of the dictionary and assign some value our use case would find useful.

Both options would satisfy our assumptions above.

An example of option 1 and option 2 in python code would be,
```python
# option 1
dog_value = m["dog"]
cat_value = m["cat"]

animal_value = dog_value + cat_value

m["animal"] = animal_value

# option 2
m["animal"] = 112
```

What if we did not want to explicitly introduce this new key like the code above?

In python we could simply introduce a new operation to perform a combined key look up, that is a variation of option 1,
```python
def soft_key_lookup(keys):
	collector = 0
	for key in keys:
		collector += m[key]
	return collector

assert combine_keys(["dog", "cat"]) == m["dog"] + m["cat"]
```

In matrix notation this would be,
```maths
queries = [
	1 1 0 0
]

k: queries * keys -> [
	1 1 0 0
]

output: k * values -> [
	12
]
```

This process is however cumbersome. We have related `"dog"` and `"cat"` but we might have missed the relationship between `"animal"` and `"tiger"`.

The ability to combine keys (based on our interpretation) is the key-concept of Attention as used in transformers and other machine learning architectures. The idea is that if we can introduce a _learnable parameter_ we can simply learn the relationship between keys and values, and for a query that might not have had an _exact match_ we can compute a "partial" or "soft" match.

The intuition for Attention comes from the idea that we go from,
```maths
animal = [
	1 1 0 0
]
```

based on our knowledge of cats and dogs, to something that we can learn from data that might eventually look like,

```maths
animal = [
	0.33 0.36 0.31 0
]
```

The idea being the data has enough signal to exact a common relationship between `"dog"`, `"cat"` and `"animal"`.

## References

[^1]: https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)
[2]: https://x.com/giffmana/status/1570152923233144832?s=20
[3]: https://docs.python.org/3/glossary.html#term-hashable
[4]: https://docs.python.org/3/faq/design.html#how-are-dictionaries-implemented-in-cpython
[5]: https://docs.python.org/3/library/sys.html#sys.hash_info
