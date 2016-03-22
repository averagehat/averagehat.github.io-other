

Functional programming principles can be productively applied even in small cases. The predictability and compositional capacity of small functions 
allows for modular, extensible code. First-order functions can be used in a variety of generic higher-order functions or "combinators". 
In the future, we will look at how static typing combined with immutability can make this code much safer as an additional benefit.

Let's use calculating [GC-content](https://en.wikipedia.org/wiki/GC-content) as an example.

Where `seq` is some DNA sequence--just a string of characters composed soley of items in the set ```{'A', 'C', 'G', 'T'}```



    def gc(seq):
        '''(G+C)/(A+T+G+C) * 100'''
        return (seq.count('G') + seq.count('C')) / float( len(seq) )
    print(gc("ACGTCCA"))


    0.571428571429


But reall, we want to model GC-content as a fraction, so at any given time we know the size of the sequence. Python has a fraction class, but it doesn't allow us to represent the state `0/0`, which would be our startin state.



    from fractions import Fraction
    class Ratio(Fraction):
        ''' new is for immutable classes.'''
        def __new__(cls, num, denom):
            '''cls is like self really '''
            if denom ==0 and num == 0:
                self = super(Fraction, cls).__new__(cls)
            else:
                self = super(Fraction, cls).__new__(cls)
            self.__str__ = Ratio.__str__
            self._numerator, self._denominator = num, denom
            return self
        def __float__(self):
            if self._denominator == 0:
                return 0.0
            else:
                return super(Fraction, self).__float__()
            
        def __str__(self):
            return "{0}/{1}".format(self._numerator, self._denominator)


Now we can update our function:



    def gc(seq):
        '''(G+C)/(A+T+G+C) * 100'''
        return Ratio(seq.count('G') + seq.count('C'),  len(seq))
    print(gc("ACGTCCA"))


    4/7


'This is all well and good, but what if we want to know the GC-content at a certain place in the read?



    seq = 'AGCTTAGGCCTTTAAAACCGGGGCCCCCGGAAGCGACTT'
    print gc(seq[:10])


    6/10


That works, but will get tiresome and inefficient...



    gc(seq[0]), gc(seq[:10]), gc(seq[-1])





    (Fraction(0, 1), Fraction(6, 10), Fraction(0, 1))



What if we want to create a histogram of the gc-content at each position in the read?



    N = len(seq)
    def gc_hist_(seq):
        return [gc_counter(seq[:i]) for i  in xrange(1, N)]


we have to re-calculate the GC-ratio each time! In terms of runtime, this is equivalent to:



    sum(i for i in xrange(N) )





    741



Well, this method is O(N^2) (about (N^2)/2, where N=length of sequence). So the runtime is quadratic.
We can fix this by writing a new function specifially for constructing a list of GC-content. 

Instead, let's try solving this problem with higher-order functions, and see where that gets us.

Functional programming works well when we compose small functions together. Let's create the smallest unit we can out of this problem:



    def content(ratio, nt):
        val = 1 if nt in 'GC' else 0
        return Ratio(ratio.numerator + val, ratio.denominator+1)


Next, let's solve our problem of creating a rolling GC-content for a histogram using a higher-order function and `content`. A higher-order function is just a function which accepts another function as one of its arguments. This feature isn't available in all programming languages, but it's available in python. The higher-order function we'll use is `accumulate`.

In haskell, accumulate is known as `scanl`, and it's corrolary is `foldl`. `accumulate` is similar to `reduce` in python, except it keeps track of every result, rather than throwing them away and only keeping the final result.



    def accumulate(iterable, func, start=None):
        '''re-implementation of accumulate that allows you to specify a start value
        like with reduce.'''
        it = iter(iterable)
        if start is None:
            total = next(it)
            yield total
        else:
            total = start
        #could also skip yielding total
        for element in it:
            total = func(total, element)
            yield total
    
    import operator
    print list(accumulate([1,2,3,4,5], operator.mul)) # --> 1 2 6 24 120
    print list(accumulate("ACGCCGT", content, Ratio(0, 0)))
    from functools import partial
    func_gc_hist = partial(accumulate, func=content, start=Ratio(0,0))


    [1, 2, 6, 24, 120]
    [Fraction(0, 1), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(4, 5), Fraction(5, 6), Fraction(5, 7)]


Now we have our histogram! Let's look more closely at how this works using `reduce` as an example. Using our function `content` with `reduce` will give us the total GC-ratio for a given sequence.
`reduce` emulates recursion by computing a new value for each element in the list and passing it onto
the next call with the element as a paramter. Let's look at a simple example of summing a list: 



    def _add2(acc, elem):
        return elem + acc
    
    print reduce(_add2, [1,2,3,4,5])


    15


Let's break this down, step by step. Here, each line represents an iteration of reduce. . . . 


```python
# under the hood, this might look like:
[1, 2, 3], 0
-> [1, 2], 3
-> [1],   5
>>>6
```

In a sense `reduce` is "substituting" parts of the list for there sum, similar to how one might simplify an algabreic equation.
We can see this pattern in accumulate's output:



    print list(accumulate([1,2,3,4,5], _add2))


    [1, 3, 6, 10, 15]


    Another simple example, reversing a list. list reversal has a straightforward recursive solution:





    def _reverse(l):
        if len(l) == 0: return []
        return [l[-1]] + _reverse(l[:-1])
    print(_reverse([1, 2, 3, 4]))


    [4, 3, 2, 1]


But it can be written much more simply using `reduce`, which handles the details of the recursive call.



    def _reverse(acc, elem):
        return [elem] + acc
    reduce(_reverse, [1,2,3,4], [])





    [4, 3, 2, 1]



Again, let's break it down:


```python
[1, 2, 3, 4], []
-> [1, 2, 3], [4]
-> [1, 2],    [4, 3]
-> [1],       [4, 3, 2]
>>>[4, 3, 2, 1]
```

notice that at any point during the traversal, the accumulated value
is correct for the traversed part of the list.
Using this model of folding an accumulating paramter over a sequence,
we can model GC-content as a "rolling ratio" over a given sequence of nucleotides.
At any point during the traversal, The ratio will be correct, and the result of "reducing"
the sequence with this model will give us our total GC-content. The following method is not
the most efficient nor the simplest (it requires building a "Ratio" subclass), but it closely
(and flexibly) models the mathematical formula that defines GC-content in the first place. '''



    seq = 'AGCTTAGGCCTTTAAAACCGGGGCCCCCGGAAGCGACTT'
    print reduce(content, seq, Ratio(0,0))


    23/39


That's all reasonable, and it's nice to see GC-content representated as a ratio. But is it practical?
Well, it will work, but there is a much more efficient way. Array-wise computations like this--which will become
quite large if we get big reads (or god forbid) a whole contig/genome. Additionally, we may want to scale to viewing
multiple reads at once, ie, as a matrix.




    from random import choice
    N = 1000000
    alpha = list('AGCT')
    seq = ''.join( choice(alpha) for _ in xrange(N) )
    from fn.iters import repeatfunc
    seq = ''.join(repeatfunc(partial(choice, alpha), N))
    %timeit functional = func_gc_hist(seq)


    The slowest run took 4.67 times longer than the fastest. This could mean that an intermediate result is being cached 
    1000000 loops, best of 3: 663 ns per loop


Quite slow. Let's see how the efficient `numpy` library can achieve much faster runtimes,
and how we can use these same functional principals--recursion, reduction, and accumulation--to get more leverage (and cleaner code)
out of numpy.



    import numpy as np
    np.array([1, 2, 3, 4, 5]).sum()





    15



`np.sum` is actually a specific (read: partial application) of np.reduce!
[source](https://github.com/numpy/numpy/blob/a9c810dd1d8fc1e3c6d0f0ca6310f41795545ec9/numpy/core/_methods.py)



    npseq = np.array(list(seq))
    gccon = ((npseq == 'C') | (npseq == 'G')).sum()/float(len(npseq))
    npresult = ((npseq == 'C') | (npseq == 'G')).cumsum()


cumulative sum rolls like accumulate!

Now, we'll divide by the index (starting at one) to simulate the ratio.



    gcs = ((npseq == 'C') | (npseq == 'G'))
    npres = gcs.cumsum()/np.arange(1,len(npresult)+1, dtype=float)
    np_filter_idx = (npres >= .5).nonzero()


Putting this together:



    def np_gc_hist(seq):
         npseq = np.array(list(seq))
         gcs = ((npseq == 'C') | (npseq == 'G'))
         npres = gcs.cumsum()/np.arange(1,len(gcs)+1, dtype=float)
         return npres


Let's plot it:



    import matplotlib
    from matplotlib import pyplot as plt
    %pylab inline
    THRESH=0.4
    def doplot(result):
        fig = plt.figure()
        fig.set_size_inches( 20.0, 8.0 )
        gs = matplotlib.gridspec.GridSpec(1,2, width_ratios=[20,1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax1.plot(result)
        '''draw a line for the threshold.'''
        ax1.axhline(y=THRESH, color='green')
        fig.show()
    doplot(npresult)


    Populating the interactive namespace from numpy and matplotlib



![png](output_43_1.png)



Let's try timing again. First, our old result:



    %timeit imperative = list(gc_hist(seq))


Now with numpy:



    %timeit numpy_sttyle = np_gc_hist(seq)


    10 loops, best of 3: 140 ms per loop

