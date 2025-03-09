## Learning summary

Chapter 6 of pmpp (third ed)

### The no-zero representation (Fig 6.5)
Working on 5 bits repr (1S, 2M, 2E)
1. The exponent bits define the major intervals of representable numbers. In this repr, there are 3 major
intervals on each side of 0 (2^-1, 2^0, 2^1) and (-2^-1, -2^0, -2^1)
2. The mantissa bits define the number of representable numbers in each interval. With two mantissa bits,
there are 2^m representable numbers in each interval
3. 0 is not representable in this format.
4. Representable numbers become closer to each other toward the neighborhood of 0. Each interval is half the size of the previous 
interval as we move toward zero.
5. With m bits in the mantissa, this style of representation would introduce 2^m times more error in the interval
closest to zero than the next interval.

### The abrupt underflow
Working on 5 bits repr (1S, 2M, 2E)
1. Whenever E is 0, the number is interpreted as 0. In 5-bit format, this 
method takes away eight representable numbers (four positive and four negative).
2. Although this method makes 0 a representable number, it creates an even larger 
gap between representable numbers in 0’s vicinity.


### Denormalization (IEEE standard)

1. The representation has uniformly spaced representable numbers in the close vicinity of 0.
2. NaN: all ones in exponent
3. inf: all ones in exponent and all 0 in mantissa
4. 0: all zeros in exponent and mantissa

### Additional infos
1. Rounding occurs if the mantissa of the result value needs too many bits to be represented exactly
2. ULP (Units in the Last Place). If the hardware is designed to perform arithmetic and rounding operations perfectly, the most error that one should 
introduce should be no more than 0.5D ULP

### Exercices
1. The additional mantissa bit adds more representable numbers in each number. In each interval, it will be 2^3=8 representable
number in each interval defined by the exponent.
2. The additional exponent bit will increase the number of intervals (the range).
3. 1 ULP
4. In his algo (reduce sum with no block divergence, see reduce2.cu in day021), each time the smallest value will
summed with the largest value because the array is sorted. And thus, the smaller operand could simply disappears 
because it is too small compared to the larger operand.
5. The full mantissa has 24 bits (the implicit “1.” plus 23 fraction bits). In this design, the iterative algorithm produces 2 bits per clock cycle over 9 cycles, giving 18 bits. The remaining 6 bits are filled with 0's. The worst-case error occurs when the true result’s lower 6 bits would have been all ones.
So the maximal error is 2^6 -1 = 63 ULP.