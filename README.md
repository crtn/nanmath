### correlation, mean, std calculation with NAN using AVX

### gcc
<pre>
processing 252 x 1000 matrix
                nan_corr_float: 466 ms             sum: 485.892          count: 80561589    
            nan_corr_float_avx: 67 ms              sum: 485.894          count: 80561589    
               nan_corr_double: 470 ms             sum: 485.893          count: 80561589    
           nan_corr_double_avx: 95 ms              sum: 485.893          count: 80561589    
</pre>

<pre>
processing 252 x 5000 matrix
                nan_corr_float: 11533 ms           sum: 1725.16          count: 2015596733  
            nan_corr_float_avx: 1636 ms            sum: 1725.16          count: 2015596733  
               nan_corr_double: 11913 ms           sum: 1725.17          count: 2015596733  
           nan_corr_double_avx: 2582 ms            sum: 1725.17          count: 2015596733  
</pre>

<pre>
processing 252 x 10000 matrix
                nan_corr_float: 46521 ms           sum: 1965.41          count: 3768224907  
            nan_corr_float_avx: 6813 ms            sum: 1965.48          count: 3768224907  
               nan_corr_double: 47984 ms           sum: 1965.48          count: 3768224907  
           nan_corr_double_avx: 10402 ms           sum: 1965.48          count: 3768224907  
</pre>


### icc
<pre>
processing 252 x 1000 matrix
                nan_corr_float: 63 ms              sum: 485.894          count: 80561589    
            nan_corr_float_avx: 57 ms              sum: 485.894          count: 80561589    
               nan_corr_double: 101 ms             sum: 485.893          count: 80561589    
           nan_corr_double_avx: 78 ms              sum: 485.893          count: 80561589    
</pre>

<pre>
processing 252 x 5000 matrix
                nan_corr_float: 1585 ms            sum: 1725.16          count: 2015596733  
            nan_corr_float_avx: 1437 ms            sum: 1725.16          count: 2015596733  
               nan_corr_double: 2886 ms            sum: 1725.17          count: 2015596733  
           nan_corr_double_avx: 2449 ms            sum: 1725.17          count: 2015596733  
</pre>

<pre>
processing 252 x 10000 matrix
                nan_corr_float: 6620 ms            sum: 1965.48          count: 3768224907  
            nan_corr_float_avx: 6008 ms            sum: 1965.48          count: 3768224907  
               nan_corr_double: 10948 ms           sum: 1965.48          count: 3768224907  
           nan_corr_double_avx: 9374 ms            sum: 1965.48          count: 3768224907  
</pre>

