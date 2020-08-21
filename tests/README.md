### Running

To run the tests:

```bash
py.test 
```
in this directory. 


### Benchmark
In addition to the tests there is a benchmark of calculations (benchmark.py) to
run for every minor release. This benchmark **must** be run and the results 
posted below to ensure consistent functionality - it should take less
than a few hours on 8 cores.

```
   Name      v_imag / cm-1     Time / min    Success
------------------------------------------------------
   sn2           -495.9           0.1           ✓         
cope_rearr       -583.3          11.3           ✓         
diels_alder      -486.8           4.3           ✓         
h_shift         -1897.9           2.3           ✓         
h_insert         -433.1          99.8           ✓         
```
