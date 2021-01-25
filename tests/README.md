### Running

To run the tests in this directory ensure the requirements are satisfied

```bash
conda install --file requirements.txt
```

set $AUTODE_FIXUNIQUE to False, e.g. in bash

```bash
export AUTODE_FIXUNIQUE=False
```

then run the tests

```bash
py.test 
```


### Benchmark
In addition to the tests there is a benchmark of calculations (benchmark.py) to
run for every minor release. This benchmark **must** be run and the results 
posted below to ensure consistent functionality - it should take a few hours 
on 8 cores. 

#### SO (small organic)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------

```

#### SM (small metal)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------

```
