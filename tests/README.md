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
SN2            -490.1         1.8            ✓
cope           -557.5         4.8            ✓
DA             -484.4         12.8           ✓
Hshift         -1899.2        9.3            ✓
C2N2O          -494.3         2.3            ✓
cycbut         -741.1         17.0           ✓
DAcpd          -470.8         6.1            ✓
ethCF2         -377.9         17.5           ✓
ene            -967.6         65.1           ✓
HFloss         -1801.1        28.3           ✓
oxir           -559.4         57.5           ✓
Ocope          -525.4         3.7            ✓
SO2loss        -324.0         48.0           ✓
```

#### SM (small metal)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------
hydroform1     -436.3         61.1           ✓
MnInsert       -302.0         136.9          ✓
grubbs         -119.3         90.0           ✓
vaskas         -95.5          63.7           ✓
```
