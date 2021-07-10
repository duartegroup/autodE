### Running

To run the tests in this directory ensure the requirements are satisfied

```bash
conda install --file requirements.txt
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
SN2            -497.0         1.3            ✓
cope           -557.4         4.8            ✓
DA             -484.7         18.1           ✓
Hshift         -1898.7        2.7            ✓
C2N2O          -493.7         1.8            ✓
cycbut         -741.3         14.2           ✓
DAcpd          -470.6         6.9            ✓
ethCF2         -378.3         17.0           ✓
ene            -966.7         25.0           ✓
HFloss         -1801.7        11.3           ✓
oxir           -567.1         7.7            ✓
Ocope          -525.3         3.0            ✓
SO2loss        -324.1         47.1           ✓
aldol          -259.4         19.7           ✓
dipolar        -442.3         8.8            ✓
```

#### SM (small metal)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------
hydroform1     -433.8         31.7           ✓
MnInsert       -296.0         76.0           ✓
grubbs         -121.8         59.3           ✓
vaskas         -94.3          44.3           ✓
```
