### Running

To run the tests in this directory ensure the requirements are satisfied

```bash
conda install --file requirements.txt
```

then run the tests

```bash
py.test 
```

***
### Benchmark
In addition to the tests there is a benchmark of calculations (*benchmark.py*) to
run for every minor release. This benchmark **must** be run and the results 
posted below to ensure consistent functionality - it should take a few hours 
on 8 cores. 

#### SO (small organic)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------
SN2            -496.9         1.2            ✓
cope           -557.2         7.1            ✓
DA             -484.9         19.9           ✓
Hshift         -1898.8        2.8            ✓
C2N2O          -493.7         1.8            ✓
cycbut         -741.3         13.9           ✓
DAcpd          -470.7         6.6            ✓
ethCF2         -377.1         15.5           ✓
ene            -966.8         16.9           ✓
HFloss         -1795.6        7.4            ✓
oxir           -570.9         4.4            ✓
Ocope          -525.3         2.9            ✓
SO2loss        -324.9         26.3           ✓
aldol          -259.8         18.9           ✓
dipolar        -442.1         8.4            ✓
```

#### SM (small metal)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------
hydroform1     -434.1         31.7           ✓
MnInsert       -302.1         68.1           ✓
grubbs         -122.6         48.2           ✓
vaskas         -94.6          39.8           ✓
```
