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
SN2            -497.4         1.3            ✓
cope           -557.4         4.8            ✓
DA             -484.4         19.4           ✓
Hshift         -1898.9        3.0            ✓
C2N2O          -494.0         2.0            ✓
cycbut         -741.2         14.6           ✓
DAcpd          -470.8         6.9            ✓
ethCF2         -377.5         16.0           ✓
ene            -966.9         19.7           ✓
HFloss         -1801.7        8.3            ✓
oxir           -567.7         6.4            ✓
Ocope          -525.4         3.0            ✓
SO2loss        -325.0         27.5           ✓
aldol          -259.6         19.2           ✓
dipolar        -442.1         8.2            ✓
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
