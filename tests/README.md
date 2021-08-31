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
SN2            -496.8         1.3            ✓
cope           -557.3         4.7            ✓
DA             -484.4         17.6           ✓
Hshift         -1898.8        2.6            ✓
C2N2O          -493.6         1.7            ✓
cycbut         -741.1         14.3           ✓
DAcpd          -470.4         6.7            ✓
ethCF2         -377.4         15.4           ✓
ene            -966.7         17.0           ✓
HFloss         -1801.7        7.4            ✓
oxir           -569.5         11.0           ✓
Ocope          -525.4         2.9            ✓
SO2loss        -324.1         26.4           ✓
aldol          -260.3         19.3           ✓
dipolar        -442.7         8.4            ✓
```

#### SM (small metal)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------
hydroform1     -434.1         38.0           ✓
MnInsert       -295.7         58.1           ✓
grubbs         -121.1         63.8           ✓
vaskas         -94.6          39.8           ✓
```
