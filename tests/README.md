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
SN2            -497.4         1.1            ✓
cope           -556.5         3.9            ✓
DA             -497.0         3.6            ✓
Hshift         -1898.6        12.3           ✓
C2N2O          -493.8         1.7            ✓
cycbut         -741.2         12.5           ✓
DAcpd          -470.9         4.6            ✓
ethCF2         -377.7         13.6           ✓
ene            -970.5         54.1           ✓
HFloss         -1801.7        16.4           ✓
oxir           -565.3         5.9            ✓
Ocope          -553.6         3.3            ✓
SO2loss        -319.6         76.6           ✓
```

#### SM (small metal)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------
hydroform1     -433.9         44.1           ✓
MnInsert       -295.9         85.3           ✓
grubbs         -118.5         45.1           ✓
vaskas         -87.8          38.2           ✓
```
