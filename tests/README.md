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
SN2            -501.4         1.8            ✓
cope           -555.0         9.7            ✓
DA             -488.8         22.8           ✓
Hshift         -1901.4        3.6            ✓
C2N2O          -492.7         2.5            ✓
cycbut         -740.6         13.7           ✓
DAcpd          -465.0         8.9            ✓
ethCF2         -376.6         14.3           ✓
ene            -972.8         72.8           ✓
HFloss         -1801.7        43.3           ✓
oxir           -555.7         9.7            ✓
Ocope          -522.9         6.6            ✓
SO2loss        -322.0         117.9          ✓
aldol          -242.5         30.3           ✓
dipolar        -444.1         13.1           ✓
```

#### SM (small metal)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------
hydroform1     -436.1         27.5           ✓
MnInsert       -293.3         71.9           ✓
grubbs         -107.8         122.2          ✓
vaskas         -93.3          87.1           ✓
```
