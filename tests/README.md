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
SN2            -481.7         1.7            ✓
cope           -532.8         9.7            ✓
DA             -469.2         22.1           ✓
Hshift         -1825.4        3.4            ✓
C2N2O          -472.9         2.4            ✓
cycbut         -711.0         11.9           ✓
DAcpd          -446.4         8.7            ✓
ethCF2         -361.3         14.0           ✓
ene            -933.9         65.4           ✓
HFloss         -1729.6        37.7           ✓
oxir           -541.2         8.2            ✓
Ocope          -502.0         6.7            ✓
SO2loss        -309.0         129.3          ✓
aldol          -233.0         24.2           ✓
dipolar        -426.3         13.4           ✓
```

#### SM (small metal)
```
Name      v_imag / cm-1    Time / min     Success
-------------------------------------------------
hydroform1     -418.6         32.6           ✓
MnInsert       -281.7         87.7           ✓
grubbs         -103.8         147.3          ✓
vaskas         -89.6          93.2           ✓
```
