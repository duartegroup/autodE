## autodE C++ extensions

Some **autodE** functionality is written in C++ for speed
and the functionality exposed to Python using Cython wrappers around
the base classes. 

Currently it's written in C++11 for compatibility with older compilers. Build and
run the tests by, in this directory:

```bash
cmake . && make -j2 && ./unit_tests
```

*assuming 2 cores are available*.