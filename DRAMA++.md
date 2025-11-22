# DRAMA++: A fast and robust DRAM address map reverse engineering tool.

The code is based on DRAMA (https://github.com/isec-tugraz/drama), but it has made the following improvements.

- added support for ARM64 architecture
- implemented a faster GF(2) solver with a polynomial time complexity (original version has an exponetial time complexity)
- fixed a logic bug, which prevented considering high-order physical address bits
- fixed a logic bug, which leaves the 'base' address remaining in the address pool when it should have been added to the set.
- improved timing measurement
- additional changes for improved functionality, usability and portability

To see the all changed, do the following:

    git diff c5c83471...HEAD re/measure.cpp

## Speed/correctness comparison

    Platform        DRAMA       DRAMA++
    ------------------------------------
    i5-2520M        3m34s (X)   0.84s (O)
