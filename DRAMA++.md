# DRAMA++: A fast and robust DRAM address-map reverse-engineering tool

This code is based on [DRAMA](https://github.com/isec-tugraz/drama), and it makes the following improvements:

- Added support for the ARM64 architecture.
- Implemented a faster GF(2) solver with polynomial-time complexity (the original version has exponential-time complexity).
- Fixed a logical bug that prevented high-order physical address bits from being considered.
- Fixed a logical bug that left the `base` address in the address pool when it should have been added to the set.
- Improved timing measurements.
- Additional changes for improved functionality, usability, and portability.

To see all changes, run:

```
git diff c5c83471...HEAD re/measure.cpp
```

## Speed Comparison


| Platform              |   DRAMA    |  DRAMA++ |
|-----------------------|------------|----------|
| Intel i5-2520M        |   3m34s    |  0.84s   |
| Intel Xeon E5-2608L v3|   >30m     |  2.64s   |
| Raspberry Pi 4        |    N/A     |  0.58s   |
