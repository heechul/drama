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
## Usage

### Prerequisites

- Linux (x86-64 or ARM64) with a recent kernel.
- `g++` and `make` installed.
- Permission to read `/proc/self/pagemap` (often requires `sudo`). The tool attempts huge pages first and falls back to regular pages if unavailable.

### Build

```
cd re
make
```

This produces the `measure` binary in `re/`.

### Basic Run

Measure DRAM bank functions and save them to `map.txt`:

```
cd re
./measure [-m <memory size in MB> | -g <memory size in GB>] [-i <number of outer loops>] [-j <number of inner loops>] [-s <expected sets>] [-q <sets for early quit>] [-t <threshold cycles>] [-f <output file>]

```

Notes:
- `-s`: expected sets = DIMMs × channels × ranks × banks (e.g., 1×1×2×8 = 16).
- `-m`/`-g`: memory to map in MB/GB (e.g., `-g 1` for 1 GB).
- `-c`: pin to a CPU core (you can also use `taskset`).
- `-i`/`-j`: outer/inner loop counts; ARM64 may benefit from a higher `-j`.
- `-t`: timing threshold (cycles) to override auto gap detection.
- `-b`/`-e`: search bit window (defaults: 5..40).
- `-q`: stop after N sets are found.
- `-v`: verbosity level.
- `-f`: output file for discovered functions (default `map.txt`).

### Outputs

- `setN.txt`: physical addresses of each discovered same-bank set.
- `map.txt`: one line per XOR function with the physical address bit indices.

### Example

1 DIMM, 1 channel, 2 ranks, 8 banks (16 sets), mapping 1 GB:

```
sudo ./measure -s 16 -g 1
```

Also see: [Found-DRAM-BankMap.md](./Found-DRAM-BankMap.md) for examples of discovered DRAM bank-mapping functions.

## Speed Comparison


| Platform              |   DRAMA    |  DRAMA++ |
|-----------------------|------------|----------|
| Intel i5-2520M        |   3m34s    |  0.84s   |
| Intel Xeon E5-2608L v3|   17m48s   |  2.64s   |
| Raspberry Pi 4        |    N/A     |  0.58s   |
