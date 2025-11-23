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

### Options

Measure DRAM bank functions and save them to `map.txt`:

```
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

## DRAM Bank Map Database
See: [Found-DRAM-BankMap.md](./Found-DRAM-BankMap.md) for examples of discovered DRAM bank-mapping functions.

## Speed Comparison


| Platform              |   DRAMA    |  DRAMA++ |
|-----------------------|------------|----------|
| Xeon E3-1220 v5 (64 banks)     |   54.5s<sup>1</sup>    |  3.4s<sup>2</sup>    | 
| Raspberry Pi 4 (8 banks)       |    N/A     |  0.6s    |

- <sup>1</sup> Used DRAMA option: "-s 64 -n 10" (the default, n=5000, took too long, +10min, and couldn't find the map.)
- <sup>2</sup> Used DRAMA++ option: "-s 64" (manually setting the threshold, e.g., "-t 300", would make it even faster and more reliable)
