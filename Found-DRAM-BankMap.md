
## Dell XPS-13 (skylake)

$ sudo ./measure -s 16  -m 1024 -r -t 550

Bits: 1, sz=1
Bits: 2, sz=4
14 18  (Correct: 87%)
15 19  (Correct: 87%)
16 20  (Correct: 75%)
17 21  (Correct: 87%)
Bits: 3, sz=4
Bits: 4, sz=6
Bits: 5, sz=6
Bits: 6, sz=4
Bits: 7, sz=4
Finishing

Inferred mapping:  
14 XOR 18
15 XOR 19
16 XOR 20
17 XOR 21

## Toshiba PORTEGE R835P-56X (Sandybridge)

Bits: 1, sz=0
Bits: 2, sz=4
14 18  (Correct: 90%)
15 19  (Correct: 72%)
16 20  (Correct: 72%)
17 21  (Correct: 72%)
Bits: 3, sz=0
Bits: 4, sz=6
Bits: 5, sz=0
Bits: 6, sz=4
Bits: 7, sz=0
Finishing

Interferred mapping:
14 XOR 18
15 XOR 19
16 XOR 20
17 XOR 21

## Intel Nehalem  (2xdimms) 

Bits: 1, sz=5
13  (Correct: 71%)
14  (Correct: 71%)
16  (Correct: 85%)
20  (Correct: 100%)
21  (Correct: 71%)
Bits: 2, sz=10
Bits: 3, sz=10
Bits: 4, sz=5
Bits: 5, sz=1
Bits: 6, sz=0
Bits: 7, sz=0
Finishing

Inferred mapping:
13
14
16
20
21

## Raspberry Pi 3

## Raspberry Pi 4 (2GB)
drama/re$ sudo ./measure -s 16
...
reduced to 5 functions
Bits: 1, sz=5
11  (Correct: 100%)
12  (Correct: 100%)
13  (Correct: 100%)
14  (Correct: 100%)
Bits: 2, sz=10
Bits: 3, sz=10
Bits: 4, sz=5
Bits: 5, sz=1
Bits: 6, sz=0
Bits: 7, sz=0
Finishing

palloc/$ sudo ./mc-mapping-pagemap -n 2  -p 0.7
Bit6: 309.34 MB/s, 206.89 ns
Bit7: 315.81 MB/s, 202.65 ns
Bit8: 318.90 MB/s, 200.69 ns
Bit9: 310.80 MB/s, 205.92 ns
Bit10: 309.70 MB/s, 206.65 ns
Bit11: 363.54 MB/s, 176.05 ns
Bit12: 377.43 MB/s, 169.57 ns
Bit13: 459.58 MB/s, 139.26 ns
Bit14: 519.06 MB/s, 123.30 ns
Bit15: 309.36 MB/s, 206.88 ns
Bit16: 309.75 MB/s, 206.62 ns
Bit17: 309.51 MB/s, 206.78 ns
Bit18: 309.39 MB/s, 206.86 ns
Bit19: 309.45 MB/s, 206.82 ns
Bit20: 309.53 MB/s, 206.77 ns
Bit21: 309.68 MB/s, 206.66 ns
Bit22: 309.40 MB/s, 206.85 ns
Bit23: 309.75 MB/s, 206.62 ns

Inferred mappnig:
11, 12, 13, 14

## Raspberry Pi 4 (4GB)

	drama/re$ sudo ./measure -s 16
	...
	reduced to 4 functions
	Bits: 1, sz=4
	11  (Correct: 100%)
	12  (Correct: 100%)
	13  (Correct: 100%)
	14  (Correct: 100%)
	Bits: 2, sz=6
	Bits: 3, sz=4
	Bits: 4, sz=1
	Bits: 5, sz=0
	Bits: 6, sz=0
	Bits: 7, sz=0
	Finishing

	palloc/$ sudo ./mc-mapping-pagemap -n 3 -p 0.7
	mem_size (MB): 2655
	allocation complete.
	worker thread begins
	worker thread begins
	worker thread begins
	Bit6: 327.00 MB/s, 195.72 ns
	Bit7: 319.04 MB/s, 200.60 ns
	Bit8: 317.86 MB/s, 201.35 ns
	Bit9: 323.99 MB/s, 197.54 ns
	Bit10: 327.19 MB/s, 195.60 ns
	Bit11: 395.38 MB/s, 161.87 ns
	Bit12: 347.05 MB/s, 184.41 ns
	Bit13: 493.28 MB/s, 129.74 ns
	Bit14: 467.45 MB/s, 136.91 ns
	Bit15: 316.39 MB/s, 202.28 ns
	Bit16: 317.29 MB/s, 201.71 ns
	Bit17: 315.36 MB/s, 202.94 ns
	Bit18: 313.01 MB/s, 204.47 ns
	Bit19: 315.52 MB/s, 202.84 ns
	Bit20: 311.93 MB/s, 205.17 ns
	Bit21: 319.72 MB/s, 200.18 ns
	Bit22: 317.66 MB/s, 201.48 ns
	Bit23: 318.08 MB/s, 201.21 ns

Inferred mapping:
11, 12, 13, 14


## NVIDIA Jetson Nano

## NVIDIA Jetson TX2

## NVIDIA Jetson Xavier

