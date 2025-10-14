
## Dell XPS-13/i5-6200U (Skylake, 8 GB, 4 ranks, 2GB/rank, )

	$ sudo ./measure -s 32 -t 340 -m 1024 -v 4

	Inferred mapping:  
	14 18
	15 19
	16 20
	17 21
	8 9 12 13 14 15

## Toshiba PORTEGE R835P-56X (Sandybridge)

	Interferred mapping:
	14 18
	15 19
	16 20
	17 21

## Dell E6420/i5-2520M (Sandybridge, 16GB, 2-dimms )

	Interferred mapping:
	14 18
	15 19
	16 20
	17 21

## Intel Xeon W3553 (Nehalem, 2xdimms) 

	Inferred mapping:
	13
	14
	16
	20
	21

## Intel E3-1220 v5 (Skylake, 4xdimms, 1 dimm = DDR4 4GB)

	Inferred mapping:
	7 14
	15 19
	16 20
	17 21
	18 22
	8 9 12 13 15 18

## Intel Intel(R) Xeon(R) CPU E5-2608L v3 (Haswell, 4xdimms, 1 dimm = DDR4 4GB)

	Inferred mapping:
	22
	19 23
	20 24
	21 25
	8 13 15 17 27
	7 12 14 16 18 26

## Raspberry Pi Zero 2 (Cortex-A53, LPDDR2, 512MB)

	Inferred mapping:
	12
	13
	14

## Raspberry Pi 4 (Cortex-A72, LPDDR4, 2GB)

	Inferred mapping:
	12
	13
	14

## Raspberry Pi 5 (Cortex-A76, LPDDR4X, 4GB)

	Inferred mapping (SDRAM_BANKLOW = 4):
	12
	13
	14
	31

	Inferred mapping (SDRAM_BANKLOW = 1, default):
	12
	30
	31
	32

## NVIDIA Jetson Nano (Cortex-A57, 64bit LPDDR4, 4GB)

	Infered mapping:
	13 19 20 21 24 25 26 28
	10 12 14 16 17 21 25 27 28
	10 16 17 18 22 23 27 29 30
	10 11 13 15 16 20 22 24 25 29

## NVIDIA Jetson Orin Nano (Cortex-A78E, LPDDR5, 8GB)

	Infered mapping:
	11 17 19 25 27 28 31
	9 12 22 23 24 31 33
	10 17 21 27 29 32 33
	10 11 15 20 21 22 24 29
	13 14 16 20 24 27 30 33
	9 12 16 17 20 21 24 26 27
	10 14 15 17 18 24 25 26 27

## NVIDIA Jetson Orin AGX (Cortex-A78E, LPDDR5, 32GB)

	Infered mapping (correct?):
	11 14 16 20 21 22 33
	11 12 13 16 19 24 33
	9 13 23 24 27 28 33
	12 13 18 22 25 29 30 31
	10 11 12 17 19 20 23 32
	10 11 13 14 18 27 28 34
	14 15 17 21 25 28 31 34
	10 13 14 15 24 26 28 29 31

