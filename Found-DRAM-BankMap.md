
## Dell XPS-13/i5-6200U (Skylake, 8 GB, 4 ranks, 2GB/rank, )

	$ sudo ./measure -s 32 -t 340 -m 1024 -v 4

	reduced to 7 functions
	Bits: 1, sz=2
	Bits: 2, sz=5
	14 18  (Correct: 100%)
	15 19  (Correct: 75%)
	16 20  (Correct: 87%)
	17 21  (Correct: 62%)
	Bits: 3, sz=8
	Bits: 4, sz=10
	Bits: 5, sz=12
	Bits: 6, sz=14
	8 9 12 13 14 15  (Correct: 87%)
	Bits: 7, sz=16
	Bits: 8, sz=17
	Bits: 9, sz=18
	Bits: 10, sz=13

	Inferred mapping:  
	14 18
	15 19
	16 20
	17 21
	8 9 12 13 14 15

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

## Dell E6420/i5-2520M (Sandybridge, 16GB, 2-dimms )


	Bits: 2, sz=4
	14 18  (Correct: 50%)
	15 19  (Correct: 50%)
	16 20  (Correct: 100%)
	17 21  (Correct: 50%)
	Bits: 3, sz=0
	Bits: 4, sz=6
	Bits: 5, sz=0
	Bits: 6, sz=4
	Bits: 7, sz=0

	Interferred mapping:
	14 XOR 18
	15 XOR 19
	16 XOR 20
	17 XOR 21

## Intel Xeon W3553 (Nehalem, 2xdimms) 

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

## Intel E3-1220 v5 (Skylake, 4xdimms, 1 dimm = DDR4 4GB)

	sudo ./measure -s 64 -t 300 -v 3 
	..

	reduced to 8 functions
	Bits: 1, sz=2
	Bits: 2, sz=6
	7 14  (Correct: 98%)
	15 19  (Correct: 94%)
	16 20  (Correct: 98%)
	17 21  (Correct: 94%)
	18 22  (Correct: 98%)
	Bits: 3, sz=10
	Bits: 4, sz=15
	Bits: 5, sz=20
	Bits: 6, sz=24
	8 9 12 13 15 18  (Correct: 98%)
	Bits: 7, sz=28
	Finishing

	Inferred mapping
	7 14
	15 19
	16 20
	17 21
	18 22
	8 9 12 13 15 18

## Intel Intel(R) Xeon(R) CPU E5-2608L v3 (Haswell, 4xdimms, 1 dimm = DDR4 4GB)

	reduced to 7 functions
	Bits: 1, sz=2
	22  (Correct: 93%)
	Bits: 2, sz=4
	19 23  (Correct: 90%)
	20 24  (Correct: 96%)
	21 25  (Correct: 96%)
	Bits: 3, sz=6
	Bits: 4, sz=6
	Bits: 5, sz=7
	8 13 15 17 27  (Correct: 90%)
	Bits: 6, sz=7
	7 12 14 16 18 26  (Correct: 96%)
	Bits: 7, sz=8
	Bits: 8, sz=11
	Bits: 9, sz=12
	Bits: 10, sz=12

	Inferred mapping
	22
	19 23
	20 24
	21 25
	8 13 15 17 27
	7 12 14 16 18 26

## Intel(R) Xeon(R) CPU E5-2658 v3 @ 2.20GHz (Haswell-EP, 8 dimms--3 have 2 ranks, 5 have 1 rank. *)

	reduced to 9 functions
	Bits: 1, sz=4
	15  (Correct: 98%)
	23  (Correct: 96%)
	Bits: 2, sz=10
	7 17  (Correct: 96%)
	20 24  (Correct: 92%)
	21 25  (Correct: 98%)
	22 26  (Correct: 100%)
	Bits: 3, sz=20
	Bits: 4, sz=31
	Bits: 5, sz=41
	8 12 14 16 18  (Correct: 95%)
	Bits: 6, sz=48
	Bits: 7, sz=50
	Bits: 8, sz=51
	Bits: 9, sz=51
	Bits: 10, sz=50

	Inferred mapping (may not be accurate?)
	15
	23
	7 17
	20 24
	21 25
	22 26
	8 12 14 16 18

## Raspberry Pi Zero 2 (Cortex-A53, LPDDR2, 512MB)

	reduced to 6 functions
	Bits: 1, sz=6
	12  (Correct: 100%)
	13  (Correct: 100%)
	14  (Correct: 100%)
	Bits: 2, sz=15
	Bits: 3, sz=20
	Bits: 4, sz=15
	Bits: 5, sz=6
	Bits: 6, sz=1
	Bits: 7, sz=0
	Bits: 8, sz=0
	Bits: 9, sz=0
	Bits: 10, sz=0

	Inferred mapping
	12
	13
	14

## Raspberry Pi 4 (Cortex-A72, LPDDR4, 2GB)

	$ sudo ./measure -s 8 -j 10 -t 120 -v 3
	..
	reduced to 4 functions
	Bits: 1, sz=4
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

	Inferred mapping
	12
	13
	14

## Raspberry Pi 5 (Cortex-A76, LPDDR4X, 4GB)

	sudo ./measure -s 8 -t 55 -m 1024 -v 3  -j 5

	reduced to 5 functions
	Bits: 1, sz=5
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

	Inferred mapping (SDRAM_BANKLOW = -1)
	12
	13
	14


## NVIDIA Jetson Nano (Cortex-A57, 64bit LPDDR4, 4GB)

	sudo ./measure -s 16 -j 20 -t 90 -v 3 > log.txt

	reduced to 5 functions
	Bits: 1, sz=1
	Bits: 2, sz=0
	Bits: 3, sz=0
	Bits: 4, sz=0
	Bits: 5, sz=0
	Bits: 6, sz=0
	Bits: 7, sz=0
	Bits: 8, sz=1
	13 19 20 21 24 25 26 28  (Correct: 100%)
	Bits: 9, sz=3
	10 12 14 16 17 21 25 27 28  (Correct: 100%)
	10 16 17 18 22 23 27 29 30  (Correct: 100%)
	Bits: 10, sz=5
	10 11 13 15 16 20 22 24 25 29  (Correct: 100%)


	Infered mapping:
	13 19 20 21 24 25 26 28
	10 12 14 16 17 21 25 27 28
	10 16 17 18 22 23 27 29 30
	10 11 13 15 16 20 22 24 25 29

## NVIDIA Jetson Orin AGX (Cortex-A78E, LPDDR5, 64GB)

	reduced to 10 functions
	Bits: 1, sz=2
	Bits: 2, sz=1
	Bits: 3, sz=0
	Bits: 4, sz=0
	Bits: 5, sz=0
	Bits: 6, sz=11
	9 13 17 18 21 22  (Correct: 100%)
	11 14 16 20 21 22  (Correct: 100%)
	11 12 13 16 19 24  (Correct: 100%)
	9 10 11 16 19 26  (Correct: 100%)
	14 15 17 21 25 28  (Correct: 100%)
	9 13 23 24 27 28  (Correct: 100%)
	9 12 17 21 25 29  (Correct: 100%)
	Bits: 7, sz=36
	10 11 12 17 19 20 23  (Correct: 100%)
	Bits: 8, sz=60
	Bits: 9, sz=92
	Bits: 10, sz=131
	Bank mapping functions saved to map.txt
	Finishing

	Infered mapping: (may not be accurate?)
	9 13 17 18 21 22
	11 14 16 20 21 22
	11 12 13 16 19 24
	9 10 11 16 19 26
	14 15 17 21 25 28
	9 13 23 24 27 28
	9 12 17 21 25 29
	10 11 12 17 19 20 23

