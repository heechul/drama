
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

## Dell E6420 (i5-2520M, 16GB, 2-dimms, Sandybridge)

	sudo sysctl -w vm.nr_hugepages=1024
	sudo ./measure -r -m 2048 -t 390 > log
	tail log

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

## Intel E3-1220 v5 (4xdimms, 1 dimm = DDR4 4GB)

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
	bank bit 0: XOR (7 14)
	bank bit 1: XOR (15 19)
	bank bit 2: XOR (16 20)
	bank bit 3: XOR (17 21)
	bank bit 4: XOR (18 22)
	bank bit 5: XOR (8 9 12 13 15 18)

## Raspberry Pi 3

## Raspberry Pi 4 (2GB)

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

## Raspberry Pi 4 (4GB)

## Raspberry Pi 5 (4GB)



## NVIDIA Jetson Nano

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
13 19 20 21 24 25 26 28  (Correct: 93%)
Bits: 9, sz=3
10 12 14 16 17 21 25 27 28  (Correct: 93%)
10 16 17 18 22 23 27 29 30  (Correct: 93%)
Bits: 10, sz=5
10 11 13 15 16 20 22 24 25 29  (Correct: 93%)
Bits: 11, sz=7
Bits: 12, sz=6

Infered mapping:
  bank bit 0: XOR (13 19 20 21 24 25 26 28)
  bank bit 1: XOR (10 12 14 16 17 21 25 27 28)
  bank bit 2: XOR (10 16 17 18 22 23 27 29 30)
  bank bit 3: XOR (10 11 13 15 16 20 22 24 25 29)

## NVIDIA Jetson TX2

## NVIDIA Jetson Xavier

