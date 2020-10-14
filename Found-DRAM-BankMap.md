
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
