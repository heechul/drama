all: measure onebank_attacker

measure: measure.cpp
	g++ -g -std=c++11 -lm -O3 measure.cpp -o measure -lpthread

onebank_attacker: onebank_attacker.cpp
	g++ -g -std=c++11 -lm -O3 onebank_attacker.cpp -o onebank_attacker -lpthread

clean:
	rm -f measure onebank_attacker