all: measure onebank_attacker

measure: measure.cpp
	g++ -g -std=c++11 -lm -O3 measure.cpp -o measure -lpthread

onebank_attacker: onebank_attacker.cpp
	g++ -g -std=c++11 -lm -O3 onebank_attacker.cpp -o onebank_attacker -lpthread

allbank_attacker: allbank_attacker.cpp
	g++ -g -std=c++11 -lm -O3 allbank_attacker.cpp -o allbank_attacker -lpthread

clean:
	rm -f measure onebank_attacker allbank_attacker
