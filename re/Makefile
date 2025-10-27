# Detect architecture and add SSE4.1 flag for x86/x86_64
ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
    ARCH_FLAGS = -msse4.1
else
    ARCH_FLAGS =
endif

all: measure onebank_attacker allbank_attacker

measure: measure.cpp
	g++ -g -std=c++11 -lm -O3 $(ARCH_FLAGS) measure.cpp -o measure -lpthread

onebank_attacker: onebank_attacker.cpp
	g++ -g -std=c++11 -lm -O3 $(ARCH_FLAGS) onebank_attacker.cpp -o onebank_attacker -lpthread

allbank_attacker: allbank_attacker.cpp
	g++ -g -std=c++11 -lm -O3 $(ARCH_FLAGS) allbank_attacker.cpp -o allbank_attacker -lpthread

clean:
	rm -f measure onebank_attacker allbank_attacker
