CC=tau_cc.sh 
CFLAGS=-O2 -optMpi 

.PHONY: run clean test

run: ping.x
	mpisub -t "" -n 4 ./ping.x 3 500

view: profile.0.0.0
	pprof

pack: profile.0.0.0
	paraprof --pack ping.ppk

ping.x: ping.c
	$(CC) -std=gnu99 -Wall $(CFLAGS) $< -o $@

clean:
	rm -f ping.x ping.o mpisub* profile* 
