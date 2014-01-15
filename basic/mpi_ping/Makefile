CC=mpicc
CFLAGS=-std=gnu99 -Wall
MPIRUN=mpisub

.PHONY: run run2 clean

run: ping.x
	$(MPIRUN) -n 2 ./ping.x 1 100

run2: ping.x
	$(MPIRUN) -n 2 -f req1.txt ./ping.x 1 100

ping.x: ping.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f ping.x mpisub*

