.PHONY: run clean

run: ping.x
	mpisub -n 2 ./ping.x 1 100
	mpisub -n 2 -f req1.txt ./ping.x 1 100

ping.x: ping.c
	mpicc -std=gnu99 -Wall $< -o $@

clean:
	rm -f ping.x mpisub*

