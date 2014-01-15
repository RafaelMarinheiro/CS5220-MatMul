CC=mpicc
CFLAGS=-O2 -std=gnu99 -Wall
MPIRUN=mpisub
PORT=8080

.PHONY: run clean test

run: ping.x
	$(MPIRUN) -n 4 ./ping.x 3 500

profile:
	ipm_parse -html `ls *.ipm.xml`
	tar -czf ping.x-ipm.tgz ping.x_*.xml_ipm_unknown/

server:
	python -m SimpleHTTPServer $(PORT) 

ping.x: ping.c
	$(CC) $(CFLAGS) $< -o $@ -lipm

clean:
	rm -f ping.x ping.o mpisub*
	rm -rf *.ipm.xml
	rm -rf ping.x_*.xml_ipm_unknown
	rm -f ping.x-ipm.tgz
