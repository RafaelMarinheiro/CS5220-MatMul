run:
	ipython runner.ipy

start:
	ipcluster start --profile=mpi -n 2 --daemonize=True

stop:
	ipcluster stop --profile=mpi

clean:
	rm -f htcondor_engines machines
