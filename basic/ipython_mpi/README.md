# IPython.parallel with MPI on HTCondor

You should load the Anaconda module and run `setup_ipython_mpi.sh`
before running this example.  Once you have done so, you can run
the demo program as follows:

1. Run `make start` to start the ipcluster.
2. Wait.  Check the queue to make sure your job is running.
3. Run `make run` to actually run the demo.
4. Run `make stop` to shut down the ipcluster.

Note that you can enter the commands in `runner.py` interactively,
and this is typically how you would actually run iPython.

This demo is [taken from the IPython docs](http://ipython.org/ipython-doc/dev/parallel/parallel_mpi.html).
