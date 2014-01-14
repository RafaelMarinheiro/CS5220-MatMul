# MPI ping-pong with IPM instrumentation

The Makefile in this directory illustrates:

1. How to build with instrumentation via IPM
2. How to generate a set of web pages of the profile
3. Optionally, how to serve up the web pages from C4.

You will need to have the `ipm` and `ploticus` modules loaded
in order to compile the code and build the web pages, respectively.
Once generated, you can download the `ping.x-ipm.tgz` tarball
to your own machine for perusal at your leisure.  

Alternately, you can serve the web page from C4; simply change to the web page
subdirectory and run

    python -m SimpleHTTPServer 8080

or whatever your favorite port number is.  You can also use `make serve` to do
the same thing via the Makefile.  Then from your own machine, use `ssh` to
forward port 8080 to a local port of your choosing.  On my OS X laptop, for
example, I would use

    ssh -N -L 8000:localhost:8080 c4

to map port 8080 on C4 to port 8000 on my own machine.  I can then view
the profile by pointing my web browser to `http://localhost:8000/`.

