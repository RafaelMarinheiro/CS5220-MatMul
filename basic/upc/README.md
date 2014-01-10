# UPC hello world

This is a bunch of different UPC subroutines that I put together a while
ago.  The uncommented ones print "Hello world" and do a stupid
Monte Carlo calculation of pi.  The usage is simply

    ./foo

As it is currently set up, running `make` will build the program and
launch it on the cluster with upcrun.  You *must* load the `upc`
module before running `make`!

