run: foo
	upcsub -n 2 ./foo

run2: foo
	upcsub -n 2 -f req1.txt ./foo

run3: foo_x2
	upcsub -n 2x2 ./foo_x2

foo: foo.upc
	upcc --network=udp $< -o $@

foo_x2: foo.upc
	upcc --network=udp --pthreads=2 $< -o $@

clean:
	rm -f foo upcsub* foo_x2*
