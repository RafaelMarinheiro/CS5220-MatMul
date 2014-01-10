run: foo
	upcsub -n 2 ./foo

foo: foo.upc
	upcc --network=udp $< -o $@

clean:
	rm -f foo upcsub*
