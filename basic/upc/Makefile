run: foo
	upcsub -n 2 ./foo

run2: foo
	upcsub -n 2 -f req1.txt ./foo

foo: foo.upc
	upcc --network=udp $< -o $@

clean:
	rm -f foo upcsub*
