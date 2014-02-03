import matplotlib.pyplot as plt
import numpy as np

def make_speedup_plot(n, rmax, tc, tt):
    """Plots speedup for student counting exercise.

    Plots the speedup for counting a class of students in
    parallel by rows (assuming each row has equal student counts)
    vs just counting the students.

    Args:
        n: Number of students in the class
        rmax: Maximum number of rows to consider
        tc: Time to count one student
        tt: Time to add a row count into a tally
    """
    r = np.arange(1,rmax+1)
    ts = n*tc
    tp = ts/r + r*tt
    speedup = ts/tp
    plt.plot(r, speedup)
    plt.xlabel('Number of rows')
    plt.ylabel('Speedup')

def make_speedup_file(fname, n, rmax, tc, tt):
    """Plots speedup for student counting exercise.

    This is exactly like make_speedup_plot, but instead of generating
    a plot directly, we generate a data file to be read by pgfplots.

    Args:
        fname: Output file name
        n: Number of students in the class
        rmax: Maximum number of rows to consider
        tc: Time to count one student
        tt: Time to add a row count into a tally
    """
    ts = n*tc
    with open(fname, 'w') as f:
        for r in range(1,rmax+1):
            tp = ts/r + r*tt
            f.write('%d %g\n' % (r, ts/tp))

if __name__ == "__main__":
    n = 100
    rmax = 12
    tc = 0.5
    tt = 3
    make_speedup_file('lec01plot.dat', n, rmax, tc, tt)
    make_speedup_plot(n, rmax, tc, tt)
    plt.savefig('lec01plot.pdf')
