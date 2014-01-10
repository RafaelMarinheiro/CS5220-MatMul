#include <upc.h>  /* Required for UPC extensions */
#include <bupc_collectivev.h>
#include <stdio.h>
#include <stdlib.h>

int trial_in_disk()
{
    double x = (double) rand() / RAND_MAX;
    double y = (double) rand() / RAND_MAX;
    return (x*x + y*y < 1) ? 1 : 0;
}

/*****************************************************************/

shared int all_hits[THREADS];
void piv1() {
    int i, hits = 0, tot = 0, trials = 1000000;   
    srand(1+MYTHREAD*17);
    for (i = 0; i < trials; ++i) 
        hits += trial_in_disk();
    all_hits[MYTHREAD] = hits;
    upc_barrier;
    if (MYTHREAD == 0) {
        for (i = 0; i < THREADS; ++i)
            tot += all_hits[i];
        printf("Pi approx %g\n", 4.0*tot/trials/THREADS);
    }
}

/*****************************************************************/

shared int tot;
void piv2() {
    int i, hits = 0, trials = 1000000;
    upc_lock_t* tot_lock = upc_all_lock_alloc();
    srand(1+MYTHREAD*17);
    for (i = 0; i < trials; ++i) 
        hits += trial_in_disk();
    upc_lock(tot_lock); 
    tot += hits; 
    upc_unlock(tot_lock);
    upc_barrier;
    if (MYTHREAD == 0) {
        printf("Pi approx %g\n", 4.0*tot/trials/THREADS);
        upc_lock_free(tot_lock);
    }
}

/*****************************************************************/

void piv3() {
    int i, hits, trials = 1000000;
    srand(1+MYTHREAD*17);
    for (i = 0; i < trials; ++i) 
        hits += trial_in_disk();
    hits = bupc_allv_reduce(int, hits, 0, UPC_ADD);
    if (MYTHREAD == 0) 
        printf("Pi approx %g\n", 4.0*tot/trials/THREADS);
}

/*****************************************************************/

#define N (10*THREADS)
#define h (1.0/N)

shared[*] double u_old[N], u[N], f[N];  /* Block layout */
void jacobi_sweeps(int nsweeps) {
    int i, it;
    for (it = 0; it < nsweeps; ++it) {
        upc_forall(i=1; i < N-1; ++i; &(u[i]))
            u[i] = (u_old[i-1] + u_old[i+1] - h*h*f[i])/2;
        upc_barrier;
        upc_forall(i=0; i < N; ++i; &(u[i]))
            u_old[i] = u[i];
        upc_barrier;
    }
}

void test_jacobi()
{
    int i;
    upc_forall (i = 0; i < N; ++i; &(u_old[i])) {
        u_old[i] = 0;
        f[i] = 2;
    }
    upc_barrier;
    jacobi_sweeps(5000);
    upc_barrier;
    if (MYTHREAD == 0) {
        double my_err = 0.0;
        for (i = 0; i < N; ++i) {
            double x = (double) i/N;
            double diff = (u[i] - x*(x-1)); 
            my_err += diff*diff;
        }
        my_err = sqrt(my_err);
        printf("Jacobi final error: %g\n", my_err);
    }
}

/*****************************************************************/

#define N_PER 20

shared double ubound[2][THREADS];  /* For ghost cells*/
double uold[N_PER+2], uloc[N_PER+2], floc[N_PER+2];
void jacobi_sweeps2(int nsweeps) {
    int i, it;
    double h1 = (1./(N_PER*THREADS+1));
    double h2 = h1*h1;

    for (it = 0; it < nsweeps; ++it) {

        /* Set boundary data */
        if (MYTHREAD > 0)       ubound[1][MYTHREAD-1] = uold[1];
        if (MYTHREAD < THREADS) ubound[0][MYTHREAD+1] = uold[N_PER];
        upc_barrier;
        
        /* Get boundary data */
        uold[0]       = ubound[0][MYTHREAD];
        uold[N_PER+1] = ubound[1][MYTHREAD];

        /* Compute */
        for (i = 1; i < N_PER+1; ++i)
            uloc[i] = (uold[i-1] + uold[i+1] - h2*floc[i])/2;

        /* Copy */
        for (i = 1; i < N_PER+1; ++i)
            uold[i] = uloc[i];
    }
}

void jacobi_sweeps3(int nsweeps) {
    int i, it;
    double h1 = (1./(N_PER*THREADS+1));
    double h2 = h1*h1;

    for (it = 0; it < nsweeps; ++it) {

        /* Set boundary data */
        if (MYTHREAD > 0)       ubound[1][MYTHREAD-1] = uold[1];
        if (MYTHREAD < THREADS) ubound[0][MYTHREAD+1] = uold[N_PER];

        /* Process interior */
        upc_notify;
        for (i = 2; i < N_PER; ++i)
            uloc[i] = (uold[i-1] + uold[i+1] - h2*floc[i])/2;        
        upc_wait;

        /* Get boundary data */
        uold[0]       = ubound[0][MYTHREAD];
        uold[N_PER+1] = ubound[1][MYTHREAD];

        /* Process boundary */
        for (i = 1; i < N_PER+1; i += N_PER)
            uloc[i] = (uold[i-1] + uold[i+1] - h2*floc[i])/2;

        /* Copy */
        for (i = 1; i < N_PER+1; ++i)
            uold[i] = uloc[i];
    }
}


void test_jacobi2()
{
    int i;
    for (i = 0; i < N_PER+2; ++i) {
        uold[i] = 0;
        floc[i] = 2;
    }
    jacobi_sweeps3(5000);
    upc_barrier;
    double my_err = 0.0;
    for (i = 1; i < N_PER+1; ++i) {
        double x = (double) ((MYTHREAD*N_PER)+i)/(N_PER*THREADS+1);
        double diff = (u[i] - x*(x-1)); 
        my_err += diff*diff;
    }
    my_err = sqrt(my_err);
    printf("%d: Jacobi final error: %g\n", MYTHREAD, my_err);
}

/*****************************************************************/

typedef struct list_t {
    int x;
    shared struct list_t* next;
} list_t;

shared struct list_t* shared head;
upc_lock_t* list_lock;

void push(int x) {
    shared list_t* item = upc_global_alloc(1, sizeof(list_t));
    upc_lock(list_lock);
    item->x = x;
    item->next = head;
    head = item;    
    upc_unlock(list_lock);
}

int pop(int* x) {
    shared list_t* item;
    upc_lock(list_lock);
    if (head == NULL) {
        upc_unlock(list_lock);
        return -1;
    }
    item = head;
    head = head->next;
    *x = item->x;
    upc_free(item);
    upc_unlock(list_lock);
    return 0;
}

void test_list()
{
    int i;
    int x;
    upc_lock_t* list_lock = upc_all_lock_alloc();
    upc_forall(i = 0; i < 100; ++i; i)
        push(i);
    upc_barrier;
    while (pop(&x) == 0)
        printf("Pop %d on %d\n", x, MYTHREAD);
    upc_barrier;
}

/*****************************************************************/

int main() 
{
    printf("Hello from %d of %d\n", 
           MYTHREAD, THREADS);
    piv1();
    piv2();
    piv3();
    upc_barrier;
//    test_list();
//    upc_barrier;
//    test_jacobi();
//    test_jacobi2();
}  
