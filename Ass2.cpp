#if 1
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cstring>
#define NDATA 5


// The comparison function to use with the library qsort
// function. This will tell qsort to sort the numbers in ascending order.
int compare(const void* x1, const void* x2) {
    const float* f1 = (const float*)x1;
    const float* f2 = (const float*)x2;
    float diff = *f1 - *f2;

    return (diff < 0) ? -1 : 1;
}

// Create a data array to hold the given number of buckets for the
// given number of total data items. All buckets are held contiguously in the busket
float* create_buckets(int nbuckets, int nitems)
{
    int i;

    int ntotal = nbuckets * nitems;

    // Pointer to an array of more pointers to each bucket
    // float* bucket = (float*)calloc(ntotal, sizeof(float*));
    float* bucket = (float*)calloc(ntotal, sizeof(float));

    for (i = 0; i < ntotal; ++i) bucket[i] = 0;

    // return the address of the array of pointers to float arrays
    return bucket;
}


// Instead of drand48(), use this
double drand48_random() {
    return static_cast<double>(rand()) / RAND_MAX;
}

void check(float* data, int nitems) {
    double sum = 0;
    int sorted = 1;
    int i;

    for (i = 0;i < nitems;i++) {
        sum += data[i];
        if (i && data[i] < data[i - 1]) sorted = 0;
    }
    printf("sum=%f, sorted=%d\n", sum, sorted);
}

// Sequential implementation of the bucket sort routine. The full
// range x1 to x2 will be divided into a number of equally spaced
// subranges according to the number of buckets. All the buckets are
// contained in the single one dimensional array "bucket".
void bucket_sort(float* data, int ndata, float x1, float x2, int nbuckets, float* bucket)
{
    int i, count;

    // The range covered by one bucket
    float stepsize = (x2 - x1) / nbuckets;

    // The number of items thrown into each bucket. We would expect each
    // bucket to have a similar number of items, but they won't be
    // exactly the same. So we keep track of their numbers here.
    int* nitems = (int*)malloc(nbuckets * sizeof(int));
    for (i = 0; i < nbuckets; ++i) nitems[i] = 0;

    // Toss the data items into the correct bucket
    for (i = 0; i < ndata; ++i) {

        // What bucket does this data value belong to?
        int bktno = (int)floor((data[i] - x1) / stepsize);
        int idx = bktno * ndata + nitems[bktno];

        printf("DATA %d %f %d %d\n", i, data[i], bktno, idx);

        // Put the data value into this bucket
        bucket[idx] = data[i];
        ++nitems[bktno];
    }

    // Sort each bucket using the standard library qsort routine. Note
    // that we need to input the correct number of items in each bucket
    count = 0;
    for (i = 0; i < nbuckets; ++i) {
        if (nitems[i]) {
            qsort(&bucket[i * ndata], nitems[i], sizeof(float), compare);
            memcpy(data, &bucket[i * ndata], nitems[i] * sizeof(float));
            data += nitems[i];
        }
    }

    // Don't need the number of items anymore
    free(nitems);

}



// Ass 2 bucket sort
int main(int argc, char* argv[]) {
    int rank=0, size=0, nitems = 20, nbuckets = 4;
    const float xmin = 0.0, xmax = 1000.0;

    int recvbuf[NDATA];
    int* sendbuf;
    int numproc=0, myid=0, i, N=0, root;
    float* buckets = NULL;

    double start_time, end_time, t_total = 0;


    //Init MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc == 2) {
        nitems = atoi(argv[1]);
    }

    float* data = (float*)malloc(nitems * sizeof(float));

    // Master process is rank equals zero and it creates random data
    if (rank == 0) {
        
        for (int i = 0;i < nitems;i++){
            data[i] = drand48_random() * (xmax - xmin - 1) + xmin;
        }
        check(data, nitems);
       buckets = create_buckets(nbuckets, nitems);    
      

    }
    // Start the timer
    start_time = MPI_Wtime();

    // Fill up the array with data to send to the destination node. Note
    // that the contents of the array will
    sendbuf = (int*)malloc(NDATA * numproc * sizeof(int));
    for (i = 0; i < NDATA * numproc; ++i) sendbuf[i] = i;

    root = 0;
    MPI_Scatter(sendbuf, NDATA, MPI_INT, recvbuf, NDATA, MPI_INT, root,MPI_COMM_WORLD);
    
    //bucket sort
    bucket_sort(data, nitems, xmin, xmax, nbuckets, create_buckets(nbuckets, nitems));

    MPI_Alltoall(sendbuf, NDATA, MPI_INT, recvbuf, NDATA, MPI_INT,MPI_COMM_WORLD);
    // Slaves gather the data back to the master(rank == 0)
    MPI_Gather(data, nitems, MPI_FLOAT, buckets, nitems, MPI_FLOAT, root, MPI_COMM_WORLD);

    // Master sorts all of the data
    if (rank == 0) {

        // Stop the timer
        end_time = MPI_Wtime();

        //Total Execution time
        t_total = end_time - start_time;

        // sort data
        qsort(buckets, nitems * size, sizeof(float), compare);

        // print the sorted data
        printf("Sorted data after gather:\n");
        for (int i = 0; i < nitems * size; i++) {
            printf("%f ", buckets[i]);
        }
        printf("\n");

        printf("Total Execution time(t_total): %f seconds\n", t_total);

        free(buckets);  // release buckets
    }

    free(data); // release data
    free(sendbuf); // release sendbuf
    
    MPI_Finalize();
    return 0;
}

#endif

