#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>

// Comparison function, used for sorting with qsort
int compare(const void* x1, const void* x2) {
    const float* f1 = (const float*)x1;
    const float* f2 = (const float*)x2;
    return (*f1 > *f2) - (*f1 < *f2);
}

// Check if the data is sorted and calculate the sum of the data
void check(float* data, int nitems) {
    double sum = 0;
    int sorted = 1;
    for (int i = 0; i < nitems; i++) {
        sum += data[i];
        if (i && data[i] < data[i - 1]) sorted = 0;
    }
    printf("sum=%f, sorted=%d\n", sum, sorted);
}

int main(int argc, char* argv[]) {
    int rank, size, nitems = 20;
    const float xmin = 0.0, xmax = 1000.0;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // If the command line argument provides the number of items, it is used.
    if (argc == 2) {
        nitems = atoi(argv[1]);
    }

    // Record the time of each stage
    double start_time, init_time, scatter_time, local_sort_time, gather_time, total_time;

    // Start total time recording
    start_time = MPI_Wtime();

    // Initialize data allocation and offsets
    std::vector<int> send_counts(size, nitems / size); // Initialize the number of elements allocated to each process
    std::vector<int> displs(size, 0); // Initialize the starting offset of each process
    int remainder = nitems % size; // Calculate the remainder, making sure all data is allocated

    // Distribute the remaining elements to the first `remainder` processes
    for (int i = 0; i < remainder; i++) {
        send_counts[i]++;
    }

    // Calculate the starting offset of each process
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + send_counts[i - 1];
    }

    // Initialize Data
    float* data = NULL;
    if (rank == 0) {
        data = (float*)malloc(nitems * sizeof(float));
        for (int i = 0; i < nitems; i++) {
            data[i] = (float)rand() / RAND_MAX * (xmax - xmin) + xmin; // Generate random numbers
        }
        check(data, nitems); // Output data sum and sorting status
    }

    // Record the time of the initialization phase
    init_time = MPI_Wtime() - start_time;

    // Allocate a local array for each process to store received data
    float* local_data = (float*)malloc(send_counts[rank] * sizeof(float));

    // Record the start time of the Scatterv phase
    double scatter_start = MPI_Wtime();

    // Use MPI_Scatterv to distribute data to each process on demand
    MPI_Scatterv(data, send_counts.data(), displs.data(), MPI_FLOAT, local_data, send_counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Record the time of the Scatterv stage
    scatter_time = MPI_Wtime() - scatter_start;

    // Output the data received by each process
    printf("Process %d received data: ", rank);
    for (int i = 0; i < send_counts[rank]; i++) {
        printf("%f ", local_data[i]);
    }
    printf("\n");

    // Record the start time of local sorting
    double local_sort_start = MPI_Wtime();

    // Sort each process's local data
    std::sort(local_data, local_data + send_counts[rank], std::less<float>());

    // Record the time of local sorting
    local_sort_time = MPI_Wtime() - local_sort_start;

    // Array prepared by the root process to collect all data
    float* gathered_data = NULL;
    if (rank == 0) {
        gathered_data = (float*)malloc(nitems * sizeof(float));
    }

    // Record the start time of the Gather phase
    double gather_start = MPI_Wtime();

    // Use MPI_Gatherv to gather data from each process to the root process
    MPI_Gatherv(local_data, send_counts[rank], MPI_FLOAT, gathered_data, send_counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Record the time of the Gather phase
    gather_time = MPI_Wtime() - gather_start;

    // The root process globally sorts the collected data and outputs it
    if (rank == 0) {
        // Sort the collected data
        std::sort(gathered_data, gathered_data + nitems);  // Sort only actual data
        printf("Sorted data after gather:\n");
        for (int i = 0; i < nitems; i++) {
            printf("%f ", gathered_data[i]);
        }
        printf("\n");

        // Free the collection array of the root process
        free(gathered_data); 
    }

    // Total recording time
    total_time = MPI_Wtime() - start_time;

    // The root process calculates and outputs the speedup ratio
    if (rank == 0) {
        // Calculate the proportion alpha of the serial part
        double alpha = (init_time + gather_time) / total_time;
        
        //Calculate speedup using Gustafson's Law
        double speedup = size - alpha * (size - 1);

        // Output timing statistics and speedup
        printf("\nTime taken for each phase:\n");
        printf("Initialization Time: %f seconds\n", init_time);
        printf("Scatter Time: %f seconds\n", scatter_time);
        printf("Local Sort Time: %f seconds\n", local_sort_time);
        printf("Gather Time: %f seconds\n", gather_time);
        printf("Total Execution Time: %f seconds\n", total_time);
        printf("Calculated Speedup (Gustafson's Law): %f\n", speedup);
    }

    // Freeing allocated memory
    free(data);
    free(local_data);
    // End the MPI environment
    MPI_Finalize();
    return 0;
}
