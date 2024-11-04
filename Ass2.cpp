#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <numeric> // for std::accumulate

// Assignment 2 bucket
int main(int argc, char* argv[]) {
    int rank, size, nitems = 20;
    const float xmin = 0.0, xmax = 1000.0;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    // If the command line argument provides the number of items, it is used.
    if (argc == 2) {
        nitems = atoi(argv[1]);
    }

    // Initialize data allocation and offsets
    std::vector<int> send_counts(size, nitems / size); 
    std::vector<int> displs(size, 0);
    int remainder = nitems % size;

    for (int i = 0; i < remainder; i++) {
        send_counts[i]++;
    }

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
    }

    float* local_data = (float*)malloc(send_counts[rank] * sizeof(float));

    // Scatter data to each process
    MPI_Scatterv(data, send_counts.data(), displs.data(), MPI_FLOAT, local_data, send_counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Divide local data into buckets for each process based on data range
    std::vector<std::vector<float>> buckets(size);
    for (int i = 0; i < send_counts[rank]; i++) {
        int bucket_idx = (local_data[i] - xmin) / (xmax - xmin) * size;
        bucket_idx = std::min(bucket_idx, size - 1);  // Ensure the index is within bounds
        buckets[bucket_idx].push_back(local_data[i]);
    }

    // Prepare data for MPI_Alltoallv
    std::vector<int> send_counts_alltoall(size, 0);
    std::vector<int> send_displs_alltoall(size, 0);
    std::vector<float> send_buffer;

    for (int i = 0; i < size; i++) {
        send_counts_alltoall[i] = buckets[i].size();
        send_displs_alltoall[i] = send_buffer.size();
        send_buffer.insert(send_buffer.end(), buckets[i].begin(), buckets[i].end());
    }

    // Determine receive counts and displacements
    std::vector<int> recv_counts_alltoall(size);
    MPI_Alltoall(send_counts_alltoall.data(), 1, MPI_INT, recv_counts_alltoall.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv_count = std::accumulate(recv_counts_alltoall.begin(), recv_counts_alltoall.end(), 0);
    std::vector<float> recv_buffer(total_recv_count);
    std::vector<int> recv_displs_alltoall(size, 0);
    for (int i = 1; i < size; i++) {
        recv_displs_alltoall[i] = recv_displs_alltoall[i - 1] + recv_counts_alltoall[i - 1];
    }

    // Use MPI_Alltoallv to exchange data
    MPI_Alltoallv(send_buffer.data(), send_counts_alltoall.data(), send_displs_alltoall.data(), MPI_FLOAT,
                  recv_buffer.data(), recv_counts_alltoall.data(), recv_displs_alltoall.data(), MPI_FLOAT, MPI_COMM_WORLD);

    // Sort received data in the large bucket (each process's final sorted data)
    std::sort(recv_buffer.begin(), recv_buffer.end());

    // Gather the sorted data back to the master process
    float* gathered_data = NULL;
    if (rank == 0) {
        gathered_data = (float*)malloc(nitems * sizeof(float));
    }
    MPI_Gatherv(recv_buffer.data(), recv_buffer.size(), MPI_FLOAT, gathered_data, send_counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Optionally, sort the final gathered data
        std::sort(gathered_data, gathered_data + nitems);
        
        // Print final sorted result (for verification purposes)
        printf("Sorted data:\n");
        for (int i = 0; i < nitems; i++) {
            printf("%f ", gathered_data[i]);
        }
        printf("\n");

        // Free the gathered data buffer on the root process
        free(gathered_data); 
    }

    free(data);
    free(local_data);
    
    MPI_Finalize();
    return 0;
}
