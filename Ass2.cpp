#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

// Print array function for debugging
void print_array(const vector<double>& array, const string& message, int rank) {
    cout << "Porcess: " << rank << " - " << message << ": ";
    for (double val : array) {
        cout << val << " ";
    }
    cout << endl;
}

// Implementation of parallel bucket sort
void parallel_bucket_sort(vector<double>& data, int n, int rank, int size,double total_parallel_time) {
    //Define time parameters for later analysis
    double start_time, end_time;
    double partition_time = 0.0, small_bucket_dist_time = 0.0; 
    double large_bucket_comm_time = 0.0, large_bucket_sort_time = 0.0, gather_time = 0.0;


     // Phase 1: Data Partitioning
    start_time = MPI_Wtime();
    // The amount of data processed by each process
    int local_n = n / size + (rank < n % size ? 1 : 0);
    vector<double> local_data(local_n);

    // sendcounts and displs for data distribution
    vector<int> sendcounts(size), displs(size);
    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = n / size + (i < n % size ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Distributing Data Using MPI_Scatterv
    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_data.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    partition_time = end_time - start_time;   //Data partition time
    // print_array(local_data, "Received data ", rank);

    // Phase 2: Bucket Allocation
    start_time = MPI_Wtime();
    // Each process puts data into buckets according to the data range
    int num_buckets = size;
    vector<vector<double>> buckets(num_buckets);
    double bucket_range = 10000 / num_buckets;
    for (double val : local_data) {
        int bucket_index = min(num_buckets - 1, int(val / bucket_range));
        buckets[bucket_index].push_back(val);
    }
    end_time = MPI_Wtime();
    small_bucket_dist_time = end_time - start_time;   //Bucket allocation time

    // Phase 3: Bulk Data Exchange
    start_time = MPI_Wtime();
    // Prepare sendbuf and recvbuf for bucket data exchange
    vector<int> sendcounts_all(size), recvcounts_all(size);
    vector<int> send_displs(size), recv_displs(size);
    vector<double> sendbuf, recvbuf;

    // Merge the data of each bucket into sendbuf and record the send count of each process
    for (int i = 0; i < size; ++i) {
        sendcounts_all[i] = buckets[i].size();
    }
    MPI_Alltoall(sendcounts_all.data(), 1, MPI_INT, recvcounts_all.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int send_offset = 0;
    for (int i = 0; i < size; ++i) {
        send_displs[i] = send_offset;
        send_offset += sendcounts_all[i];
        sendbuf.insert(sendbuf.end(), buckets[i].begin(), buckets[i].end());
    }

    int recv_total = 0;
    for (int i = 0; i < size; ++i) {
        recv_displs[i] = recv_total;
        recv_total += recvcounts_all[i];
    }
    recvbuf.resize(recv_total);
    MPI_Alltoallv(sendbuf.data(), sendcounts_all.data(), send_displs.data(), MPI_DOUBLE, recvbuf.data(), recvcounts_all.data(), recv_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    large_bucket_comm_time = end_time - start_time;  //Bucket data exchange time
    // print_array(recvbuf, "Data after exchange ", rank);

    // Phase 4: Bucket sorting
    start_time = MPI_Wtime();
    // Each process sorts the received data (the sort method used is the C++ std standard library method, and the quick sort method is used by default)
    sort(recvbuf.begin(), recvbuf.end());
    end_time = MPI_Wtime();
    large_bucket_sort_time = end_time - start_time;  //Bucket sorting time
    // print_array(recvbuf, "Sorted local data ", rank);

    // Phase 5: Collecting sorting results
    start_time = MPI_Wtime();
    // Use MPI_Gather to collect the recvbuf.size() of each process to the master process
    int recvbuf_size = recvbuf.size();
    MPI_Gather(&recvbuf_size, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // The main process computes displs for MPI_Gatherv
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }

        //Print sendcounts and displs
        // cout << " - sendcounts: ";
        // for (int val : sendcounts) {
        //     cout << val << " ";
        // }
        // cout << endl;

        // cout << " - displs: ";
        // for (int val : displs) {
        //     cout << val << " ";
        // }
        // cout << endl;
    }

    // Use MPI_Gatherv to gather all sorted data
    vector<double> gathered_data;
    if (rank == 0) {
        gathered_data.resize(n);
    }
    MPI_Gatherv(recvbuf.data(), recvbuf.size(), MPI_DOUBLE, gathered_data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    gather_time = end_time - start_time;   //Time to collect sort results


    // Calculate the total parallel time
    total_parallel_time = partition_time + small_bucket_dist_time + large_bucket_comm_time + large_bucket_sort_time + gather_time;

    // The main process prints the time of each stage
    if (rank == 0) {
        // print_array(gathered_data, "The final sorted data ", rank);

        cout << "Time of each stage (unit: seconds):" << endl;
        cout << "1. Partition phase time: " << partition_time << endl;
        cout << "2. Bucket allocation phase time: " << small_bucket_dist_time << endl;
        cout << "3. Large bucket data exchange phase time: " << large_bucket_comm_time << endl;
        cout << "4. Bucket sorting phase time: " << large_bucket_sort_time << endl;
        cout << "5. Results collection phase time: " << gather_time << endl;
        cout << "Total parallel time: " << total_parallel_time << endl;
    }
}

int main(int argc, char* argv[]) {
    
    double total_parallel_time = 0.0;  //Define total parallel time parameters
    double sequential_time =0.0; //Defining Serial Time
    //Initialize the MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check command line arguments
    int n;
    if (argc > 1) {
        n = atoi(argv[1]); // Get the value of n from the command line
    } else {
        if (rank == 0) {
            cerr << "usage: " << argv[0] << " <Data quantity>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Make sure n is a valid positive integer
    if (n <= 0) {
        if (rank == 0) {
            cerr << "Error: The number of data n must be a positive integer." << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    vector<double> data;
    if (rank == 0) {
        data.resize(n);
        srand(time(0));
       // The Master process generates random data
        for (int i = 0; i < n; ++i) {
            data[i] = (double)rand() / RAND_MAX * 10000;
           
        }

        // Print initialization data (check whether the generation is successful)
        // cout << "Print initialization data: ";
        // for (int i = 0; i < n; ++i) {
        //     cout << data[i] << " ";
        // }
        // cout << endl;   
     }
    
    //If it is a single process, the serial time of the output should be calculated additionally.
    if (size == 1) {
        // Single process mode: measures the entire program as a serial time
        double start_time = MPI_Wtime();
        parallel_bucket_sort(data, n, rank, size, total_parallel_time);
        double end_time = MPI_Wtime();
        sequential_time = end_time - start_time;
        
        if (rank == 0) {
            cout << "Total serial time (single process): " << sequential_time << " seconds" << endl;
        }
    } else {
        // Multi-process mode: record the total parallel time
        parallel_bucket_sort(data, n, rank, size, total_parallel_time);

    }
    

    MPI_Finalize();
    return 0;
}
