#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MASTER 0
#define M 4294967296ULL // 2^32
#define MAX_ENTRIES 1000

typedef unsigned long long ULLONG;

typedef struct {
    int k;
    ULLONG k_a;
    ULLONG k_c;
    int flag;
} JumpConstant;

int readFile(const char* file_path, JumpConstant jumpConstants[], int* count) {
    FILE* file = fopen(file_path, "r");

    if (file == NULL) {
        fprintf(stderr, "Error: Could not open the file.\n");
        return 1;
    }

    char line[100];
    *count = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (sscanf(line, "%d %llu %llu %d", &jumpConstants[*count].k, &jumpConstants[*count].k_a, &jumpConstants[*count].k_c, &jumpConstants[*count].flag) == 4) {
            (*count)++;
            if (*count >= MAX_ENTRIES) {
                fprintf(stderr, "Warning: Maximum number of entries reached.\n");
                break;
            }
        }
        else {
            fprintf(stderr, "Error: Could not parse line: %s", line);
        }
    }

    fclose(file);
    return 0;
}

// Evaluate (ax+c) mod m
ULLONG modlin(ULLONG a, ULLONG x, ULLONG c, ULLONG m) {
    return (a * x + c) % m;
}

// Scale integer n to the range [x1, x2]
double rescale(ULLONG N, ULLONG n, double x1, double x2) {
    double f = (double)n / (double)N;
    return x1 + f * (x2 - x1);
}

int main(int argc, char* argv[]) {
    int num_tasks, task_id;
    ULLONG total_num_points, points_in_circle = 0;
    ULLONG total_points_in_circle;
    double x, y, pi_estimate;
    ULLONG seed;
    ULLONG a = 0, c = 0;
    const ULLONG sidelen = 65536; // sqrt of m

    double start_time, end_time, t_total;
    double t_serial1 = 0, t_serial2 = 0, t_parallel = 0;
    double s_t1, e_t1, s_t2, e_t2, s_parallel, e_parallel;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    if (argc != 2) {
        if (task_id == MASTER) printf("Usage: %s <total number of points>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    // The total number of points across all processes
    total_num_points = atoll(argv[1]);

    if (task_id == MASTER) {
        start_time = MPI_Wtime();  // Record program start time

        // Generate random seed time
        s_t1 = MPI_Wtime();
        seed = (ULLONG)time(NULL);
        e_t1 = MPI_Wtime();
        t_serial1 = e_t1 - s_t1;

        // Read jump constants from a file
        JumpConstant jumpConstants[MAX_ENTRIES];
        int count = 0;
        const char* file_path = "jumpconstants.dat";
        if (readFile(file_path, jumpConstants, &count) != 0) {
            fprintf(stderr, "Error: Could not read the file properly.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < count; i++) {
            if (jumpConstants[i].k == num_tasks) {
                a = jumpConstants[i].k_a;
                c = jumpConstants[i].k_c;
                break;
            }
        }

        // The communication time for the Master process to send the seed and jump constant
        s_t2 = MPI_Wtime();
        for (int i = 1; i < num_tasks; i++) {
            MPI_Send(&seed, 1, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD);
            MPI_Send(&a, 1, MPI_UNSIGNED_LONG_LONG, i, 1, MPI_COMM_WORLD);
            MPI_Send(&c, 1, MPI_UNSIGNED_LONG_LONG, i, 2, MPI_COMM_WORLD);
        }
        e_t2 = MPI_Wtime();
        t_serial2 = e_t2 - s_t2;
    }
    else {
        // The slave process receives the seed, a, and c sent by the master process
        MPI_Recv(&seed, 1, MPI_UNSIGNED_LONG_LONG, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&a, 1, MPI_UNSIGNED_LONG_LONG, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&c, 1, MPI_UNSIGNED_LONG_LONG, MASTER, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process %d: Received seed = %llu, a = %llu, c = %llu\n", task_id, seed, a, c);
        fflush(stdout);
    }

    // Start time of the parallel part
    s_parallel = MPI_Wtime();
    printf("Process %d: Starting computation in round-robin with total points %llu.\n", task_id, total_num_points);
    fflush(stdout);

    // Each process handles points assigned in a round-robin fashion
    ULLONG i_prev = seed + task_id; // Initialize i_prev with a unique seed based on task_id
    ULLONG task_count = 0; // Counter to track the number of tasks handled by each process

    for (ULLONG i = task_id; i < total_num_points; i += num_tasks) {
        ULLONG i_random = modlin(a, i_prev, c, M);
        i_prev = i_random;
        ULLONG ix = i_random % sidelen;
        ULLONG iy = i_random / sidelen;
        x = rescale(sidelen, ix, -1, 1);
        y = rescale(sidelen, iy, -1, 1);
        if (x * x + y * y <= 1.0) points_in_circle++;
        task_count++; // Increment task counter
    }

    printf("Process %d: Completed computation. Points in circle = %llu. Tasks handled = %llu\n", task_id, points_in_circle, task_count);
    fflush(stdout);

    // End time of the parallel part
    e_parallel = MPI_Wtime();
    t_parallel = e_parallel - s_parallel;

    // Collect the results of each process into the main process
    MPI_Reduce(&points_in_circle, &total_points_in_circle, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MASTER, MPI_COMM_WORLD);

    // Collect task counts to verify total distribution
    ULLONG total_task_count = 0;
    MPI_Reduce(&task_count, &total_task_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if (task_id == MASTER) {
        end_time = MPI_Wtime();  // Record the end time of the program
        t_total = end_time - start_time;

        // Compute and print an estimate of PI
        pi_estimate = 4.0 * total_points_in_circle / total_num_points;

        // Compute speedup
        double t_speedup = t_total / (t_serial1 + t_serial2 + (t_parallel / num_tasks));

        printf("Estimated value of Pi: %f\n", pi_estimate);
        printf("Total number of points: %llu\n", total_num_points);
        printf("Total points in circle: %llu\n", total_points_in_circle);
        printf("Total tasks handled by all processes: %llu\n", total_task_count);
        printf("Total Execution time (t_total): %f seconds\n", t_total);
        printf("Create seed number time (t_serial1): %f seconds\n", t_serial1);
        printf("Master Send-Recv Communication time (t_serial2): %f seconds\n", t_serial2);
        printf("Parallel time (t_parallel): %f seconds\n", t_parallel);
        printf("Speedup Time (t_speedup): %f\n", t_speedup);

        // Verify if the total tasks handled matches the expected total_num_points
        if (total_task_count != total_num_points) {
            fprintf(stderr, "Error: Total tasks handled (%llu) does not match expected total number of points (%llu)\n", total_task_count, total_num_points);
        }
    }

    MPI_Finalize();
    return 0;
}
