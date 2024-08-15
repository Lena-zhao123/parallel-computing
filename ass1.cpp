#if 1
#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MASTER 0
#define  M 4294967296UL // 2^32
#define MAX_LINE_LENGTH 100
#define MAX_ENTRIES 1000

typedef unsigned long ULONG;



typedef struct {
    int k;
    unsigned long k_a;
    unsigned long k_c;
    int flag;
} JumpConstant;

// Function to read file and populate the array of JumpConstant
int readFile(const char* file_path, JumpConstant jumpConstants[], int* count) {
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open the file.\n");
        return 1;
    }

    char line[MAX_LINE_LENGTH];



    // Read the file line by line
    *count = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (sscanf(line, "%d %lu %lu %d", &jumpConstants[*count].k, &jumpConstants[*count].k_a, &jumpConstants[*count].k_c, &jumpConstants[*count].flag) == 4) {
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

// Evaluate (ax+c) mod m Leapfrog calculate seed
ULONG modlin(ULONG a, ULONG x, ULONG c, unsigned long long m)
{
    ULONG i_r = (a * x + c) % m;
    return i_r;
}

// Put integer n in range x1 , x2 with the maximum integer value
double rescale(ULONG N, ULONG n, double x1, double x2)
{
    double f = static_cast<double>(n) / static_cast<double>(N);
    return x1 + f * (x2 - x1);
}


int main(int argc, char* argv[]) {
    int num_tasks, task_id, num_points, points_in_circle = 0;
    int total_points_in_circle,server=0;
    double x, y, pi_estimate;
    double start_time, end_time=0, s_t1,e_d1=0, s_t2,e_d2 = 0;
    double t_total=0, t_serial1, t_serial2, t_parallel, t_speedup;
    unsigned long* rand_nums;
    
    ULONG i_next=0,i_random = 0;
    ULONG seed = time(NULL);// Seed value
    ULONG a=0, c=0;
    const ULONG sidelen = 65536; // sqrt of m
   

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);    //id
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);  //total processes
   // printf("Number of My rank= %d and tasks= %d\n ", task_id,num_tasks);

    //argv[0] is project name.
    if (argc != 2) {
        if (task_id == MASTER) {
            printf("Usage: %s <number of points per process>\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }
    //argv[1] is the number of points.
    num_points = atoi(argv[1]);   //Enter the number of points to be run when calling the method

    if (task_id == MASTER) {


        JumpConstant jumpConstants[MAX_ENTRIES];
        int count = 0;
        // Specify the path to the file
   //     const char* file_path = "C:\\Users\\ASUS\\Documents\\S2_Courses\\159735 Studies in Parallel and Distributed Systems\\ass1\\jumpconstants.dat";
        const char* file_path = "jumpconstants.dat";

        // Call the readFile function
        if (readFile(file_path, jumpConstants, &count) != 0) {
            fprintf(stderr, "Error: Could not read the file properly.\n");
            return 1;
        }
        // Example usage: Print all the read data
        for (int i = 0; i < count; i++) {
            if (jumpConstants[i].k == num_tasks) {
                printf("k: %d, A: %lu, C: %lu, flag: %d\n", jumpConstants[i].k, jumpConstants[i].k_a, jumpConstants[i].k_c, jumpConstants[i].flag);
                a = jumpConstants[i].k_a;
                c = jumpConstants[i].k_c;
               
            }
        }


        // Master process generates random numbers for all processes
        rand_nums = (unsigned long*)malloc(num_points * num_tasks * sizeof(unsigned long));

        // Start the timer
        start_time = MPI_Wtime();

        s_t1 = MPI_Wtime();
        //Use the leapfrog computing principle to assign tasks to each server for execution
        for (ULONG n = 0; n < num_points * num_tasks; ++n) {
            
            rand_nums[n] = seed;
            i_next = modlin(a, seed, c, M);
            seed = i_next;
           
        }
        e_d1 = MPI_Wtime();
        //Create seed number time
        t_serial1 = e_d1 - s_t1;
       

        s_t2 = MPI_Wtime();
        for (int i = 1; i < num_tasks; i++) {
           
            MPI_Send(&rand_nums[i * num_points], num_points, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
            MPI_Recv(&rand_nums[i * num_points], num_points, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           
        }      
        e_d2 = MPI_Wtime();

        //Master Send-Recv Communication time
        t_serial2 = e_d2 - s_t2;


        // Master process also uses its portion of the random numbers
       
        for (ULONG i = 0; i < num_points; ++i) {
            
    
            i_random = rand_nums[i];
            ULONG ix = i_random % sidelen;
            ULONG iy = i_random / sidelen;
            // Scale current random integer to value from 0−1
            x = rescale(sidelen, ix, -1, 1);
            y = rescale(sidelen, iy, -1, 1);
            // Now we have an (x , y) pair generated from a single random integer

          //  printf("master process gets value of x = %lf ,y = %lf\n ", x, y);
            if (x * x + y * y <= 1.0) {
                points_in_circle++;
            }
        
        
        }
            
    }
    else {
        // Other processes receive their random numbers
        rand_nums = (unsigned long*)malloc(num_points * sizeof(unsigned long));
        MPI_Recv(rand_nums, num_points, MPI_UNSIGNED_LONG, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(rand_nums, num_points, MPI_UNSIGNED_LONG, MASTER, 0, MPI_COMM_WORLD);

        // Each process performs its computation
       // seed = time(NULL);
        for (ULONG i = 0; i < num_points; ++i) {
          
            // Scale the random number to a random 2−d position
            i_random = rand_nums[i];
            ULONG ix = i_random % sidelen;
            ULONG iy = i_random / sidelen;
            // Scale current random integer to value from 0−1
            x = rescale(sidelen, ix, -1, 1);
            y = rescale(sidelen, iy, -1, 1);
            // Now we have an (x , y) pair generated from a single random integer

       //     printf("slave process gets value of x = %lf ,y = %lf\n ", x, y);
            if (x * x + y * y <= 1.0) {
                points_in_circle++;
            }
        }
    }

    // Reduce the results to get the total number of points in the circle
    MPI_Reduce(&points_in_circle, &total_points_in_circle, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

   
    if (task_id == MASTER) {
        // Stop the timer
        end_time = MPI_Wtime();
       
      
        // Calculate and print the value of π
        pi_estimate = 4.0 * total_points_in_circle / (num_points * num_tasks);
       
        //Total Execution time
        t_total = end_time - start_time;


        //Parallel time
        t_parallel = t_total - t_serial1 - t_serial2 ;

        //Speedup Time
        t_speedup = t_total / ((t_serial1 + t_serial2 ) + t_parallel / num_tasks);

        printf("Estimated value of Pi: %f\n", pi_estimate);
        printf("Total number of points: %d\n", num_points * num_tasks);
        printf("Total points in circle: %d\n", total_points_in_circle);
        printf("                          \n");
        printf("Total Execution time(t_total): %f seconds\n", t_total);
        printf("Create seed number time(t_serial1): %f seconds\n", t_serial1);
        printf("Master Send-Recv Communication time(t_serial2): %f seconds\n", t_serial2);
        printf("Parallel time(t_parallel): %f seconds\n", t_parallel);
        printf("Speedup Time(S): %f seconds\n", t_speedup);
       
        
    }

    // Free allocated memory
    free(rand_nums);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}
#endif
