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
    // FILE* file;
    // fopen_s(&file, file_path, "r");
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

// 将整数 n 缩放到范围 [x1, x2]
double rescale(ULLONG N, ULLONG n, double x1, double x2) {
    double f = static_cast<double>(n) / static_cast<double>(N);
    return x1 + f * (x2 - x1);
}

int main(int argc, char* argv[]) {
    int num_tasks, task_id;
    ULLONG num_points, points_in_circle = 0;
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
        if (task_id == MASTER) printf("Usage: %s <number of points per process>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    num_points = atoll(argv[1]);  // 使用 atoll 解析为长整型

    if (task_id == MASTER) {
        start_time = MPI_Wtime();  // 记录程序开始时间

        // 生成随机种子时间
        s_t1 = MPI_Wtime();
        seed = static_cast<ULLONG>(time(NULL));
        e_t1 = MPI_Wtime();
        t_serial1 = e_t1 - s_t1;

        // 从文件中读取跳跃常数
        JumpConstant jumpConstants[MAX_ENTRIES];
        int count = 0;
        // const char* file_path = "D:\\Code\\c\\Project1\\jumpconstants.dat";
        const char* file_path = "/home/zxt/assignments/jumpconstants.dat";
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

        // Master进程发送种子和跳跃常数的通信时间
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
        // 从属进程接收主进程发送的种子、a 和 c
        MPI_Recv(&seed, 1, MPI_UNSIGNED_LONG_LONG, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&a, 1, MPI_UNSIGNED_LONG_LONG, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&c, 1, MPI_UNSIGNED_LONG_LONG, MASTER, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process %d: Received seed = %llu, a = %llu, c = %llu\n", task_id, seed, a, c);
        fflush(stdout);
    }

    // 并行部分的开始时间
    s_parallel = MPI_Wtime();
    printf("Process %d: Starting computation with %llu points.\n", task_id, num_points);
    fflush(stdout);

    // 使用种子和进程 ID 生成随机数序列并进行计算
    ULLONG i_prev = seed + task_id;
    for (ULLONG i = 0; i < num_points; ++i) {
        ULLONG i_random = modlin(a, i_prev, c, M);
        i_prev = i_random;
        ULLONG ix = i_random % sidelen;
        ULLONG iy = i_random / sidelen;
        x = rescale(sidelen, ix, -1, 1);
        y = rescale(sidelen, iy, -1, 1);
        if (x * x + y * y <= 1.0) points_in_circle++;
    }

    printf("Process %d: Completed computation. Points in circle = %llu\n", task_id, points_in_circle);
    fflush(stdout);

    // 并行部分的结束时间
    e_parallel = MPI_Wtime();
    t_parallel = e_parallel - s_parallel;

    // 归约各进程的结果到主进程
    MPI_Reduce(&points_in_circle, &total_points_in_circle, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if (task_id == MASTER) {
        end_time = MPI_Wtime();  // 记录程序结束时间
        t_total = end_time - start_time;

        // 计算并输出 π 的估计值
        pi_estimate = 4.0 * total_points_in_circle / (num_points * num_tasks);

        // 计算加速比
        double t_speedup = t_total / (t_serial1 + t_serial2 + (t_parallel / num_tasks));

        printf("Estimated value of Pi: %f\n", pi_estimate);
        printf("Total number of points: %llu\n", num_points * num_tasks);
        printf("Total points in circle: %llu\n", total_points_in_circle);
        printf("Total Execution time (t_total): %f seconds\n", t_total);
        printf("Create seed number time (t_serial1): %f seconds\n", t_serial1);
        printf("Master Send-Recv Communication time (t_serial2): %f seconds\n", t_serial2);
        printf("Parallel time (t_parallel): %f seconds\n", t_parallel);
        printf("Speedup Time (t_speedup): %f\n", t_speedup);
    }

    MPI_Finalize();
    return 0;
}
