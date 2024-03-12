#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include "array_types.hpp"
#include <vector>

#include "mpi.h"

using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

void fill_random(vec<double> x, double xmin, double xmax, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(xmin, xmax);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

void fill_random(matrix<double> x, double dispersion, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0, dispersion);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

// matrix storage: ma_i rows of a and mb_i rows of b on i-th process
// result storage: ma_i rows of c on i-th process
matrix<double> matmul_ikj(matrix<double> a, matrix<double> b, std::vector<int>& sendcounts, std::vector<int>& displs, int n, MPI_Comm comm) 
{
    int rowa = a.nrows();
    int rowb = b.nrows();
    int cola = a.ncols();
    int colb = b.ncols();
    
    ptrdiff_t i, j, k;

    int myrank, comm_size;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &myrank);

    matrix<double> c(rowa, colb);
    for (i=0; i < c.length(); i++)
    {
        c(i) = 0;
    }

    double last = 0;
    
    
    for (int rank_b = 0; rank_b < comm_size; rank_b++)
    {
        matrix<double> btemp(sendcounts[rank_b]/colb, colb);
        matrix<double> bmatr = myrank == rank_b ? b : btemp;
        MPI_Bcast(bmatr.raw_ptr(), sendcounts[rank_b], MPI_DOUBLE, rank_b, comm);

        for (i=0; i<rowa; i++)
        {
            for (k=0; k<sendcounts[rank_b]/ colb; k++)
            {
                ptrdiff_t k_global = k + displs[rank_b]/colb;//rank_b * rowb;
                
                double a_ik = a(i, k_global);
                for (j=0; j<colb; j++)
                {
                    c(i, j) += a_ik * bmatr(k,j);
                }
            }
        }
    }
    return c;
}

// read an integer number from stdin into `n`
void read_integer(int* n, int rank, MPI_Comm comm)
{
    if (rank==0)
    {
        std::cin >> *n;
    }

    MPI_Bcast(n, 1, MPI_INT, 0, comm);
}

void scatter_matrix(matrix<double>& source, matrix<double>& dest, std::vector<int>& sendcounts, std::vector<int>& displs, int root, MPI_Comm comm) {
    int myrank, comm_size;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &comm_size);

    // int n = source.ncols();
    // std::vector<int> sendcounts(comm_size), displs(comm_size);

    // int rows_per_process = total_rows / comm_size;
    // int remaining_rows = total_rows % comm_size;


    // int offset = 0;
    // for (int i = 0; i < comm_size; ++i) {
    //     sendcounts[i] = (rows_per_process + (i < remaining_rows ? 1 : 0)) * n;
    //     displs[i] = offset;
    //     offset += sendcounts[i];
        
    //     std::cout << "Process " << i << " - sendcounts: " << sendcounts[i] << ", displs: " << displs[i] << std::endl;
    // }

    MPI_Scatterv(source.raw_ptr(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 dest.raw_ptr(), sendcounts[myrank], MPI_DOUBLE, root, comm);


}


int main(int argc, char* argv[])
{
    int n;

    int myrank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    read_integer(&n, myrank, MPI_COMM_WORLD);
    
    // assuming n is multiple of world_size
    int rows_per_process = n / world_size;
    int remaining_rows = n % world_size;
    int local_rows = rows_per_process + (myrank < remaining_rows ? 1 : 0);
    matrix<double> a(local_rows, n), b(local_rows, n);

    std::vector<int> sendcounts(world_size), displs(world_size);

    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
        sendcounts[i] = (rows_per_process + (i < remaining_rows ? 1 : 0)) * n;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // generate matrix on rank 0 (for simplicity)
    if (myrank == 0)
    {
        matrix<double> a_all(n, n);
        fill_random(a_all, 1.0, 9876);
        scatter_matrix(a_all, a, sendcounts, displs, 0, MPI_COMM_WORLD);
        
        fill_random(a_all, 1.0, 9877);
        scatter_matrix(a_all, b, sendcounts, displs, 0, MPI_COMM_WORLD);
        // std::cout << myrank << "@\n";
    }
    else
    {
        scatter_matrix(a, a, sendcounts, displs, 0, MPI_COMM_WORLD);
        scatter_matrix(b, b,  sendcounts, displs, 0, MPI_COMM_WORLD);
        
    }

    double t0 = MPI_Wtime();

    matrix<double> c = matmul_ikj(a, b,sendcounts, displs, n, MPI_COMM_WORLD); //matrix<double>

    double t1 = MPI_Wtime();

    if (myrank == world_size-1)
    {
        std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                << "world_size: " << world_size << "\n"
                << "N: " << n << "\n"
                << "Timing: " << t1 - t0 << " sec\n"
                << "Answer[n, n] = " << c(c.length()-1)
                << std::endl;
    }

    MPI_Finalize();
    return 0;
}
