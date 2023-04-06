#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <vector>

int main(int argc, char *argv[])
{
    // Initialize the MPI communicator
    MPI_Init(&argc, &argv);

    // Get rank/size
    int myRank, numRanks;

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int nGlobal = atoi(argv[1]);

    // Setup local array
    int nLocal = (int)nGlobal / numRanks;
    if (myRank == 0)
        nLocal += nGlobal % numRanks;

    // Allocate and initialize the vector
    // x is allocate with 2 ghost zones
    std::vector<float> x(nLocal + 2);
    std::vector<float> y(nLocal);
    for (int i = 0; i < nLocal + 2; i++)
    {
        x[i] = i + myRank * nLocal + std::min(myRank, 1) * nGlobal % numRanks;
        // std::cout << "o " << myRank << " " << i << " " << x[i] << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Now specify neighbors
    int leftNeigh = myRank - 1;
    if (myRank == 0)
        leftNeigh = numRanks - 1;
    int rghtNeigh = myRank + 1;
    if (myRank == numRanks - 1)
        rghtNeigh = 0;

    // Send and Recv tags need to be paired correctly!
    int sendTag = 0;
    int recvTag = 1;
    if (myRank == 0)
    {
        sendTag = 0;
        recvTag = numRanks - 1;
    }
    else if (myRank == numRanks - 1)
    {
        sendTag = numRanks - 1;
        recvTag = 0;
    }
    else if (myRank % 2 == 0)
    {
        sendTag = numRanks - 1;
        recvTag = 0;
    }
    else if (myRank % 2 == 1)
    {
        sendTag = 0;
        recvTag = numRanks - 1;
    }

    // std::cout << "rank " << myRank << " left: " << leftNeigh << " rght: " << rghtNeigh << " send: " << sendTag << " recv " << recvTag << std::endl; 

    if (numRanks == 1)
    {
        x[0] = x[nLocal];
        x[nLocal + 1] = x[1];
    }
    else
    {
        // Left exchange
        MPI_Sendrecv(&x[1], 1, MPI_FLOAT, leftNeigh, leftNeigh, &x[nLocal + 1], 1, MPI_FLOAT, rghtNeigh, myRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // std::cout << "rank " << myRank << " done" << std::endl;

        // Right exchange
        MPI_Sendrecv(&x[nLocal], 1, MPI_FLOAT, rghtNeigh, rghtNeigh, &x[0], 1, MPI_FLOAT, leftNeigh, myRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // for (int i = 0; i < nLocal+2; i++)
    // {
    //     std::cout << "x " << myRank << " " << i << " " << x[i] << std::endl;
    // }
    
    MPI_Barrier(MPI_COMM_WORLD);
    // perform the averaging
    for (int i = 0; i < nLocal; i++)
    {
        y[i] = (x[i] + x[i + 1] + x[i + 2]) / 3.;
        std::cout << myRank << " " << i << " " << y[i] << std::endl;
    }

    MPI_Finalize();
}