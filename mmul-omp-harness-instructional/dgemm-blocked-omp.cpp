#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"
#include <cstring>
const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   std::cout << "Insert your blocked matrix multiply with copy optimization, openmp-parallel edition here " << std::endl;
   #pragma omp parallel 
   {
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
      int numBlocks = n / block_size;
      int blockSizeSq = block_size * block_size;
      #pragma omp parallel for
      for (int blockRow = 0; blockRow < numBlocks; blockRow++) {
         for (int blockCol = 0; blockCol < numBlocks; blockCol++) {
            for (int i = 0; i < block_size; i++) {
               for (int j = 0; j < block_size; j++) {
                  double blockResult = 0.0;
                  for (int blockK = 0; blockK < numBlocks; blockK++) {
                     double blockA = 0.0;
                     double blockB = 0.0;
                     for (int x = 0; x < block_size; x++) {
                        blockA += A[x + blockRow * block_size + (blockK * block_size + j) * n];
                        blockB += B[x + blockK * block_size + (blockCol * block_size + j) * n];
                     }
                     blockResult += blockA * blockB;
                  }
                C[blockRow * block_size + i + (blockCol * block_size + j) * n] = blockResult;
               }
            }
         }
      }  

      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
   }
}
