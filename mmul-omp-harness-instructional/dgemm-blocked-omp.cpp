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

      int N = n/block_size;
      int b = block_size * block_size;
      double tempa[b];
      double tempb[b];
      double tempc[b];
      #pragma omp for
      for(int i=0;i<N;i++){
         for(int j=0;j<N;j++){
            for(int u=0;u<block_size;u++){
               for(int h=0;h<block_size;h++){
                  memcpy(&tempc[u+h*block_size],&C[i*block_size+u+(j*block_size+h)*n], sizeof(double));
               }
            }
            for(int k=0;k<N;k++){
               for(int x=0;x<block_size;x++){
                  for(int y=0;y<block_size;y++){
                     memcpy(&tempa[x+y*block_size], &A[x+i*block_size+(k*block_size+y)*n], sizeof(double));
                     memcpy(&tempb[x+y*block_size], &B[x+k*block_size+(j*block_size+y)*n], sizeof(double));
                  }
               }
               for(int l=0;l<block_size;l++){
                  for(int m=0;m<block_size;m++){
                     for(int p=0; p<block_size;p++)
                     tempc[l+m*block_size] += tempa[l+p*block_size] * tempb[p+m*block_size];
                  }
               } 
               
            }
            for(int u=0;u<block_size;u++){
               for(int h=0;h<block_size;h++){
                  memcpy(&C[i*block_size+u+(j*block_size+h)*n], &tempc[u+h*block_size], sizeof(double));
               }
            }
         }
      }
      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
   }
}