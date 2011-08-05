// JacobiRel.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

__global__ void kernelOne(float* a, float* newa, float* lchange, int n, int m,
  float w0, float w1, float w2 )
{
     int ti = threadIdx.x, tj = threadIdx.y; /* local indices */
     int i = blockIdx.x*16+ti+1, j = blockIdx.y*16+tj+1; /* global */
     newa[j*m+i] = w0 * a[j*m+i] + 
                 w1 * (a[j*m+i-1] + a[(j-1)*m+i] + a[j*m+i+1] + a[(j+1)*m+i]) +
                 w2 * (a[(j-1)*m+i-1] + a[(j+1)*m+i-1] + a[(j-1)*m+i+1] + a[(j+1)*m+i+1]);

    __shared__ float mychange[16*16];
    /* store this thread's "change" */
    mychange[ti+16*tj] = fabsf(newa[j*m-i]-a[j*m+i]);
   __syncthreads();
   /* reduce all "change" values for this thread block
   * to a single value */
   int nn = 256;
   while( (nn >>= 1) > 0 ){
          if( ti+tj*16 < nn )
          mychange[ti+tj*16] = fmaxf( mychange[ti+tj*16],
          mychange[ti+tj*16+nn]);
          __syncthreads();
          }
   /* store this thread block's "change" */
   if( ti==0 && tj==0 )
       lchange[blockIdx.x+gridDim.x*blockIdx.y] = mychange[0];
}
  
__global__ void kernelTwo( float* lchange, int n )
{
   __shared__ float mychange[256];
   float mych;
   int i = threadIdx.x, m;
   mych = lchange[i];
   m = 256;
   while( m <= n ){
          mych = fmaxf(mych,lchange[i+m]);
          m += 256;
   }
   mychange[i] = mych;
   __syncthreads();
   n = 256;
   while( (n >>= 1) > 0 ){
          if(i<n) mychange[i] = fmaxf(mychange[i],mychange[i+n]);
          __syncthreads();
   }
   if(i==0) lchange[0] = mychange[0];
}

int main(void)
{
  float change;
  float *newa, *a;
  float *da, *dnewa, *lchange;
  int i, j, m, n;

  printf("Please give m and n: ");
  scanf("%d %d",&m,&n);
  
  size_t memsize = sizeof(float)*n*m;
  
  a = (float *)malloc(memsize);
  newa = (float *)malloc(memsize);
 
  cudaMalloc( &da, memsize );
  cudaMalloc( &dnewa, memsize );
  cudaMalloc( &lchange, ((n-2)/16)*((m-2)/16)*sizeof(float) );
  
  for (i=0; i<m; i++){
     for (j=0; j<n; j++){
         a[i*n+j] = i;
       }
     }
  
  cudaMemcpy( da, a, memsize, cudaMemcpyHostToDevice );
  
  dim3 threads( 16, 16 );
  dim3 blocks( (n-2)/16, (m-2)/16 );
  kernelOne<<<blocks,threads>>>( da, dnewa, lchange, n, m, 1.0, 1.0, 1.0 );
  kernelTwo<<<1,256>>>( lchange, ((n-2)/16)*((m-2)/16) );
  
  cudaMemcpy( newa, dnewa, memsize, cudaMemcpyDeviceToHost );
  cudaMemcpy( &change, lchange, 4, cudaMemcpyDeviceToHost );
  
  // check results
  printf("Change: %f\n", change);
  for (i=0; i<m; i++)
     for (j=0; j<n; j++)
       printf("%f\n", newa[i*n+j]);
  
  cudaFree( da );
  cudaFree( dnewa );
  cudaFree( lchange );
}
