#include <math.h>
#include <stdio.h>

extern "C" __global__ void
jacobikernel( float* a, float* newa, float* lchange, int n, int m, float w0, float w1, float w2 )
{
    int ti = threadIdx.x;
    int tj = threadIdx.y;
    int i = blockIdx.x * blockDim.x + ti + 1;
    int j = blockIdx.y * blockDim.y + tj + 1;

    newa[j*m+i] = w0*a[j*m+i] +
	    w1 * (a[j*m+i-1] + a[(j-1)*m+i] +
		  a[j*m+i+1] + a[(j+1)*m+i]) +
	    w2 * (a[(j-1)*m+i-1] + a[(j+1)*m+i-1] +
		  a[(j-1)*m+i+1] + a[(j+1)*m+i+1]);

    __shared__ float mychange[256];
    int ii = ti+blockDim.x*tj;
    mychange[ii] = fabsf( newa[j*m+i] - a[j*m+i] );
    __syncthreads();
    int nn = blockDim.x * blockDim.y;
    while( (nn>>=1) > 0 ){
	if( ii < nn )
	    mychange[ii] = fmaxf( mychange[ii], mychange[ii+nn] );
	__syncthreads();
    }
    if( ii == 0 )
	lchange[blockIdx.x + gridDim.x*blockIdx.y] = mychange[0];
}

extern "C" __global__ void
reductionkernel( float* lchange, int n )
{
    __shared__ float mychange[256];
    float mych = 0.0f;
    int ii = threadIdx.x, m;
    if( ii < n ) mych = lchange[ii];
    m = blockDim.x;
    while( m <= n ){
	mych = fmaxf( mych, lchange[ii+m] );
	m += blockDim.x;
    }
    mychange[ii] = mych;
    __syncthreads();
    int nn = blockDim.x;
    while( (nn>>=1) > 0 ){
	if( ii < nn )
	    mychange[ii] = fmaxf(mychange[ii],mychange[ii+nn]);
	__syncthreads();
    }
    if( ii == 0 )
	lchange[0] = mychange[0];
}

static float JacobiIter( float* a, int n, int m, float w0, float w1, float w2 )
{
    int bx, by, gx, gy;
    size_t memsize;
    float change;
    bx = 16;
    by = 16;
    gx = (n-2)/bx;
    gy = (m-2)/by;
    float *da, *dnewa, *lchange;
    memsize = sizeof(float) * n * m;
    cudaMalloc( &da, memsize );
    cudaMalloc( &dnewa, memsize );
    cudaMalloc( &lchange, gx * gy * sizeof(float) );

    dim3 block( bx, by );
    dim3 grid( gx, gy );
    cudaMemcpy( da, a, memsize, cudaMemcpyHostToDevice );
    cudaMemcpy( dnewa, a, memsize, cudaMemcpyHostToDevice );
    jacobikernel<<< grid, block >>>( da, dnewa, lchange, n, m, w0, w1, w2 );
    reductionkernel<<< 1, bx*by >>>( lchange, gx*gy );

    cudaMemcpy( a, dnewa, memsize, cudaMemcpyDeviceToHost );
    cudaMemcpy( &change, lchange, sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree( da );
    cudaFree( dnewa );
    cudaFree( lchange );
    return change;
}

extern "C"
void JacobiGPU( float* a, int n, int m, float w0, float w1, float w2, float tol )
{
    float change;
    int iters;


    iters = 0;
    do{
	++iters;
	change = JacobiIter( a, n, m, w0, w1, w2 );
    }while( change > tol );
    printf( "JacobiGPU  converged in %d iterations to residual %f\n", iters, change );
}
