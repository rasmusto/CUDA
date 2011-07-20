#include <math.h>
#include <stdio.h>

extern "C" __global__ void
jacobikernel( float* a, float* newa, float* lchange, int n, int m, float w0, float w1, float w2 )
{
    int ti = threadIdx.x;
    int tj = threadIdx.y;
    int i = blockIdx.x * blockDim.x + ti + 1;
    int j = blockIdx.y * blockDim.y + tj + 1;
    __shared__ float mychange[18*18];
    float mnewa, molda;


    mychange[tj*18+ti] = a[(j-1)*m+i-1];
    if( ti < 2 ) mychange[tj*18+ti+16] = a[(j-1)*m+i+15];
    if( tj < 2 ) mychange[(tj+16)*18+ti] = a[(j+15)*m+i-1];
    if( tj < 2 && ti < 2 ) mychange[(tj+16)*18+ti+16] = a[(j+15)*m+i+15];

    __syncthreads();

    molda = mychange[(tj+1)*18+(ti+1)];
    mnewa = w0*molda +
	    w1 * (mychange[(tj+1)*18+(ti  )] + mychange[(tj  )*18+(ti+1)] +
		  mychange[(tj+1)*18+(ti+2)] + mychange[(tj+2)*18+(ti+1)]) +
	    w2 * (mychange[(tj  )*18+(ti  )] + mychange[(tj+2)*18+(ti  )] +
		  mychange[(tj  )*18+(ti+2)] + mychange[(tj+2)*18+(ti+2)]);
    newa[j*m+i] = mnewa;
    __syncthreads();

    int ii = ti+blockDim.x*tj;
    mychange[ii] = fabsf( mnewa - molda );
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

static float sumtime;

extern "C"
void JacobiGPU( float* a, int n, int m, float w0, float w1, float w2, float tol )
{
    float change;
    int iters;
    size_t memsize;
    int bx, by, gx, gy;
    float *da, *dnewa, *lchange;
    cudaEvent_t e1, e2;

    bx = 16;
    by = 16;
    gx = (n-2)/bx;
    gy = (m-2)/by;

    sumtime = 0.0f;
    memsize = sizeof(float) * n * m;
    cudaMalloc( &da, memsize );
    cudaMalloc( &dnewa, memsize );
    cudaMalloc( &lchange, gx * gy * sizeof(float) );
    cudaEventCreate( &e1 );
    cudaEventCreate( &e2 );

    dim3 block( bx, by );
    dim3 grid( gx, gy );

    iters = 0;
    cudaMemcpy( da, a, memsize, cudaMemcpyHostToDevice );
    cudaMemcpy( dnewa, a, memsize, cudaMemcpyHostToDevice );
    do{
	float msec;
	++iters;

	cudaEventRecord( e1 );
	jacobikernel<<< grid, block >>>( da, dnewa, lchange, n, m, w0, w1, w2 );
	reductionkernel<<< 1, bx*by >>>( lchange, gx*gy );
	cudaEventRecord( e2 );

	cudaMemcpy( &change, lchange, sizeof(float), cudaMemcpyDeviceToHost );
	cudaEventElapsedTime( &msec, e1, e2 );
	sumtime += msec;
	float *ta;
	ta = da;
	da = dnewa;
	dnewa = ta; 
    }while( change > tol );
    printf( "JacobiGPU  converged in %d iterations to residual %f\n", iters, change );
    printf( "JacobiGPU  used %f seconds total\n", sumtime/1000.0f );
    cudaMemcpy( a, dnewa, memsize, cudaMemcpyDeviceToHost );
    cudaFree( da );
    cudaFree( dnewa );
    cudaFree( lchange );
    cudaEventDestroy( e1 );
    cudaEventDestroy( e2 );
}
