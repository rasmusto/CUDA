#include <math.h>
#include <stdio.h>

extern "C" __global__ void
jacobikernel( float* a, float* newa, float* lchange, int n, int m, float w0, float w1, float w2 )
{
    int ii = 0;						//Unknown Variable
    int nn = 0;						//Unknown Variable 
    int ti = threadIdx.x;				// The ID of the thread in the x dimension within a block
    int tj = threadIdx.y;				// The ID of the thread in the y dimension within a block
    int i = blockIdx.x * blockDim.x + ti + 1;		// The Address of a thread in the x dimension if all blocks  are laid out linearly
    int j = blockIdx.y * blockDim.y + tj + 1;		// The Address of a thread in the y dimension if all blocks are laid out linearly
    __shared__ float mychange[18*18];			// Shared memory for the block
    float mnewa, molda;					// new value for a, old value for a


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

    ii = ti+blockDim.x*tj;
    mychange[ii] = fabsf( mnewa - molda );
    __syncthreads();
    nn = blockDim.x * blockDim.y;
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
    printf("\nJacobi GPU calculating\n");
    printf("\nYou entered the following parameters:\nNumber of Rows   	:  %i\nNumber of Colomuns	:  %i\n\n", n, m);
    float change;						// The amount of total change from one iteration to the next
    int iters;							// The number of iterations
    size_t memsize;						
    int bx, by, gx, gy;						// Block and grid dimensions (x2)
    float *da, *dnewa, *lchange;				// Device Matrix, New Device Matrix, Amount the matrix changes within a grid
    cudaEvent_t e1, e2;						// Cuda Events
    float msec;							// Amount of time between Cuda Event 1 and Cuda Event 2					
    float *ta;							// Temporary Pointer For Swithcing Memory Between da And dnewa

    bx = 16;							// Block Deminsion in x direction
    by = 16;							// Block Dimension in y direction
    gx = (n-2)/bx;						// Grid Dimension in x direction
    gy = (m-2)/by;						// Grid Dimension in y direction

    sumtime = 0.0f;						// Time Taken To Compute Accross All Iterations
    memsize = sizeof(float) * n * m;				// Amount of memory needed for device side array
   
    cudaMalloc( &da, memsize );					// Allocate Memory for array on device side
    cudaMalloc( &dnewa, memsize );				// Allocate Memory for new array on device side
    cudaMalloc( &lchange, gx * gy * sizeof(float) );		// Allocate Memory for Change array for device. (Change across a grid)
    cudaEventCreate( &e1 );					
    cudaEventCreate( &e2 );

    dim3 block( bx, by );					// Cuda Dimension Variable for blocks
    dim3 grid( gx, gy );					// Cuda Dimension Variable for grids

    iters = 0;							// Set iterations to zero
    cudaMemcpy( da, a, memsize, cudaMemcpyHostToDevice );	// Copy Values from Array given (a) to Device Array (da)
    cudaMemcpy( dnewa, a, memsize, cudaMemcpyHostToDevice );	// Copy Values from Array given (a) to Device Array (dnewa)
    do{							
        msec;									
	++iters;	

	cudaEventRecord( e1 );									// Record Event One	
	
        // Call Kernel <<<X by Y Grids, X by Y Blocks>>>(Device Array, Device NewArray, Change Array, Columns?, Rows?, )
	jacobikernel<<< grid, block >>>( da, dnewa, lchange, n, m, w0, w1, w2 );	
	reductionkernel<<< 1, bx*by >>>( lchange, gx*gy );

	cudaEventRecord( e2 );									// Record Event Two

	cudaMemcpy( &change, lchange, sizeof(float), cudaMemcpyDeviceToHost );			// Copy the Change Matrix
	cudaEventElapsedTime( &msec, e1, e2 );							// Get The Elapsed Time Between Cuda Event One and Cuda Event Two

	sumtime += msec;									// Add the Time Taken for the Last Iteration To the Total Time Taken										
	ta = da;										// Point da at dnewa memory and point dnewa at da memory
	da = dnewa;
	dnewa = ta; 
    }while( change > tol );

    printf( "JacobiGPU  converged in %d iterations to residual %f\n", iters, change );		// Print Out Results
    printf( "JacobiGPU  used %f seconds total\n", sumtime/1000.0f );
    cudaMemcpy( a, dnewa, memsize, cudaMemcpyDeviceToHost );					// Copy Memory from Device Back To Host
    cudaFree( da );										
    cudaFree( dnewa );										// Free Cuda Memory
    cudaFree( lchange );
    cudaEventDestroy( e1 );
    cudaEventDestroy( e2 );
}
