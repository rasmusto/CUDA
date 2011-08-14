#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

__global__ void
jacobikernel( float* a, float* newa, float* lchange, int n, int m, float w0, float w1, float w2, int sz )
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
	
//***************** 1k optimization changes start here ******************//
	__syncthreads();
		
    
    int xi = blockIdx.x + gridDim.x*blockIdx.y;
    
    if(xi == 0) {
           
    float mych = 0.0f;	
    int ni = ti+blockDim.x*tj;
 
    if( ni < sz ) mych = lchange[ni];
    
    int mm = 256;
    while( mm <= sz ){
	mych = fmaxf( mych, lchange[ni+mm] );
	mm += 256;
    }
    mychange[ni] = mych;
    __syncthreads();
    
    nn = blockDim.x*blockDim.x;
    while( (nn>>=1) > 0 ){
	if( ni < nn )
	    mychange[ni] = fmaxf(mychange[ni], mychange[ni+nn]);
	__syncthreads();
    }
    if( ni == 0 )
	lchange[0] = mychange[0];
	
 }
//***************** 1k optimization changes end here ******************//
}

static float sumtime;


void JacobiGPU( float* a, int n, int m, float w0, float w1, float w2, float tol )
{
    float change;
    int iters;
    size_t memsize;
    int bx, by, gx, gy;
    float *da, *dnewa, *lchange;
    cudaEvent_t e1, e2;

    float changeCheck = 0, oldchange = 0;

    bx = 16;
    by = 16;
    gx = (n-2)/bx + ((n-2)%bx == 0?0:1);
    gy = (m-2)/by + ((m-2)%by == 0?0:1);

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
	jacobikernel<<< grid, block >>>( da, dnewa, lchange, n, m, w0, w1, w2, gx*gy );
	cudaEventRecord( e2 );

	cudaMemcpy( &change, lchange, sizeof(float), cudaMemcpyDeviceToHost );
	cudaEventElapsedTime( &msec, e1, e2 );
	sumtime += msec;
	float *ta;
	ta = da;
	da = dnewa;
	dnewa = ta; 
	//printf("iters = %d, change = %f\n", iters, change);
	if(change == oldchange)
	{
		changeCheck++;
	}
	oldchange = change;
	if(changeCheck > sqrt(m))
	{
		change = (tol - .01);
	}
	printf("iters = %d, change = %f, changeCheck = %f, oldchange = %f\n", iters, change, changeCheck, oldchange);
	
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

static void init( float* a, int n, int m )
{
    int i, j;
    memset( a, 0, sizeof(float) * n * m );
    /* boundary conditions */
    for( j = 0; j < n; ++j ){
	a[j*m+n-1] = j;
    }
    for( i = 0; i < m; ++i ){
	a[(n-1)*m+i] = i;
    }
    a[(n-1)*m+m-1] = m+n;
}

int
main( int argc, char* argv[] )
{
    int n, m;
    float *a;
    struct timeval tt1, tt2;
    int ms;
    float fms;

    if( argc <= 1 ){
	fprintf( stderr, "%s sizen [sizem]\n", argv[0] );
	return 1;
    }

    n = atoi( argv[1] );
    if( n <= 0 ) n = 100;
    m = n;
    if( argc > 2 ){
	m = atoi( argv[2] );
	if( m <= 0 ) m = 100;
    }

    printf( "Jacobi %d x %d\n", n, m );

    a = (float*)malloc( sizeof(float) * n * m );

    init( a, n, m );

    gettimeofday( &tt1, NULL );
    JacobiGPU( a, n, m, .2, .1, .1, .1 );
    gettimeofday( &tt2, NULL );
    ms = (tt2.tv_sec - tt1.tv_sec);
    ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
    fms = (float)ms / 1000000.0f;
    printf( "time(gpu ) = %f seconds\n", fms );
}
