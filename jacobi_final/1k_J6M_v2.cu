/*
*   1k_J6M_v2.cu 
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

    __global__ void
jacobikernel( float* a, float* newa, float* lchange, int n, int m, float w0, float w1, float w2, int sz, int THR )
{
    int ti = threadIdx.x;
    int tj = threadIdx.y;
    int i = blockIdx.x * blockDim.x + ti + 1;
    int j = blockIdx.y * blockDim.y + tj + 1;
    int TPT = THR+2;
    int TMO = THR-1;
    __shared__ float mychange[23*23];
    float mnewa, molda;


    mychange[tj*TPT+ti] = a[(j-1)*m+i-1];
    if( ti < 2 ) mychange[tj*TPT+ti+THR] = a[(j-1)*m+i+TMO];
    if( tj < 2 ) mychange[(tj+THR)*TPT+ti] = a[(j+TMO)*m+i-1];
    if( tj < 2 && ti < 2 ) mychange[(tj+THR)*TPT+ti+THR] = a[(j+TMO)*m+i+TMO];

    __syncthreads();

    molda = mychange[(tj+1)*TPT+(ti+1)];
    mnewa = w0*molda +
        w1 * (mychange[(tj+1)*TPT+(ti  )] + mychange[(tj  )*TPT+(ti+1)] +
                mychange[(tj+1)*TPT+(ti+2)] + mychange[(tj+2)*TPT+(ti+1)]) +
        w2 * (mychange[(tj  )*TPT+(ti  )] + mychange[(tj+2)*TPT+(ti  )] +
                mychange[(tj  )*TPT+(ti+2)] + mychange[(tj+2)*TPT+(ti+2)]);
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

    //***************** 1k optimization changes start here ******************//	
    __syncthreads();		

    int xi = blockIdx.x + gridDim.x*blockIdx.y;

    if(xi == 0) {

        float mych = 0.0f;	
        int ni = ti+blockDim.x*tj;

        if( ni < sz ) mych = lchange[ni];

        int mm = THR*THR;
        while( mm <= sz ){
            if(ni+mm < sz)
                mych = fmaxf( mych, lchange[ni+mm] );
            mm += THR*THR;
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


void JacobiGPU( float* a, int n, int m, int numThr,  float w0, float w1, float w2, float tol )
{
    float change;
    int iters;
    size_t memsize;
    int bx, by, gx, gy;
    float *da, *dnewa, *lchange;
    cudaEvent_t e1, e2;

    bx = numThr;
    by = numThr;
    gx = (n-2)/bx + ((n-2)%bx == 0?0:1);
    gy = (m-2)/by + ((m-2)%by == 0?0:1);

    printf("Number of threads = %i and %i.\nNumber of Grids = %i and %i\n", bx, by, gx, gy);

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
        jacobikernel<<< grid, block >>>( da, dnewa, lchange, n, m, w0, w1, w2, gx*gy, numThr );
        cudaEventRecord( e2 );

        cudaMemcpy( &change, lchange, sizeof(float), cudaMemcpyDeviceToHost );
        cudaEventElapsedTime( &msec, e1, e2 );
        sumtime += msec;
        float *ta;
        ta = da;
        da = dnewa;
        dnewa = ta; 
    }while( change > tol );

    double time = sumtime/1000.0f;
    double dNumOps = 14.0 * iters * n *m;
    double gflops = dNumOps/time/1e9;


    printf( "JacobiGPU  converged in %d iterations to residual %f\n", iters, change );
    printf( "JacobiGPU  used %.5f seconds total\n", sumtime/1000.0f );
    printf( "Size(Number of Operations) = %.0f Ops/sec \n", dNumOps );
    printf( "Throughtput = %.4f GFlops/sec \n",gflops );
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

int main( int argc, char* argv[] )
{
    int n, m;
    float *a;
    struct timeval tt1, tt2;
    int ms;
    float fms;
    int numThr;

    if( argc <= 1 ){
        fprintf( stderr, "%s sizen [sizem]\n", argv[0] );
        return 1;
    }

    n = atoi( argv[1] );
    if( n <= 0 ) n = 100;
    m = n;
    if( argc > 3 ){
        m = atoi( argv[2] );
        numThr = atoi (argv[3] );
        if( m <= 0 ) m = 100;
    }

    printf( "Jacobi %d x %d\n", n, m );

    a = (float*)malloc( sizeof(float) * n * m );

    init( a, n, m );

    gettimeofday( &tt1, NULL );
    JacobiGPU( a, n, m, numThr, .2, .1, .1, .1 );
    gettimeofday( &tt2, NULL );
    ms = (tt2.tv_sec - tt1.tv_sec);
    ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
    fms = (float)ms / 1000000.0f;
    printf( "time(gpu ) = %f seconds\n", fms );
}
