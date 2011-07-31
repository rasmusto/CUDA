#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

extern "C" void JacobiHost( float* a, int n, int m, float w0, float w1, float w2, float tol );
extern "C" void JacobiGPU( float* a, int n, int m, float w0, float w1, float w2, float tol );

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
    printf("\nStart\n");
    int n, m;
    float *a;

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

    // init( a, n, m );

    // JacobiHost( a, n, m, .2, .1, .1, .1 );

    init( a, n, m );

    JacobiGPU( a, n, m, .2, .1, .1, .1 );
}
