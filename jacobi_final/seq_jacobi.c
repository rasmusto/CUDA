#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

static float JacobiIter( float* a, float* newa, int n, int m, float w0, float w1, float w2 )
{
    int i, j;
    float change;

    change = 0.0f;
    for( j = 1; j < n-1; ++j ){
	for( i = 1; i < m-1; ++i ){
	    newa[j*m+i] = w0*a[j*m+i] +
		    w1 * (a[j*m+i-1] + a[(j-1)*m+i] +
			  a[j*m+i+1] + a[(j+1)*m+i]) +
		    w2 * (a[(j-1)*m+i-1] + a[(j+1)*m+i-1] +
			  a[(j-1)*m+i+1] + a[(j+1)*m+i+1]);
	    change = fmaxf( change, fabsf( newa[j*m+i] - a[j*m+i] ));
	}
    }
    return change;
}

void JacobiHost( float* a, int n, int m, float w0, float w1, float w2, float tol )
{
    int i, j;
    float change;
    float *ta;
    float *newa;
    int iters;

    newa = (float*)malloc( sizeof(float) * n * m );
    /* copy boundary conditions */
    for( j = 0; j < n; ++j ){
	newa[j*m+0] = a[j*m+0];
	newa[j*m+m-1] = a[j*m+m-1];
    }
    for( i = 0; i < m; ++i ){
	newa[0*m+i] = a[0*m+i];
	newa[(n-1)*m+i] = a[(n-1)*m+i];
    }
    iters = 0;
    do{
	++iters;
	change = JacobiIter( a, newa, n, m, w0, w1, w2 );
	/* swap pointers */
	ta = a;
	a = newa;
	newa = ta;
    }while( change > tol );
    printf( "JacobiHost converged in %d iterations to residual %f\n", iters, change );
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
    JacobiHost( a, n, m, .2, .1, .1, .1 );
    gettimeofday( &tt2, NULL );
    ms = (tt2.tv_sec - tt1.tv_sec);
    ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
    fms = (float)ms / 1000000.0f;
    printf( "time(host) = %f seconds\n", fms );
}
