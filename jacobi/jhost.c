#include <math.h>
#include <stdlib.h>
#include <stdio.h>

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
