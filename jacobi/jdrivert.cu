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

int main( int argc, char* argv[] )
{
    printf("\nJacobi Driver Initiated\n");			    
    
    int n, m;							// The number of rows and columns in the matrix
    float *a;							// The answer vector
    struct timeval tt1, tt2;					// time structures for evaluating the time of day
    int ms;							// miliseconds
    float fms;							// final miliseconds

    if( argc <= 1 ){
	fprintf( stderr, "Error Number of Arguments <=1, Need Col x Row Data\n", argv[0] );
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

 init( a, n, m );
/* 
<<<<<<< HEAD
    //
=======
>>>>>>> 5f41663343c459d76df14c9dab9add8c93673cba
    init( a, n, m );	
    
    //If the number of rows and columns are both under 10 then print to the screen
    if( n < 10 && m < 10)
    {
    	for(int i = 0; i< m; ++i)
 	{
		for(int t=0; t<n; ++t)
		{
			printf("%f ", a[i*m + t]);
		}
		printf("\n");
	}    
    }
<<<<<<< HEAD
=======
>>>>>>> 4ad927989f9a2fcaba65d52b10d2ccb6b885f643
*/ 


    gettimeofday( &tt1, NULL );
    JacobiHost( a, n, m, .2, .1, .1, .1 );
    gettimeofday( &tt2, NULL );
    ms = (tt2.tv_sec - tt1.tv_sec);
    ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
    fms = (float)ms / 1000000.0f;
    printf( "time(host) = %f seconds\n", fms );

    init( a, n, m );

    gettimeofday( &tt1, NULL );
    JacobiGPU( a, n, m, .2, .1, .1, .1 );
    gettimeofday( &tt2, NULL );
    ms = (tt2.tv_sec - tt1.tv_sec);
    ms = ms * 1000000 + (tt2.tv_usec - tt1.tv_usec);
    fms = (float)ms / 1000000.0f;
    printf( "time(gpu ) = %f seconds\n", fms );
}
