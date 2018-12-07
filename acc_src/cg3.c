#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

// size of plate
#define MAX_ITER 10

// largest permitted change in temp (This value takes about 3400 steps)
#define MAX_TEMP_ERROR 0.01
int n;
float EPS = 0.0001;


//   helper routines
void initialize();
void track_progress(int iteration);
int main(int argc, char *argv[]) {

    int i, j;                                            // grid indexes
    int iter  = 0;
    struct timeval start_time, stop_time, elapsed_time;  // timers
    n = atoi(argv[2]);
    int max_iter  = atoi(argv[3]);
    float p[n+2][n+2];      // p grid
    float x[n+2][n+2];      // p grid
    float z[n+2][n+2];      // p grid
    float r[n+2][n+2];      // p grid
    float b[n+2][n+2];      // p grid
    //float p_temp[n+2][n+2]; // p grid from last iteration
  // Unix timer

    //initialize();                   // initialize Temp_last including boundary conditions
    for(i = 0; i <= n+1; i++){
        for (j = 0; j <= n+1; j++){
            //p_temp[i][j] = 0.0;
            p[i][j] = 0.0;
            r[i][j] = 0.0;
            b[i][j] = 0.0;
            z[i][j] = 0.0;
            x[i][j] = 0.0;
        }
    }

    FILE *myfile=fopen(argv[1], "r");
    for(i = 1; i <= n; i++)
    {
        for (j = 1 ; j <= n; j++)
        {
          fscanf(myfile,"%f",&b[i][j]);
        }
    }
    for(int i = 1;i<=n;i++){
        for(int j = 1;j<=n;j++){
            r[i][j] = b[i][j];
            p[i][j] = b[i][j];
        }
    }

    float del_new = 0,  del_old;

    for(int i = 1;i<=n;i++){
        for(int j = 1;j<=n;j++){
            del_new += b[i][j]*b[i][j];
        }
    }
    del_old = del_new;
     float vv, alpha, beta;
/*    float float1 = del_new;
    float float2 = del_old;
    float float3 = sqrt(abs(float1));
    float put;
    //iterations
   
    float leps = log10(EPS);
    float sqold =   sqrt(float2); 
    put = log10(float3/sqold ) ;
    printf("sqold %.6f %.6f\n", put, sqold);*/
    // do until error is minimal or until max steps
     int gangs = 16; int vecs = 64;
     //int gangs = atoi(argv[4]); int vecs = atoi(argv[5]);
        gettimeofday(&start_time,NULL);
    #pragma acc data copy(p, z, x, r)
    while ( iter < max_iter) {

        // main calculation: average my four neighbors
        #pragma acc kernels loop gang(gangs) vector(vecs)
        for(i = 1; i <= n; i++) {
            for(j = 1; j <= n; j++) {
                 z[i][j] = 4*p[i][j] - (p[i+1][j] + p[i-1][j] + p[i][j+1] + p[i][j-1]);
            }
        }
        #pragma acc update host(z)
        //printf("%f\n",z[4][4] );

        vv=0.0;

        #pragma acc parallel loop collapse(2) reduction(+:vv) num_gangs(gangs) vector_length(vecs)
        for(i = 1;i<=n;i++){
            for(j = 1;j<=n;j++){
                vv += z[i][j]*p[i][j];
            }
        }
        alpha = del_old/vv;
       // printf("%.6f %.6f\n", vv, alpha );
        #pragma acc parallel loop num_gangs(gangs) vector_length(vecs) collapse(2) 
        for(i = 1;i<=n;i++){
            for(j = 1;j<=n;j++){
                x[i][j] = x[i][j] + alpha*p[i][j];
            }
        }
        #pragma acc update host(x)
       // printf("%.6f\n",x[4][4] );

        #pragma acc parallel loop  num_gangs(gangs) vector_length(vecs) collapse(2)
        for(i = 1;i<=n;i++){
            for(j = 1;j<=n;j++){
                r[i][j] = r[i][j] - alpha*z[i][j];
            }
        }   
        #pragma acc update host(r)
       // printf("%.6f\n",r[4][4] );
        del_new=0.0;

        #pragma acc parallel loop  reduction(+:del_new) num_gangs(gangs) vector_length(vecs) collapse(2)
        for(i = 1;i<=n;i++){
            for(j = 1;j<=n;j++){
                del_new += r[i][j]*r[i][j];
            }
        }

        beta = del_new/del_old;
       // printf("%.6f %.6f\n",del_new, beta );

        #pragma acc parallel loop  num_gangs(gangs) vector_length(vecs) collapse(2)
        for(i = 1;i<=n;i++){
            for(j = 1;j<=n;j++){
                p[i][j] = r[i][j] + beta*p[i][j];
            }
        }
         #pragma acc update host(p)
       // printf("%.6f\n",p[4][4] );

        del_old = del_new;
        /*float1 = del_new;
        float3 = sqrt(abs(float1));
        put = log10(float3/sqold) ;*/
        // copy grid to old grid for next iteration and find latest dt
       /* #pragma acc kernels
        for(i = 1; i <= n; i++){
            for(j = 1; j <= n; j++){
    	      p[i][j] = z[i][j];
            }
        }
         */
      //  printf("%0.6f %0.6f %0.6f %0.6f\n ", put, float1, float3, sqold );
        


        iter++;
    }
//printf("\n");
//printf("iterations %d\n",iter );
    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine

    printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);

}


// initialize plate and boundary conditions
// Temp_last is used to to start first iteration
/*void initialize(){

    int i,j;

    for(i = 0; i <= n+1; i++){
        for (j = 0; j <= n+1; j++){
            p[i][j] = 1.0;
        }
    }

    // these boundary conditions never change throughout run

    // set left side to 0 and right to a linear increase
    for(i = 0; i <= n+1; i++) {
        p[i][0] = 0.0;
        p[i][n+1] = 0.0;
    }
    
    // set top to 0 and bottom to linear increase
    for(j = 0; j <= n+1; j++) {
        p[0][j] = 0.0;
        p[n+1][j] = 0.0;
    }
}
*/
// print diagonal in bottom right corner where most action is
/*void track_progress(int iteration) {

    int i,j;

    printf("---------- Iteration number: %d ------------\n", iteration);
    for(i = 1;i<=n;i++){
        for(j=1;j<=n;j++){
            printf("%f ", p[i][j] );
        }
        printf("\n");
    }
}*/
