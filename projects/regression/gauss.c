#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

double gauss(void)
{

//  srand( time(NULL) );
  double x = (double)rand() / RAND_MAX,
         y = (double)rand() / RAND_MAX,
         z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  return z;
}


int main() {
	srand( time(NULL));
	double val = gauss();
	printf("%lf\n", val);
	double r = (double)rand();
	printf("%lf\n", r);
}
