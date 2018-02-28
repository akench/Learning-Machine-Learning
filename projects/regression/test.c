#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main() {

	double x = -0.000000000000001;
	double y = 0.000000001;
	
	printf("%lf", fabs(x) + fabs(y));
}
