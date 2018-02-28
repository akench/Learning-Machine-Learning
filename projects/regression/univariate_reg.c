#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct data_struct { 

	int num_samples;
	long double *x_data;
	long double *y_data;

} DataObj ;

void shuffle(long double *arr1, long double *arr2, size_t n) {
	if (n > 1) {
  	        size_t i;
			for (i = 0; i < n - 1; i++) {
				size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
				long double t = arr1[j];
				arr1[j] = arr1[i];
				arr1[i] = t;
		
	  			t = arr2[j];
				arr2[j] = arr2[i];
				arr2[i] = t;
        }
    }
}

DataObj *create_fake_data(int num_samples, long double bias, long double slope, long double noise) {
	
	long double max_x = 50.0;

	long double *x_data = (long double *)malloc(num_samples * sizeof(long double));
	long double *y_data = (long double *)malloc(num_samples * sizeof(long double));	
	
	int i;
	for(i = 0; i < num_samples; i++) {
		
		long double curr_noise = ((long double)rand()/(long double)(RAND_MAX)) * 2 * noise  -  noise;
	
		long double x = ((long double)rand()/(long double)(RAND_MAX)) * max_x;
		long double y = bias + slope*x + curr_noise;

		x_data[i] = x;
		y_data[i] = y; 
	}

	shuffle(x_data, y_data, num_samples);	

	DataObj *data_ptr = malloc(sizeof(DataObj));
	data_ptr->num_samples = num_samples;
	data_ptr->x_data = x_data;
	data_ptr->y_data = y_data;
	
	return data_ptr;
}



long double update_vars(long double *theta_0, long double *theta_1, DataObj *data_ptr,  long double learning_rate) {
	
	long double deriv_0 = 0.0, deriv_1 = 0.0;

	int i;
	for(i = 0; i < data_ptr->num_samples; i++) {

		long double x = data_ptr->x_data[i];
		long double y = data_ptr->y_data[i];

		deriv_0 += ((*theta_0 + *theta_1 * x) - y);
		deriv_1 += ((*theta_0 + *theta_1 * x) - y) * x;

	}
	
	deriv_0 /= data_ptr->num_samples;
	deriv_1 /= data_ptr->num_samples;
	
	*theta_0 = *theta_0 - learning_rate * deriv_0;
	*theta_1 = *theta_1 - learning_rate * deriv_1;

	return fabs(deriv_0) + fabs(deriv_1);
	
}


void train(DataObj *data_ptr, double learning_rate) {
	
	long double theta_0 = ((long double)rand()/(long double)(RAND_MAX)) * 2 - 1;
	long double theta_1 = ((long double)rand()/(long double)(RAND_MAX)) * 2 - 1;
	
	int it = 0;
	long double sum_of_derivs = 1000000.0;

	while( sum_of_derivs > 0.000001  ) {
	
		sum_of_derivs = update_vars(&theta_0, &theta_1, data_ptr,  learning_rate);
		it++;
	}	


	printf("Line of best fit:   %LF*X + %LF\n", theta_1, theta_0);
	printf("Took %d iterations to complete\n", it);
 	
}


int main(int argc, char *argv) {

	int num_samples;
	long double bias, slope, noise;
	printf("Enter number of samples\n");
	scanf("%d", &num_samples);

	printf("Enter bias (long double)\n");
	scanf("%LF", &bias);
			
	printf("Enter slope (long double)\n");
	scanf("%LF", &slope);		
			
	printf("Enter noise (long double)\n");
	scanf("%LF", &noise);		
			
	printf("samples %d, bias %LF, slope %LF, noise %LF\n", num_samples, bias, slope, noise);

	
	DataObj *data_ptr = create_fake_data(num_samples, bias, slope, noise);

	clock_t begin = clock();
	train(data_ptr, 0.001);
	clock_t end = clock();
	
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Took %lf seconds\n", time_spent); 
	return 0;
}
