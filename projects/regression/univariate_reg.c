#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct data_struct { 

	int num_samples;
	double bias;
	double slope;
	double noise;
	double *x_data;
	double *y_data;

} DataObj ;

void shuffle(double *arr1, double *arr2, size_t n) {
	if (n > 1) {
  	        size_t i;
			for (i = 0; i < n - 1; i++) {
				size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
				double t = arr1[j];
				arr1[j] = arr1[i];
				arr1[i] = t;
		
	  			t = arr2[j];
				arr2[j] = arr2[i];
				arr2[i] = t;
        }
    }
}

DataObj *create_fake_data(int num_samples, double bias, double slope, double noise) {
	
	double max_x = 50.0;

	double *x_data = (double *)malloc(num_samples * sizeof(double));
	double *y_data = (double *)malloc(num_samples * sizeof(double));	
	
	int i;
	for(i = 0; i < num_samples; i++) {
		double x = ((double)rand()/(double)(RAND_MAX)) * max_x;
		double y = bias + slope*x;

		x_data[i] = x;
		y_data[i] = y; 
	}

	shuffle(x_data, y_data, num_samples);	

	DataObj *data_ptr = malloc(sizeof(DataObj));
	data_ptr->num_samples = num_samples;
	data_ptr->bias = bias;
	data_ptr->slope = slope;
	data_ptr->noise = noise;
	data_ptr->x_data = x_data;
	data_ptr->y_data = y_data;
	
	return data_ptr;
}

double abs_val(double x) {
	if(x < 0) {
		return -x;
	}
	else {
		return x;
	}
}


void update_vars(double *theta_0, double *theta_1, DataObj *data_ptr,  double learning_rate) {
	
	double deriv_0 = 0.0, deriv_1 = 0.0;

	int i;
	for(i = 0; i < data_ptr->num_samples; i++) {

		double x = data_ptr->x_data[i];
		double y = data_ptr->y_data[i];

		deriv_0 += ((*theta_0 + *theta_1 * x) - y);
		deriv_1 += ((*theta_0 + *theta_1 * x) - y) * x;

	}
	
	deriv_0 /= data_ptr->num_samples;
	deriv_1 /= data_ptr->num_samples;
	
	*theta_0 = *theta_0 - learning_rate * deriv_0;
	*theta_1 = *theta_1 - learning_rate * deriv_1;

	printf("deriv0=%lf   and deriv1=%lf\n", deriv_0, deriv_1);

	
}


void train(DataObj *data_ptr, double learning_rate) {
	
	double theta_0 = ((double)rand()/(double)(RAND_MAX)) * 2 - 1;
	double theta_1 = ((double)rand()/(double)(RAND_MAX)) * 2 - 1;
	
	int it = 0;
	double deriv_0 = 10000.0;
	double deriv_1 = 10000.0;

	while( abs_val(deriv_0) + abs_val(deriv_1) > 0.001) {
		
		printf("sum is %lf\n", abs_val(deriv_0) + abs_val(deriv_1));
		
		printf("%lf*X + %lf\n", theta_1, theta_0);
		update_vars(&theta_0, &theta_1, data_ptr,  learning_rate);
		it++;
	}	


	printf("Line of best fit:   %lf*X + %lf\n", theta_1, theta_0);
	printf("Took %d iterations to complete", it);
 	
}


int main(int argc, char *argv) {

	int num_samples;
	double bias, slope, noise;
	printf("Enter number of samples\n");
	scanf("%d", &num_samples);

	printf("Enter bias (double)\n");
	scanf("%lf", &bias);
			
	printf("Enter slope (double)\n");
	scanf("%lf", &slope);		
			
	printf("Enter noise (double)\n");
	scanf("%lf", &noise);		
			
	printf("samples %d, bias %lf, slope %lf, noise %lf\n", num_samples, bias, slope, noise);
	
	DataObj *data_ptr = create_fake_data(num_samples, bias, slope, noise);


	train(data_ptr, 0.001);
	return 0;
}
