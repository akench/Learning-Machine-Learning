#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct data_struct { 

	int num_samples;
	float bias;
	float slope;
	float noise;
	float *x_data;
	float *y_data;

} DataObj ;

void shuffle(float *arr1, float *arr2, size_t n) {
	if (n > 1) {
  	        size_t i;
			for (i = 0; i < n - 1; i++) {
				size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
				float t = arr1[j];
				arr1[j] = arr1[i];
				arr1[i] = t;
		
	  			t = arr2[j];
				arr2[j] = arr2[i];
				arr2[i] = t;
        }
    }
}

DataObj *create_fake_data(int num_samples, float bias, float slope, float noise) {
	
	float max_x = 50.0;

	float *x_data = (float *)malloc(num_samples * sizeof(float));
	float *y_data = (float *)malloc(num_samples * sizeof(float));	
	
	int i;
	for(i = 0; i < num_samples; i++) {
		float x = ((float)rand()/(float)(RAND_MAX)) * max_x;
		float y = bias + slope*x;

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

float abs_val(float x) {
	if(x < 0) {
		return -x;
	}
	else {
		return x;
	}
}


void update_vars(float *theta_0, float *theta_1, DataObj *data_ptr,  float learning_rate) {
	
	float deriv_0 = 0.0, deriv_1 = 0.0;

	int i;
	for(i = 0; i < data_ptr->num_samples, i++) {

		float x = data_ptr->x_data[i];
		float y = data_ptr->y_data[i];

		deriv_0 += ((*theta_0 + *theta_1 * x) - y);
		deriv_1 += ((*theta_0 + *theta_1 * x) - y) * x;

	}
	
	deriv_0 /= data_ptr->num_samples;
	deriv_1 /= data_ptr->num_samples;
	
	*theta_0 = *theta_0 - learning_rate * deriv_0;
	*theta_1 = *theta_1 - learning_rate * deriv_1;
	
}


void train(DataObj *data_ptr, float learning_rate) {
	
	float theta_0 = ((float)rand()/(float)(RAND_MAX)) * 2 - 1;
	float theta_1 = ((float)rand()/(float)(RAND_MAX)) * 2 - 1;
	
	int it = 0;
	float deriv_0 = 10000.0;
	float deriv_1 = 10000.0;

	while( abs_val(deriv_0) + abs_val(deriv_1) > 0.00001) {


		update_vars(&theta_0, &theta_1, data_ptr,  learning_rate);

	}	 	
}


int main(int argc, char *argv) {

	int num_samples;
	float bias, slope, noise;
	printf("Enter number of samples\n");
	scanf("%d", &num_samples);

	printf("Enter bias (float)\n");
	scanf("%f", &bias);
			
	printf("Enter slope (float)\n");
	scanf("%f", &slope);		
			
	printf("Enter noise (float)\n");
	scanf("%f", &noise);		
			
	printf("samples %d, bias %f, slope %f, noise %f\n", num_samples, bias, slope, noise);
	
	DataObj *data_ptr = create_fake_data(num_samples, bias, slope, noise);


	train(data_ptr, 0.001);
	return 0;
}
