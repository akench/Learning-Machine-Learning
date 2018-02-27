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


	DataObj *data_ptr = malloc(sizeof(DataObj));
	data_ptr->num_samples = num_samples;
	data_ptr->bias = bias;
	data_ptr->slope = slope;
	data_ptr->noise = noise;
	data_ptr->x_data = x_data;
	data_ptr->y_data = y_data;
	
	return data_ptr;
}



int main(int argc, char *argv) {

	printf("%s\n", "args =  num_samples, bias, slope, noise");
	if(argc != 5) {
		printf("Incorrect number of args\n");
		return -1;
	}
	int num_samples = (int) argv[1];
	float bias = (float) argv[2];
	float slope = (float) argv[3];
	float noise = (float) argv[4];
	printf("samples %d, bias %lf, slope %lf, noise %lf\n", num_samples, bias, slope, noise);
	
	DataObj *data_ptr = create_fake_data(num_samples, bias, slope, noise);
	
}
