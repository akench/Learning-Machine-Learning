import matplotlib.pyplot as plt
import numpy as np


def plot_state(theta_0, theta_1, data):

	x_vals_real = [sample[0] for sample in data]
	y_vals_real = [sample[1] for sample in data]

	plt.scatter(x_vals_real, y_vals_real)

	x_vals_fake = np.linspace(0., np.max(x_vals_real))
	y_vals_fake = [theta_0+theta_1*x for x in x_vals_fake]

	plt.plot(x_vals_fake, y_vals_fake)
	plt.show()


def make_fake_data(num, slope):
	min_x = 0.
	max_x = 50.

	data=[]
	for _ in range(num):
		x = np.random.uniform(min_x, max_x)
		data.append((x, slope*x))

	return data


#cost function = mean squared error
# 1/2m * sum(f(x) - y)^2
def calc_error(theta_0, theta_1, data):

	error_sum = 0.
	for x,y in data:
		error_sum += ((theta_0 + theta_1*x) - y)**2

	error = error_sum / (len(data) * 2)
	return error




def train():

	theta_0 = np.random.normal()
	theta_1 = np.random.normal()
	data = make_fake_data(50, 2)

	error = calc_error(theta_0, theta_1, data)
	print(error, 'is error')
	plot_state(theta_0, theta_1, data)


train()




