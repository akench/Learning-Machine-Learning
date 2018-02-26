import matplotlib.pyplot as plt
import numpy as np


def plot_state(theta_0, theta_1, data, it):

	x_vals_real = [sample[0] for sample in data]
	y_vals_real = [sample[1] for sample in data]

	plt.scatter(x_vals_real, y_vals_real)

	x_vals_fake = np.linspace(0., np.max(x_vals_real))
	y_vals_fake = [theta_0+theta_1*x for x in x_vals_fake]

	plt.plot(x_vals_fake, y_vals_fake)
	plt.axis([0, np.max(x_vals_real), -np.max(y_vals_real), np.max(y_vals_real)])
	plt.savefig('out/{}.png'.format(it), bbox_inches='tight')
	plt.gcf().clear()


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


def update_vars(theta_0, theta_1, data, learning_rate):

	deriv_0 = [(theta_0 + theta_1*x - y) for x,y in data]
	deriv_0	= np.sum(deriv_0) / len(data)

	deriv_1 = [(theta_0 + theta_1*x - y)*x for x,y in data]
	deriv_1 = np.sum(deriv_1) / len(data)

	theta_0 = theta_0 - learning_rate*deriv_0
	theta_1 = theta_1 - learning_rate*deriv_1

	return theta_0, theta_1


def train(data = make_fake_data(num=50, slope=2), learning_rate = 0.001):

	theta_0 = np.random.normal()
	theta_1 = np.random.normal()

	error = calc_error(theta_0, theta_1, data)

	it = 0
	while error > 0.1:
		error = calc_error(theta_0, theta_1, data)
		print(error, 'is error')
		plot_state(theta_0, theta_1, data, it)

		theta_0, theta_1 = update_vars(theta_0, theta_1, data, learning_rate)
		it += 1

	print('Line of best fit = {:0.4f}x + {:0.4f}'.format(theta_1, theta_0))


train()




