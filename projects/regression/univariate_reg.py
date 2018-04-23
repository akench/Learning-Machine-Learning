import matplotlib.pyplot as plt
import numpy as np
import time


def plot_state(theta_0, theta_1, data, it):

	x_vals_real = [sample[0] for sample in data]
	y_vals_real = [sample[1] for sample in data]

	plt.scatter(x_vals_real, y_vals_real)

	x_vals_fake = np.linspace(0., np.max(x_vals_real) + 5)
	y_vals_fake = [theta_0+theta_1*x for x in x_vals_fake]

	plt.plot(x_vals_fake, y_vals_fake, 'r--')
	plt.axis([0, np.max(x_vals_real)+5, -np.max(y_vals_real) - 5, np.max(y_vals_real)+5])
	plt.savefig('out/{}.png'.format(it), bbox_inches='tight')
	plt.gcf().clear()


def make_fake_data(num, bias, slope, noise_factor):
	min_x = 0.
	max_x = 50.

	data=[]
	for _ in range(num):
		x = np.random.uniform(min_x, max_x)
		data.append((x, slope*x + bias + noise_factor*np.random.normal()))

	np.random.shuffle(data)
	return data


#cost function = mean squared error
# 1/2m * sum(f(x) - y)^2
def calc_error(theta_0, theta_1, data):

	error_sum = [((theta_0 + theta_1*x) - y)**2 for x,y in data]
	error = np.sum(error_sum) / (len(data) * 2)
	return error


def update_vars(theta_0, theta_1, data, learning_rate):

	deriv_0 = [((theta_0 + theta_1*x) - y) for x,y in data]
	deriv_0	= np.sum(deriv_0) / len(data)

	deriv_1 = [((theta_0 + theta_1*x) - y)*x for x,y in data]
	deriv_1 = np.sum(deriv_1) / len(data)

	theta_0 = theta_0 - learning_rate*deriv_0
	theta_1 = theta_1 - learning_rate*deriv_1

	# print('deriv0=  {}      deriv1=  {}'.format(deriv_0, deriv_1))
	return theta_0, theta_1, deriv_0, deriv_1


def train(data, learning_rate = 0.001):

	
	theta_0 = np.random.normal()
	theta_1 = np.random.normal()

	error = calc_error(theta_0, theta_1, data)

	it = 0
	deriv_1 = 10000
	deriv_0 = 10000
	plot_state(theta_0, theta_1, data, it)
	while abs(deriv_0) + abs(deriv_1) > 0.000001:

		# print('theta0 = {}  theta1= {}'.format(theta_0, theta_1))

		error = calc_error(theta_0, theta_1, data)
		# print('ERROR   ', error)

		# if it % 10000 == 0:
			# plot_state(theta_0, theta_1, data, it)
		

		theta_0, theta_1, deriv_0, deriv_1 = update_vars(theta_0, theta_1, data, learning_rate)

		it += 1

	print('Line of best fit = {:0.4f}x + {:0.4f}'.format(theta_1, theta_0))
	plot_state(theta_0, theta_1, data, it)


inp = input('num_sample | bias | slope | noise\n')
inps = inp.split(' | ')
inps[0] = int(inps[0])
inps[1] = float(inps[1])
inps[2] = float(inps[2])
inps[3] = float(inps[3])

data = make_fake_data(*inps)

t0 = time.time()
train(data, learning_rate = 0.001)
print("took seconds ", time.time() - t0)


