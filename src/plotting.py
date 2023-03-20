import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from search_loss_graph import my_arr


def create_graph_arr():
	fig, ax = plt.subplots(1, 1, figsize=(10, 5))
	for values in my_arr:
		x = range(1,len(values)+1,1)
		ax.plot(x, values)
	plt.savefig('graph_plotting.png')

def create_graph(filename):
	f = open(filename,'r+')
	values = f.readlines()
	values = [float(v) for v in values]
	x = range(1,len(values)+1,1)
	fig, ax = plt.subplots(1, 1, figsize=(10, 7))
	ax.plot(x, values)
	ax.legend()
	plt.savefig('graph_'+filename.replace('.loss','.png'))


def main():
	for name in ["14_04_2022_run.loss", "14_04_2022_run_2.loss", "14_04_2022_run_3.loss", "14_04_2022_run_4.loss", "21_11_2022.loss", "21_11_2022_2.loss", "21_11_2022_3.loss"]:
		create_graph(name)


if __name__ == '__main__':
	#main()
	create_graph_arr()
