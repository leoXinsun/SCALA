import matplotlib.pyplot as plt
import csv

file_names = ['movies.csv', 'users.csv']
picture_names = ['visualise movies using the 2 PCs', 'visualise users using the 2 PCs']

for i in range(len(file_names)):
	file_name = file_names[i]
	x = []
	with open(file_name) as f:
		reader = csv.reader(f)
		for row in reader:
			x.append(float(row[0]))
	y = []
	with open(file_name) as f:
		reader = csv.reader(f)

		for row in reader:
			y.append(float(row[1]))

	plt.figure()
	plt.scatter(x, y, s=1)
	plt.title(picture_names[i])
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig(file_name[:-4] + '.png')