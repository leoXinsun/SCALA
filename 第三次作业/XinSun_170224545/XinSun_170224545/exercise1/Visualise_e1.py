import matplotlib.pyplot as plt
import csv

file_name = 'exercise1.csv'

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
plt.title("Visualise the 5811 papers using the 2 PCs")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('exercise.png')
