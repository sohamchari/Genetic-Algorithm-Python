from turtle import color
import matplotlib.pyplot as plt
import numpy as np

# Plot max fitness across generations
# maxFitness = [42, 29, 22, 17, 14]
# generations = [100, 200, 300, 400, 500]

# maxFitness = [49, 42, 31]
# generations = [50, 100, 200]

# plt.plot(generations, maxFitness)
# plt.title('Max fitness across #edges')
# plt.xlabel('Number of Edges')
# plt.ylabel('Max fitness')
# plt.show()


# x = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
# y = [35, 37, 39, 40, 42, 0, 0, 0, 0]
# z = [35, 36, 36, 38, 40, 40, 42, 42, 43]

# ax = plt.subplot(111)
# ax.bar(np.array(x)-0.8, y, width=0.8, color='r', align='center')
# ax.bar(x, z, width=0.8, color='b', align='center')
# plt.xlabel('Generations')
# plt.ylabel('Fitness')
# plt.title('Comparison of fitness with improvement (Generating 2 children)')
# plt.legend(['Without improvement', 'With improvement'])
# plt.show()

x = [50, 100, 200]
# y = [49, 42, 31]
# z = [50, 48, 36]

y = [45.28, 46.01, 46.97]
z = [0.94, 7.03, 9.5]

y = [40, 22, 13]
z = [37, 135, 120]

plt.plot(x, y, label='Without improvement', color='r')
plt.plot(x, z, label='With improvement', color='b')
plt.title('Comparison of number of generations')
plt.xlabel('Number of edges')
plt.ylabel('Number of generations')
plt.legend()
# plt.show()


# Save it as png
plt.savefig('Final Generations.png')