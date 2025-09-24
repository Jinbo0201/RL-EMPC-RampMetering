from src.simulation.input import Input
import matplotlib.pyplot as plt


input = Input()

print(len(input.demand_o))
print(len(input.get_input(0,100000000)[1]))

plt.figure()
plt.plot(input.get_input(0,100000000)[0], label='demand-o')
plt.plot(input.get_input(0,100000000)[1], label='demand-r')
plt.plot(input.get_input(0,100000000)[2], label='density-e')
plt.legend()

plt.show()