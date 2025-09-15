from src.simulation.input import Input

input = Input()

print(len(input.demand_o))
print(len(input.get_input(0,100000000)[1]))