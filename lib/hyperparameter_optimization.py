import random
from math import log, ceil
from numpy import argsort


def get_random_hyperparameter_configuration():
    return {}


def run_then_return_val_loss(num_iters, hyperparameters):
    # print('run_then_return_val_loss', num_iters, hyperparameters)
    return random.randint(0, 10)


# NOTE: It is found that adjusting learning rate forces faster descent but not better in the overall
# run; therefore, it is less apropriate to tune learning rate with hyperband. 

max_iter = 81  # maximum iterations/epochs per configuration
max_initial_iterations = 9  # maximum number of initial iterations as the last rounds of 
eta = 3  # defines downsampling rate (default=3)
s_max = int(
    log(max_initial_iterations) / log(eta)) + 1  # number of unique executions of Successive Halving
B = s_max * max_initial_iterations  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

print('Maximum iterations of initial:', max_initial_iterations)
print('Maximum iterations of objective function:', max_iter)
print('Downsampling rate:', eta)
print('Number of executions of successive halving:', s_max)
print('Iterations per execution of successive halving:', B)

# Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
for s in reversed(range(s_max)):
    n = int(ceil(B / max_initial_iterations / (s + 1) * eta**s))  # initial number of configurations
    r = max_iter * eta**(-s)  # initial number of iterations to run configurations for
    print('Initial number of config:', n)
    print('Initial number of iterations of objective function:', r)
    print('Number of rounds:', s + 1)

    # Begin Finite Horizon Successive Halving with (n,r)
    T = [get_random_hyperparameter_configuration() for i in range(n)]
    for i in range(s + 1):
        # Run each of the n_i configs for r_i iterations and keep best n_i/eta
        print('Number of configurations:', len(T))
        n_i = n * eta**(-i)
        r_i = r * eta**(i)
        val_losses = [run_then_return_val_loss(num_iters=r_i, hyperparameters=t) for t in T]
        T = [T[i] for i in argsort(val_losses)[0:int(n_i / eta)]]
    # End Finite Horizon Successive Halving with (n,r)
