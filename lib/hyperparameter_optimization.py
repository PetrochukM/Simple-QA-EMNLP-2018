from math import log, ceil

# you need to write the following hooks for your custom problem
from problem import get_random_hyperparameter_configuration, run_then_return_val_loss


def ask():
    pass


def tell():
    pass


n_epochs = 81  # maximum iterations/epochs per configuration
eta = 3  # defines downsampling rate (default=3)
s_max = int(
    log(max_iter) / log(eta)) + 1  # number of unique executions of Successive Halving (minus one)
B = s_max * max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

#### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
for s in reversed(range(s_max)):
    n = int(ceil(B / max_iter / (s + 1) * eta**s))  # initial number of configurations
    r = max_iter * eta**(-s)  # initial number of iterations to run configurations for

    #### Begin Finite Horizon Successive Halving with (n,r)
    T = [get_random_hyperparameter_configuration() for i in range(n)]
    for i in range(s + 1):
        # Run each of the n_i configs for r_i iterations and keep best n_i/eta
        n_i = n * eta**(-i)
        r_i = r * eta**(i)
        val_losses = [run_then_return_val_loss(num_iters=r_i, hyperparameters=t) for t in T]
        T = [T[i] for i in argsort(val_losses)[0:int(n_i / eta)]]
    #### End Finite Horizon Successive Halving with (n,r)