import timeit
import random

import numpy as np
import numba

@numba.njit
def modulo_mirror(value, mod):
    return np.fmod(value, mod)


@numba.njit
def if_mirror(value, mod):
    if value < -mod:
        return value + mod
    if value > mod:
        return value - mod
    return value


@numba.njit
def round_mirror(value, box, invbox):
    return value - box*round(value*invbox)


@numba.njit
def round_mirror2(value, box):
    return value - box*round(value/box)


def potential(value):
    return value*(2/value**7 - 1/value**4)


modulo_mirror(1.2, 2.0)
if_mirror(1.2, 2.0)
round_mirror(1.2, 2.0, 0.5)
potential(1.3224398)

values = np.random.normal(0.5, 0.33, 10_000)
grid_size = 1.2
inv_grid = 1/1.2

for x in [0.4326789, -0.75346, 1.5347892]:
    in_results = timeit.repeat('modulo_mirror(x, grid_size)', number=10_000, globals=globals(), repeat=1000)
    in_results1 = timeit.repeat('if_mirror(x, grid_size)', number=10_000, globals=globals(), repeat=1000)
    in_results2 = timeit.repeat('round_mirror(x, grid_size, inv_grid)', number=10_000, globals=globals(), repeat=1000)
    in_results5 = timeit.repeat('round_mirror2(x, grid_size)', number=10_000, globals=globals(),
                                repeat=100)
    print(x)
    print(f'mod: {np.average(in_results):.2e} {np.median(in_results):.2e} {np.std(in_results):.2e}')
    print(f'if: {np.average(in_results1):.2e} {np.median(in_results1):.2e} {np.std(in_results1):.2e}')
    print(f'round: {np.average(in_results2):.2e} {np.median(in_results2):.2e} {np.std(in_results2):.2e}')
    print(f'round2: {np.average(in_results5):.2e} {np.median(in_results5):.2e} {np.std(in_results5):.2e}')
    print('')


in_results = timeit.repeat('modulo_mirror(random.gauss(0.5, 0.33), grid_size)', number=10_000, globals=globals(), repeat=100)
in_results1 = timeit.repeat('if_mirror(random.gauss(0.5, 0.33), grid_size)', number=10_000, globals=globals(), repeat=100)
in_results2 = timeit.repeat('round_mirror(random.gauss(0.5, 0.33), grid_size, inv_grid)', number=10_000, globals=globals(), repeat=100)
in_results3 = timeit.repeat('random.gauss(0.5, 0.33)', number=10_000, globals=globals(), repeat=100)
in_results4 = timeit.repeat('potential(random.gauss(0.5, 0.33))', number=10_000, globals=globals(), repeat=100)
in_results5 = timeit.repeat('round_mirror2(random.gauss(0.5, 0.33), grid_size)', number=10_000, globals=globals(), repeat=100)
print('random')
print(f'mod: {np.average(in_results):.2e} {np.median(in_results):.2e} {np.std(in_results):.2e}')
print(f'if: {np.average(in_results1):.2e} {np.median(in_results1):.2e} {np.std(in_results1):.2e}')
print(f'round: {np.average(in_results2):.2e} {np.median(in_results2):.2e} {np.std(in_results2):.2e}')
print(f'round2: {np.average(in_results5):.2e} {np.median(in_results5):.2e} {np.std(in_results5):.2e}')
print(f'random: {np.average(in_results3):.2e} {np.median(in_results3):.2e} {np.std(in_results3):.2e}')
print(f'potential: {np.average(in_results4):.2e} {np.median(in_results4):.2e} {np.std(in_results4):.2e}')
print('')

in_results = timeit.repeat('modulo_mirror(random.uniform(-1.5, 1.5), grid_size)', number=10_000, globals=globals(), repeat=100)
in_results1 = timeit.repeat('if_mirror(random.uniform(-1.5, 1.5), grid_size)', number=10_000, globals=globals(), repeat=100)
in_results2 = timeit.repeat('round_mirror(random.uniform(-1.5, 1.5), grid_size, inv_grid)', number=10_000, globals=globals(), repeat=100)
in_results3 = timeit.repeat('random.uniform(-1.5, 1.5)', number=10_000, globals=globals(), repeat=100)
in_results4 = timeit.repeat('potential(random.uniform(-1.5, 1.5))', number=10_000, globals=globals(), repeat=100)
in_results5 = timeit.repeat('round_mirror2(random.uniform(-1.5, 1.5), grid_size)', number=10_000, globals=globals(), repeat=100)
print('random')
print(f'mod: {np.average(in_results):.2e} {np.median(in_results):.2e} {np.std(in_results):.2e}')
print(f'if: {np.average(in_results1):.2e} {np.median(in_results1):.2e} {np.std(in_results1):.2e}')
print(f'round: {np.average(in_results2):.2e} {np.median(in_results2):.2e} {np.std(in_results2):.2e}')
print(f'round2: {np.average(in_results5):.2e} {np.median(in_results5):.2e} {np.std(in_results5):.2e}')
print(f'random: {np.average(in_results3):.2e} {np.median(in_results3):.2e} {np.std(in_results3):.2e}')


