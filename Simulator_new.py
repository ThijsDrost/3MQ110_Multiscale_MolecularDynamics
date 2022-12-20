import math

import numpy as np
import numba
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, t, steps, particles, masses=None, save_every=1, verlet_type='basic',
                 boundary_condition=False, gaussian_mass=False):
        if verlet_type not in ['basic', 'velocity', 'euler']:
            raise ValueError(rf"`verlet_type` should be either 'basic' or 'velocity' not {verlet_type}")
        self.verlet_type = verlet_type

        if masses is None:
            if gaussian_mass:
                self.particle_mass = np.random.normal(1, 1, (particles, 1))
                self.particle_mass[self.particle_mass < 0] *= -1
            else:
                self.particle_mass = np.ones((particles, 1))
        else:
            try:
                self.particle_mass = np.array(masses).reshape((particles, 1))
            except ValueError as e:
                raise ValueError('Something is wrong with the given particle masses, NumPy gave the following error:\n'+
                                 str(e))

        self.steps = steps
        self.particles = particles
        self.boundary_condition = boundary_condition
        self.dt = t/(steps-1)
        self.save_every = save_every
        self.box_size = 1.2*(math.ceil(particles ** (1 / 3)))
        self._flat_mass = self.particle_mass.flatten()

        save_num = (steps//save_every)
        save_num += 1 if steps % save_every != 0 else 0

        self.times = np.linspace(0, t, steps)
        self.kinetic = np.zeros(save_num)
        self.potential = np.zeros(save_num)
        self.total_particle_velocities = np.zeros((save_num, particles, 3))
        self.total_particle_locations = np.zeros((save_num, particles, 3))
        self.particle_acceleration = np.zeros((particles, 3))

        self.particle_velocities = self._initial_vel(particles, self.particle_mass)
        self.particle_locations = self._initial_pos(particles)
        self.total_particle_locations[0] = self.particle_locations
        self.total_particle_velocities[0] = self.particle_velocities
        boxsize = self.box_size if boundary_condition else -1
        self.kinetic[0], self.potential[0] = self.calc_energies(self.particle_locations, self.particle_velocities,
                                                                self._flat_mass, self.particles, boxsize)
        self.last_particle_locations = self.particle_locations

    def execute(self):
        if self.boundary_condition:
            box_size = self.box_size
        else:
            box_size = -1

        if self.verlet_type == 'basic':
            def stepper(): self._do_step_verlet(box_size)
        elif self.verlet_type == 'velocity':
            def stepper(): self._do_step_velocity_verlet(box_size)
        elif self.verlet_type == 'euler':
            def stepper(): self._do_step_euler(box_size)
        else:
            raise ValueError("`verlet_type` not recognized")

        # First step
        forces = calc_forces(self.particle_locations, self.particles, box_size)
        acc = forces / self.particle_mass
        self.particle_locations = self.last_particle_locations + self.particle_velocities * self.dt \
                                  + 0.5 * acc * self.dt * self.dt
        self._save_energies(1)

        # Further steps
        for index, time_step in enumerate(self.times[2:], start=2):
            stepper()
            if (index % self.save_every) == 0:
                self._save_energies(index // self.save_every)
            if (index % int(self.steps/100)) == 0:
                print(f'\r{index // int(self.steps/100)}% done', end='')
        print(f'\r100% done', end='')
        print('')

    def _do_step_verlet(self, box_size):
        forces = calc_forces(self.particle_locations, self.particles, box_size)
        self.particle_acceleration = forces / self.particle_mass
        new_particle_locations = 2 * self.particle_locations - self.last_particle_locations \
                                  + self.particle_acceleration * self.dt * self.dt
        if box_size > 0:
            new_particle_locations = new_particle_locations % box_size

        self.last_particle_locations = self.particle_locations
        self.particle_locations = new_particle_locations
        self.particle_velocities = (self.particle_locations - self.last_particle_locations) / self.dt

    def _do_step_velocity_verlet(self, box_size):
        self.last_particle_locations = self.particle_locations
        self.particle_velocities += 0.5 * self.particle_acceleration * self.dt
        self.particle_locations = self.particle_locations + self.particle_velocities * self.dt

        forces = calc_forces(self.particle_locations, self.particles, box_size)
        self.particle_acceleration = forces / self.particle_mass
        self.particle_velocities += 0.5 * self.particle_acceleration * self.dt

        if box_size > 0:
            self.particle_locations = self.particle_locations % box_size

    def _do_step_euler(self, box_size):
        self.last_particle_locations = self.particle_locations
        forces = calc_forces(self.particle_locations, self.particles, box_size)
        self.particle_acceleration = forces / self.particle_mass
        self.particle_velocities += self.particle_acceleration * self.dt
        self.particle_locations = self.particle_locations + self.particle_velocities * self.dt
        if box_size > 0:
            self.particle_locations = self.particle_locations % box_size

    def _save_energies(self, index):
        self.total_particle_locations[index] = self.particle_locations
        self.total_particle_velocities[index] = self.particle_velocities
        boxsize = self.box_size if self.boundary_condition else -1
        self.kinetic[index], self.potential[index] = self.calc_energies(self.particle_locations,
                                                                        self.particle_velocities,
                                                                        self._flat_mass, self.particles, boxsize)

    def plot(self, start=0, stop=-1, every=1):
        ax = plt.figure().add_subplot(projection='3d')
        for i in range(len(self.total_particle_locations[0])):
            ax.plot(self.total_particle_locations[start:stop:every, i, 0],
                    self.total_particle_locations[start:stop:every, i, 1],
                    self.total_particle_locations[start:stop:every, i, 2])
    
    def plot_energy(self, start=0, stop=-1, every=1):
        plt.figure()
        plt.plot(self.times[::self.save_every][start:stop:every], self.kinetic[start:stop:every], label='kinetic')
        plt.plot(self.times[::self.save_every][start:stop:every], self.potential[start:stop:every], label='potential')
        plt.plot(self.times[::self.save_every][start:stop:every], self.total_energy[start:stop:every], label='total')
        plt.ylabel('Energy')  # TODO: unit?
        plt.xlabel('Time')  # TODO: unit?
        plt.legend()

    def save(self, loc):
        np.savez_compressed(loc, time=self.save_times, loc=self.total_particle_locations, masses=self.particle_mass,
                            velocity=self.total_particle_velocities, save_every=self.save_every,
                            boundary=[self.boundary_condition], boxsize=[self.box_size], verlet_type=[self.verlet_type])

    @property
    def total_energy(self):
        return self.potential + self.kinetic

    @property
    def save_times(self):
        return self.times[::self.save_every]

    @staticmethod
    def read(loc):
        read = np.load(loc)
        times = read['time']
        total_particle_locations = read['loc']
        particle_mass = read['masses']
        total_particle_velocities = read['velocity']
        save_every = read['save_every']
        boundary_condition = read['boundary'][0]
        box_size = read['boxsize'][0]
        verlet_type = read['verlet_type'][0]

        simulation = Simulation(t=times[-1], steps=len(times)*save_every, particles=total_particle_locations.shape[1],
                                masses=particle_mass, save_every=save_every, boundary_condition=boundary_condition,
                                verlet_type=verlet_type)
        simulation.box_size = box_size
        simulation.total_particle_locations = total_particle_locations
        simulation.total_particle_velocities = total_particle_velocities

        boxsize = box_size if boundary_condition else -1
        for index in range(len(times)):
            simulation.kinetic[index], simulation.potential[index] = Simulation.calc_energies(total_particle_locations[index],
                                                                                              total_particle_velocities[index],
                                                                                              simulation._flat_mass,
                                                                                              simulation.particles, boxsize)
        return simulation

    @staticmethod
    def load(loc):
        return Simulation.read(loc)

    @staticmethod
    def run(t, steps, particles, masses=None, save_every=1, verlet_type='basic', boundary_condition=False):
        simulation = Simulation(t, steps, particles, masses, save_every, verlet_type, boundary_condition)
        simulation.execute()
        return simulation

    @staticmethod
    @numba.njit
    def calc_energies(particle_locations, particle_velocity, masses, particle_num, boxsize):
        kinetic = np.sum(0.5*masses*np.sum(particle_velocity**2, axis=1))
        potential = 0
        inv_box_size = 1/boxsize
        for i in range(particle_num):
            for j in range(i+1, particle_num):
                xij = particle_locations[i, 0] - particle_locations[j, 0]
                yij = particle_locations[i, 1] - particle_locations[j, 1]
                zij = particle_locations[i, 2] - particle_locations[j, 2]

                if boxsize > 0:
                    xij = xij - boxsize * round(xij * inv_box_size)
                    yij = yij - boxsize * round(yij * inv_box_size)
                    zij = zij - boxsize * round(zij * inv_box_size)

                rij2 = xij * xij + yij * yij + zij * zij
                potential += 4.0 * (1.0 / (rij2 ** 6) - 1.0 / (rij2 ** 3))
        return kinetic, potential

    @staticmethod
    def _initial_pos(particles):  # TODO: Should not be static?
        particle_locations = np.zeros((particles, 3), dtype=float)
        particles_per_dimension = math.ceil(particles ** (1 / 3))
        value = 1.2 * np.linspace(0, particles_per_dimension, particles_per_dimension, endpoint=False)
        x, y, z = np.meshgrid(value, value, value, indexing='ij')
        particle_locations[:, 0] = x.flatten()[:particles]
        particle_locations[:, 1] = y.flatten()[:particles]
        particle_locations[:, 2] = z.flatten()[:particles]
        return particle_locations

    @staticmethod
    def _initial_vel(particles, particle_mass):  # TODO: Should not be static?
        initial_particle_velocities = np.random.uniform(-1, 1, (particles, 3))
        velocities = np.sum(particle_mass * initial_particle_velocities, axis=0)
        initial_particle_velocities -= velocities / np.sum(particle_mass)
        if np.any(np.sum(particle_mass * initial_particle_velocities, axis=0) > 1e-5):
            raise ValueError('Centre of mass moves')
        return initial_particle_velocities


@numba.njit()
def calc_forces(particle_locations, particles, boxsize):
    forces = np.zeros((particles, 3), dtype=float)
    inv_box_size = 1 / boxsize
    for i in range(particles):
        for j in range(i + 1, particles):
            xij = particle_locations[i, 0] - particle_locations[j, 0]
            yij = particle_locations[i, 1] - particle_locations[j, 1]
            zij = particle_locations[i, 2] - particle_locations[j, 2]

            if boxsize > 0:
                xij = xij - boxsize * round(xij * inv_box_size)
                yij = yij - boxsize * round(yij * inv_box_size)
                zij = zij - boxsize * round(zij * inv_box_size)

            rij2 = xij * xij + yij * yij + zij * zij

            factor = 4.0 * (12.0 / (rij2 ** 7) - 6.0 / (rij2 ** 4))

            Fijx = factor * xij
            Fijy = factor * yij
            Fijz = factor * zij

            forces[i, 0] += Fijx
            forces[i, 1] += Fijy
            forces[i, 2] += Fijz
            forces[j, 0] -= Fijx
            forces[j, 1] -= Fijy
            forces[j, 2] -= Fijz
    return forces


if __name__ == '__main__':
    np.random.seed(123456)
    sim = Simulation.run(1e1, int(1e5), 2, save_every=100, verlet_type='basic', boundary_condition=False)
    sim.plot()
    plt.show()
    sim.plot_energy()
    plt.show()

    plt.figure()
    plt.plot(sim.total_energy[1:])
    plt.show()


