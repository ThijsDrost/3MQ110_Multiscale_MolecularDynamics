import math

import numpy as np
import numba
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, t_0, t_1, steps, particles, masses=None, save_every=1, verlet_type='basic',
                 boundary_condition=False):
        if verlet_type not in ['basic', 'velocity', 'euler']:
            raise ValueError(rf"`verlet_type` should be either 'basic' or 'velocity' not {verlet_type}")
        self.verlet_type = verlet_type

        if masses is None:
            self.particle_mass = np.ones((particles, 1))
        else:
            try:
                self.particle_mass = np.array(masses).reshape((particles, 1))
            except ValueError as e:
                raise ValueError('Something is wrong with the given particle masses, NumPy gave the following error:\n'+
                                 str(e))

        self.steps = steps
        self.particles = particles
        self.times = np.linspace(t_0, t_1, steps)
        self.dt = self.times[1] - self.times[0]
        self.save_every = save_every
        self.kinetic = np.zeros((steps//save_every))
        self.potential = np.zeros((steps//save_every))
        self.total_particle_velocities = np.zeros((steps//save_every, particles, 3))
        self.total_particle_locations = np.zeros((steps//save_every, particles, 3))
        self._flat_mass = self.particle_mass.flatten()

        self.particle_velocities = self._initial_vel(particles, self.particle_mass)
        self.particle_locations = self._initial_pos(particles)
        self.total_particle_locations[0] = self.particle_locations
        self.total_particle_velocities[0] = self.particle_velocities
        self.kinetic[0], self.potential[0] = self.calc_energies(self.particle_locations, self.particle_velocities,
                                                                self._flat_mass, self.particles)
        self.last_particle_locations = self.particle_locations

    def execute(self):
        if self.verlet_type == 'basic':
            def stepper(): self._do_step_loc(self._do_step_numba_basic)
        elif self.verlet_type == 'velocity':
            def stepper(): self._do_step_loc_vel(self._do_step_numba_velocity)
        elif self.verlet_type == 'euler':
            def stepper(): self._do_step_loc_vel(self._do_step_numba_euler)
        else:
            raise ValueError("`verlet_type` not recognized")

        # First step
        forces = calc_forces(self.particle_locations, self.particles)
        acc = forces / self.particle_mass
        self.particle_locations = self.last_particle_locations + self.total_particle_velocities[0] * self.dt \
                                  + 0.5 * acc * self.dt * self.dt
        self._save_energies(1)

        # Further steps
        for index, time_step in enumerate(self.times[2:], start=2):
            stepper()
            if (index % self.save_every) == 0:
                self._save_energies(index // self.save_every)
            if (index % int(self.steps/100)) == 0:
                print(f'step {index} of {self.steps}')

    def _do_step_loc(self, numba_func):
        new_particle_locations = numba_func(self.particle_mass, self.particle_locations, self.last_particle_locations,
                                            self.particles, self.dt)
        self.last_particle_locations = self.particle_locations
        self.particle_locations = new_particle_locations
        self.particle_velocities = (self.particle_locations - self.last_particle_locations) / self.dt

    def _do_step_loc_vel(self, numba_func):
        self.last_particle_locations = self.particle_locations
        self.particle_locations, self.particle_velocities = numba_func(self.particle_mass, self.particle_locations,
                                                                       self.particle_velocities, self.particles,
                                                                       self.dt)

    # def _do_step_basic(self):
    #     new_particle_locations = self._do_step_numba_basic(self.particle_mass, self.particle_locations,
    #                                                        self.last_particle_locations, self.particles, self.dt)
    #     self.last_particle_locations = self.particle_locations
    #     self.particle_locations = new_particle_locations
    #     self.particle_velocities = (self.particle_locations-self.last_particle_locations)/self.dt
    #
    # def _do_step_velocity(self):
    #     self.last_particle_locations = self.particle_locations
    #     self.particle_locations, self.particle_velocities = self._do_step_numba_velocity(self.particle_mass,
    #                                                                                      self.particle_locations,
    #                                                                                      self.particle_velocities,
    #                                                                                      self.particles, self.dt)
    #
    # def _do_step_euler(self):
    #     self.last_particle_locations = self.particle_locations
    #     self.particle_locations, self.particle_velocities = self._do_step_numba_euler(self.particle_mass,
    #                                                                                   self.particle_locations,
    #                                                                                   self.particle_velocities,
    #                                                                                   self.particles, self.dt)

    def _save_energies(self, index):
        self.total_particle_locations[index] = self.particle_locations
        self.total_particle_velocities[index] = self.particle_velocities
        self.kinetic[index], self.potential[index] = self.calc_energies(self.particle_locations,
                                                                        self.particle_velocities,
                                                                        self._flat_mass, self.particles)

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
                            velocity=self.total_particle_velocities, save_every=self.save_every)

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

        simulation = Simulation(times[0], times[1], len(times)*save_every, total_particle_locations.shape[1],
                                particle_mass, save_every)
        simulation.total_particle_velocities = total_particle_velocities
        for index in range(len(times)):
            simulation.kinetic[index], simulation.potential[index] = Simulation.calc_energies(total_particle_locations[index],
                                                                                              total_particle_velocities[index],
                                                                                              simulation._flat_mass,
                                                                                              simulation.particles)
        return simulation

    @staticmethod
    def run(t_0, t_1, steps, particles, masses=None, save_every=1, verlet_type='basic', boundary_condition=False):
        simulation = Simulation(t_0, t_1, steps, particles, masses, save_every, verlet_type, boundary_condition)
        simulation.execute()
        return simulation

    @staticmethod
    @numba.njit
    def _do_step_numba_basic(particle_mass, particle_loc, last_particle_loc, particles, dt):
        forces = calc_forces(particle_loc, particles)
        acc = forces / particle_mass
        new_particle_locations = 2 * particle_loc - last_particle_loc + acc * dt * dt
        return new_particle_locations

    @staticmethod
    @numba.njit
    def _do_step_numba_velocity(particle_mass, particle_loc, particle_velocity, particles, dt):
        forces = calc_forces(particle_loc, particles)
        acc = forces / particle_mass
        particle_velocity = particle_velocity + 0.5*acc*dt

        particle_loc += particle_velocity*dt

        forces = calc_forces(particle_loc, particles)
        acc = forces / particle_mass
        particle_velocity = particle_velocity + 0.5 * acc * dt
        return particle_loc, particle_velocity

    @staticmethod
    @numba.njit
    def _do_step_numba_euler(particle_mass, particle_loc, particle_vel, particles, dt):
        forces = calc_forces(particle_loc, particles)
        acc = forces / particle_mass
        particle_vel = particle_vel + acc*dt
        particle_loc = particle_loc + particle_vel*dt
        return particle_loc, particle_vel

    @staticmethod
    @numba.njit
    def calc_energies(particle_locations, particle_velocity, masses, particle_num):
        kinetic = np.sum(0.5*masses*np.sum(particle_velocity**2, axis=1))
        potential = 0
        for i in range(particle_num):
            for j in range(i+1, particle_num):
                xij = particle_locations[i, 0] - particle_locations[j, 0]
                yij = particle_locations[i, 1] - particle_locations[j, 1]
                zij = particle_locations[i, 2] - particle_locations[j, 2]

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
def calc_forces(particle_locations, particles):
    forces = np.zeros((particles, 3), dtype=float)
    for i in range(particles):
        for j in range(i+1, particles):
            xij = particle_locations[i, 0] - particle_locations[j, 0]
            yij = particle_locations[i, 1] - particle_locations[j, 1]
            zij = particle_locations[i, 2] - particle_locations[j, 2]

            rij2 = xij*xij + yij*yij + zij*zij

            factor = 4.0 * (12.0 / (rij2**7) - 6.0 / (rij2**4))

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
    sim = Simulation.run(0, 1e1, int(1e6), 2, save_every=100, verlet_type='velocity')
    sim.plot()
    plt.show()
    sim.plot_energy()
    plt.show()

    plt.figure()
    plt.plot(sim.total_energy[1:])
    plt.show()


