""" This module provides classes that that can be used to setup and
    run simulation models of a ship in transit.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import NamedTuple, List


class ShipConfiguration(NamedTuple):
    dead_weight_tonnage: float
    coefficient_of_deadweight_to_displacement: float
    bunkers: float
    ballast: float
    length_of_ship: float
    width_of_ship: float
    added_mass_coefficient_in_surge: float
    added_mass_coefficient_in_sway: float
    added_mass_coefficient_in_yaw: float
    mass_over_linear_friction_coefficient_in_surge: float
    mass_over_linear_friction_coefficient_in_sway: float
    mass_over_linear_friction_coefficient_in_yaw: float
    nonlinear_friction_coefficient__in_surge: float
    nonlinear_friction_coefficient__in_sway: float
    nonlinear_friction_coefficient__in_yaw: float


class EnvironmentConfiguration(NamedTuple):
    current_velocity_component_from_north: float
    current_velocity_component_from_east: float
    wind_speed: float
    wind_direction: float


class SimulationConfiguration(NamedTuple):
    initial_north_position_m: float
    initial_east_position_m: float
    initial_yaw_angle_rad: float
    initial_forward_speed_m_per_s: float
    initial_sideways_speed_m_per_s: float
    initial_yaw_rate_rad_per_s: float
    integration_step: float
    simulation_time: float


class LoadOnPowerSources(NamedTuple):
    load_on_main_engine: float
    load_on_electrical: float
    load_percentage_on_main_engine: float
    load_percentage_on_electrical: float


class MachineryModeParams(NamedTuple):
    main_engine_capacity: float
    electrical_capacity: float
    shaft_generator_state: str


class MachineryMode:
    def __init__(self, params: MachineryModeParams):
        self.main_engine_capacity = params.main_engine_capacity
        self.electrical_capacity = params.electrical_capacity
        self.shaft_generator_state = params.shaft_generator_state
        self.available_propulsion_power = 0
        self.available_propulsion_power_main_engine = 0
        self.available_propulsion_power_electrical = 0

    def update_available_propulsion_power(self, hotel_load):
        if self.shaft_generator_state == 'MOTOR':
            self.available_propulsion_power = self.main_engine_capacity + self.electrical_capacity - hotel_load
            self.available_propulsion_power_main_engine = self.main_engine_capacity
            self.available_propulsion_power_electrical = self.electrical_capacity - hotel_load
        elif self.shaft_generator_state == 'GEN':
            self.available_propulsion_power = self.main_engine_capacity - hotel_load
            self.available_propulsion_power_main_engine = self.main_engine_capacity - hotel_load
            self.available_propulsion_power_electrical = 0
        else:  # shaft_generator_state == 'off'
            self.available_propulsion_power = self.main_engine_capacity
            self.available_propulsion_power_main_engine = self.main_engine_capacity
            self.available_propulsion_power_electrical = 0


    def distribute_load(self, load_perc, hotel_load):
        total_load_propulsion = load_perc * self.available_propulsion_power
        if self.shaft_generator_state == 'MOTOR':
            load_main_engine = min(total_load_propulsion, self.main_engine_capacity)
            load_electrical = total_load_propulsion + hotel_load - load_main_engine
            load_percentage_electrical = load_electrical / self.electrical_capacity
            if self.main_engine_capacity == 0:
                load_percentage_main_engine = 0
            else:
                load_percentage_main_engine = load_main_engine / self.main_engine_capacity
        elif self.shaft_generator_state == 'GEN':
            # Here the rule is that electrical handles hotel as far as possible
            load_electrical = min(hotel_load, self.electrical_capacity)
            load_main_engine = total_load_propulsion + hotel_load - load_electrical
            load_percentage_main_engine = load_main_engine / self.main_engine_capacity
            if self.electrical_capacity == 0:
                load_percentage_electrical = 0
            else:
                load_percentage_electrical = load_electrical / self.electrical_capacity
        else:  # shaft_generator_state == 'off'
            load_main_engine = total_load_propulsion
            load_electrical = hotel_load
            load_percentage_main_engine = load_main_engine / self.main_engine_capacity
            load_percentage_electrical = load_electrical / self.electrical_capacity

        return LoadOnPowerSources(
            load_on_main_engine=load_main_engine,
            load_on_electrical=load_electrical,
            load_percentage_on_main_engine=load_percentage_main_engine,
            load_percentage_on_electrical=load_percentage_electrical
        )


class MachineryModes:
    def __init__(self, list_of_modes: List[MachineryMode]):
        self.list_of_modes = list_of_modes


class FuelConsumptionCoefficients(NamedTuple):
    a: float
    b: float
    c: float


class MachinerySystemConfiguration(NamedTuple):
    hotel_load: float
    machinery_modes: MachineryModes
    machinery_operating_mode: int
    rated_speed_main_engine_rpm: float
    linear_friction_main_engine: float
    linear_friction_hybrid_shaft_generator: float
    gear_ratio_between_main_engine_and_propeller: float
    gear_ratio_between_hybrid_shaft_generator_and_propeller: float
    propeller_inertia: float
    propeller_speed_to_torque_coefficient: float
    propeller_diameter: float
    propeller_speed_to_thrust_force_coefficient: float
    rudder_angle_to_sway_force_coefficient: float
    rudder_angle_to_yaw_force_coefficient: float
    max_rudder_angle_degrees: float
    specific_fuel_consumption_coefficients_me: FuelConsumptionCoefficients
    specific_fuel_consumption_coefficients_dg: FuelConsumptionCoefficients


class SimplifiedPropulsionMachinerySystemConfiguration(NamedTuple):
    hotel_load: float
    machinery_modes: MachineryModes
    machinery_operating_mode: int
    specific_fuel_consumption_coefficients_me: FuelConsumptionCoefficients
    specific_fuel_consumption_coefficients_dg: FuelConsumptionCoefficients
    thrust_force_dynamic_time_constant: float
    rudder_angle_to_sway_force_coefficient: float
    rudder_angle_to_yaw_force_coefficient: float
    max_rudder_angle_degrees: float


class SpecificFuelConsumptionWartila6L26:
    def __init__(self):
        self.a = 128.9
        self.b = -168.9
        self.c = 246.8

    def fuel_consumption_coefficients(self):
        return FuelConsumptionCoefficients(
            a=self.a,
            b=self.b,
            c=self.c
        )


class SpecificFuelConsumptionBaudouin6M26Dot3:
    def __init__(self):
        self.a = 108.7
        self.b = -289.9
        self.c = 324.9

    def fuel_consumption_coefficients(self):
        return FuelConsumptionCoefficients(
            a=self.a,
            b=self.b,
            c=self.c
        )


class RudderConfiguration(NamedTuple):
    rudder_angle_to_sway_force_coefficient: float
    rudder_angle_to_yaw_force_coefficient: float
    max_rudder_angle_degrees: float


class BaseMachineryModel:
    def __init__(self,
                 fuel_coeffs_for_main_engine: FuelConsumptionCoefficients,
                 fuel_coeffs_for_diesel_gen: FuelConsumptionCoefficients,
                 rudder_config: RudderConfiguration,
                 machinery_modes: MachineryModes,
                 hotel_load: float,
                 operating_mode: int,
                 time_step: float):
        self.machinery_modes = machinery_modes
        self.hotel_load = hotel_load  # 200000  # 0.2 MW
        self.update_available_propulsion_power()
        self.mode = self.machinery_modes.list_of_modes[operating_mode]

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(time_step)

        self.c_rudder_v = rudder_config.rudder_angle_to_sway_force_coefficient
        self.c_rudder_r = rudder_config.rudder_angle_to_yaw_force_coefficient
        self.rudder_ang_max = rudder_config.max_rudder_angle_degrees * np.pi / 180

        self.fuel_coeffs_for_main_engine = fuel_coeffs_for_main_engine
        self.fuel_coeffs_for_diesel_gen = fuel_coeffs_for_diesel_gen
        self.fuel_cons_me = 0.0  # Initial fuel cons for ME
        self.fuel_cons_electrical = 0.0  # Initial fuel cons for HSG
        self.fuel_cons = 0.0  # Initial total fuel cons
        self.power_me = []  # Array for storing ME power cons. data
        self.power_hsg = []  # Array for storing HSG power cons. data
        self.me_rated = []  # Array for storing ME rated power data
        self.hsg_rated = []  # Array for storing HSG rated power data
        self.load_hist = []  # Array for storing load percentage history
        self.fuel_rate_me = []  # Array for storing ME fuel cons. rate
        self.fuel_rate_hsg = []  # Array for storing HSG fuel cons. rate
        self.fuel_me = []  # Array for storing ME fuel cons.
        self.fuel_hsg = []  # Array for storing HSG fuel cons.
        self.fuel = []  # Array for storing total fuel cons
        self.fuel_rate = []
        self.load_perc_me = []
        self.load_perc_hsg = []
        self.power_total = []
        self.power_prop = []

    def update_available_propulsion_power(self):
        for mode in self.machinery_modes.list_of_modes:
            mode.update_available_propulsion_power(self.hotel_load)

    def mode_selector(self, mode: int):
        self.mode = self.machinery_modes.list_of_modes[mode]

    def load_perc(self, load_perc):
        """ Calculates the load percentage on the main engine and the diesel_gens based on the
            operating mode of the machinery system (MSO-mode).

            Args:
                load_perc (float): Current load on the machinery system as a fraction of the
                    total power that can be delivered by the machinery system in the current mode.
            Returns:
                load_perc_me (float): Current load on the ME as a fraction of ME MCR
                load_perc_hsg (float): Current load on the HSG as a fraction of HSG MCR
        """
        load_data = self.mode.distribute_load(load_perc=load_perc, hotel_load=self.hotel_load)
        return load_data.load_percentage_on_main_engine, load_data.load_percentage_on_electrical

    @staticmethod
    def spec_fuel_cons(load_perc, coeffs: FuelConsumptionCoefficients):
        """ Calculate fuel consumption rate for engine.
        """
        rate = coeffs.a * load_perc ** 2 + coeffs.b * load_perc + coeffs.c
        return rate / 3.6e9

    def fuel_consumption(self, load_perc):
        '''
            Args:
                load_perc (float): The fraction of produced power over the online power production capacity.
            Returns:
                rate_me (float): Fuel consumption rate for the main engine
                rate_hsg (float): Fuel consumption rate for the HSG
                fuel_cons_me (float): Accumulated fuel consumption for the ME
                fuel_cons_hsg (float): Accumulated fuel consumption for the HSG
                fuel_cons (float): Total accumulated fuel consumption for the ship
        '''
        load_data = self.mode.distribute_load(load_perc=load_perc, hotel_load=self.hotel_load)
        if load_data.load_on_main_engine == 0:
            rate_me = 0
        else:
            rate_me = load_data.load_on_main_engine * self.spec_fuel_cons(
                load_data.load_percentage_on_main_engine, coeffs=self.fuel_coeffs_for_main_engine
            )

        if load_data.load_percentage_on_electrical == 0:
            rate_electrical = 0
        else:
            rate_electrical = load_data.load_on_electrical * self.spec_fuel_cons(
                load_data.load_percentage_on_electrical, coeffs=self.fuel_coeffs_for_diesel_gen
            )

        self.fuel_cons_me = self.fuel_cons_me + rate_me * self.int.dt
        self.fuel_cons_electrical = self.fuel_cons_electrical + rate_electrical * self.int.dt
        self.fuel_cons = self.fuel_cons + (rate_me + rate_electrical) * self.int.dt
        return rate_me, rate_electrical, self.fuel_cons_me, self.fuel_cons_electrical, self.fuel_cons


class ShipMachineryModel(BaseMachineryModel):
    def __init__(self,
                 machinery_config: MachinerySystemConfiguration,
                 initial_propeller_shaft_speed_rad_per_sec: float,
                 time_step: float,
                 ):
        super().__init__(
            fuel_coeffs_for_main_engine=machinery_config.specific_fuel_consumption_coefficients_me,
            fuel_coeffs_for_diesel_gen=machinery_config.specific_fuel_consumption_coefficients_dg,
            rudder_config=RudderConfiguration(
                rudder_angle_to_yaw_force_coefficient=machinery_config.rudder_angle_to_yaw_force_coefficient,
                rudder_angle_to_sway_force_coefficient=machinery_config.rudder_angle_to_sway_force_coefficient,
                max_rudder_angle_degrees=machinery_config.max_rudder_angle_degrees
            ),
            machinery_modes=machinery_config.machinery_modes,
            hotel_load=machinery_config.hotel_load,
            operating_mode=machinery_config.machinery_operating_mode,
            time_step=time_step)
        self.w_rated_me = machinery_config.rated_speed_main_engine_rpm * np.pi / 30
        self.d_me = machinery_config.linear_friction_main_engine
        self.d_hsg = machinery_config.linear_friction_hybrid_shaft_generator
        self.r_me = machinery_config.gear_ratio_between_main_engine_and_propeller
        self.r_hsg = machinery_config.gear_ratio_between_hybrid_shaft_generator_and_propeller
        self.jp = machinery_config.propeller_inertia
        self.kp = machinery_config.propeller_speed_to_torque_coefficient
        self.dp = machinery_config.propeller_diameter
        self.kt = machinery_config.propeller_speed_to_thrust_force_coefficient
        self.shaft_speed_max = 1.1 * self.w_rated_me * self.r_me

        self.omega = initial_propeller_shaft_speed_rad_per_sec
        self.d_omega = 0

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(time_step)

        self.specific_fuel_coeffs_for_main_engine = FuelConsumptionCoefficients(a=128.89, b=-168.93, c=246.76)
        self.specific_fuel_coeffs_for_dg = FuelConsumptionCoefficients(a=180.71, b=-289.90, c=324.90)
        self.fuel_cons_me = 0.0  # Initial fuel cons for ME
        self.fuel_cons_electrical = 0.0  # Initial fuel cons for HSG
        self.fuel_cons = 0.0  # Initial total fuel cons
        self.power_me = []  # Array for storing ME power cons. data
        self.power_hsg = []  # Array for storing HSG power cons. data
        self.me_rated = []  # Array for storing ME rated power data
        self.hsg_rated = []  # Array for storing HSG rated power data
        self.load_hist = []  # Array for storing load percentage history
        self.fuel_rate_me = []  # Array for storing ME fuel cons. rate
        self.fuel_rate_hsg = []  # Array for storing HSG fuel cons. rate
        self.fuel_me = []  # Array for storing ME fuel cons.
        self.fuel_hsg = []  # Array for storing HSG fuel cons.
        self.fuel = []  # Array for storing total fuel cons
        self.fuel_rate = []
        self.load_perc_me = []
        self.load_perc_hsg = []
        self.power_total = []
        self.power_prop = []

    def shaft_eq(self, torque_main_engine, torque_hsg):
        ''' Updates the time differential of the shaft speed
            equation.
        '''
        eq_me = (torque_main_engine - self.d_me * self.omega) / self.r_me
        eq_hsg = (torque_hsg - self.d_hsg * self.omega) / self.r_hsg
        self.d_omega = (eq_me + eq_hsg - self.kp * self.omega ** 2) / self.jp

    def thrust(self):
        ''' Updates the thrust force based on the shaft speed (self.omega)
        '''
        return self.dp ** 4 * self.kt * self.omega * abs(self.omega)

    def main_engine_torque(self, load_perc):
        ''' Returns the torque of the main engine as a
            function of the load percentage parameter
        '''
        if load_perc is None:
            return 0
        return min(load_perc * self.mode.available_propulsion_power_main_engine / (self.omega + 0.1),
                       self.mode.available_propulsion_power_main_engine / 5 * np.pi / 30)

    def hsg_torque(self, load_perc):
        ''' Returns the torque of the HSG as a
            function of the load percentage parameter
        '''
        if load_perc is None:
            return 0
        return min(load_perc * self.mode.available_propulsion_power_electrical / (self.omega + 0.1),
                   self.mode.available_propulsion_power_electrical / 5 * np.pi / 30)

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead
        '''
        self.omega = self.int.integrate(x=self.omega, dx=self.d_omega)

    def update_shaft_equation(self, load_percentage):
        self.shaft_eq(
            torque_main_engine=self.main_engine_torque(load_perc=load_percentage),
            torque_hsg=self.hsg_torque(load_perc=load_percentage)
        )


class SimplifiedMachineryModel(BaseMachineryModel):
    def __init__(self, machinery_config: SimplifiedPropulsionMachinerySystemConfiguration,
                 time_step: float,
                 initial_thrust_force: float):
        super().__init__(
            fuel_coeffs_for_main_engine=machinery_config.specific_fuel_consumption_coefficients_me,
            fuel_coeffs_for_diesel_gen=machinery_config.specific_fuel_consumption_coefficients_dg,
            rudder_config=RudderConfiguration(
                rudder_angle_to_sway_force_coefficient=machinery_config.rudder_angle_to_sway_force_coefficient,
                rudder_angle_to_yaw_force_coefficient=machinery_config.rudder_angle_to_yaw_force_coefficient,
                max_rudder_angle_degrees=machinery_config.max_rudder_angle_degrees
            ),
            machinery_modes=machinery_config.machinery_modes,
            hotel_load=machinery_config.hotel_load,
            operating_mode=machinery_config.machinery_operating_mode,
            time_step=time_step)

        self.update_available_propulsion_power()

        self.thrust = initial_thrust_force
        self.d_thrust = 0
        self.k_thrust = 2160 / 790
        self.thrust_time_constant = machinery_config.thrust_force_dynamic_time_constant

    def update_thrust_force(self, load_perc):
        ''' Updates the thrust force based on engine power
        '''
        power = load_perc * (self.mode.available_propulsion_power_main_engine
                             + self.mode.available_propulsion_power_electrical)
        self.d_thrust = (-self.k_thrust * self.thrust + power) / self.thrust_time_constant

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead
        '''
        self.thrust = self.int.integrate(x=self.thrust, dx=self.d_thrust)


class BaseShipModel:
    def __init__(
            self, ship_config: ShipConfiguration,
            simulation_config: SimulationConfiguration,
            environment_config: EnvironmentConfiguration
    ):

        payload = 0.9 * (ship_config.dead_weight_tonnage - ship_config.bunkers)
        lsw = ship_config.dead_weight_tonnage / ship_config.coefficient_of_deadweight_to_displacement \
              - ship_config.dead_weight_tonnage
        self.mass = lsw + payload + ship_config.bunkers + ship_config.ballast

        self.l_ship = ship_config.length_of_ship  # 80
        self.w_ship = ship_config.width_of_ship  # 16.0
        self.x_g = 0
        self.i_z = self.mass * (self.l_ship ** 2 + self.w_ship ** 2) / 12

        # zero-frequency added mass
        self.x_du, self.y_dv, self.n_dr = self.set_added_mass(ship_config.added_mass_coefficient_in_surge,
                                                              ship_config.added_mass_coefficient_in_sway,
                                                              ship_config.added_mass_coefficient_in_yaw)

        self.t_surge = ship_config.mass_over_linear_friction_coefficient_in_surge
        self.t_sway = ship_config.mass_over_linear_friction_coefficient_in_sway
        self.t_yaw = ship_config.mass_over_linear_friction_coefficient_in_yaw
        self.ku = ship_config.nonlinear_friction_coefficient__in_surge  # 2400.0  # non-linear friction coeff in surge
        self.kv = ship_config.nonlinear_friction_coefficient__in_sway  # 4000.0  # non-linear friction coeff in sway
        self.kr = ship_config.nonlinear_friction_coefficient__in_yaw  # 400.0  # non-linear friction coeff in yaw

        # Environmental conditions
        self.vel_c = np.array([environment_config.current_velocity_component_from_north,
                               environment_config.current_velocity_component_from_east,
                               0.0])
        self.wind_dir = environment_config.wind_direction
        self.wind_speed = environment_config.wind_speed

        # Initialize states
        self.north = simulation_config.initial_north_position_m
        self.east = simulation_config.initial_east_position_m
        self.yaw_angle = simulation_config.initial_yaw_angle_rad
        self.forward_speed = simulation_config.initial_forward_speed_m_per_s
        self.sideways_speed = simulation_config.initial_sideways_speed_m_per_s
        self.yaw_rate = simulation_config.initial_yaw_rate_rad_per_s

        # Initialize differentials
        self.d_north = 0
        self.d_east = 0
        self.d_yaw = 0
        self.d_forward_speed = 0
        self.d_sideways_speed = 0
        self.d_yaw_rate = 0

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(simulation_config.integration_step)
        self.int.set_sim_time(simulation_config.simulation_time)

        # Instantiate ship draw plotting
        self.drw = ShipDraw()  # Instantiate the ship drawing class
        self.ship_drawings = [[], []]  # Arrays for storing ship drawing data

        # Setup wind effect on ship
        self.rho_a = 1.2
        self.h_f = 8.0  # mean height above water seen from the front
        self.h_s = 8.0  # mean height above water seen from the side
        self.proj_area_f = self.w_ship * self.h_f  # Projected are from the front
        self.proj_area_l = self.l_ship * self.h_s  # Projected area from the side
        self.cx = 0.5
        self.cy = 0.7
        self.cn = 0.08

    def set_added_mass(self, surge_coeff, sway_coeff, yaw_coeff):
        ''' Sets the added mass in surge due to surge motion, sway due
            to sway motion and yaw due to yaw motion according to given coeffs.

            args:
                surge_coeff (float): Added mass coefficient in surge direction due to surge motion
                sway_coeff (float): Added mass coefficient in sway direction due to sway motion
                yaw_coeff (float): Added mass coefficient in yaw direction due to yaw motion
            returns:
                x_du (float): Added mass in surge
                y_dv (float): Added mass in sway
                n_dr (float): Added mass in yaw
        '''
        x_du = self.mass * surge_coeff
        y_dv = self.mass * sway_coeff
        n_dr = self.i_z * yaw_coeff
        return x_du, y_dv, n_dr

    def get_wind_force(self):
        ''' This method calculates the forces due to the relative
            wind speed, acting on teh ship in surge, sway and yaw
            direction.

            :return: Wind force acting in surge, sway and yaw
        '''
        uw = self.wind_speed * np.cos(self.wind_dir - self.yaw_angle)
        vw = self.wind_speed * np.sin(self.wind_dir - self.yaw_angle)
        u_rw = uw - self.forward_speed
        v_rw = vw - self.sideways_speed
        gamma_rw = -np.arctan2(v_rw, u_rw)
        wind_rw2 = u_rw ** 2 + v_rw ** 2
        c_x = -self.cx * np.cos(gamma_rw)
        c_y = self.cy * np.sin(gamma_rw)
        c_n = self.cn * np.sin(2 * gamma_rw)
        tau_coeff = 0.5 * self.rho_a * wind_rw2
        tau_u = tau_coeff * c_x * self.proj_area_f
        tau_v = tau_coeff * c_y * self.proj_area_l
        tau_n = tau_coeff * c_n * self.proj_area_l * self.l_ship
        return np.array([tau_u, tau_v, tau_n])

    def three_dof_kinematics(self):
        ''' Updates the time differientials of the north position, east
            position and yaw angle. Should be called in the simulation
            loop before the integration step.
        '''
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])
        dx = np.dot(self.rotation(), vel)
        self.d_north = dx[0]
        self.d_east = dx[1]
        self.d_yaw = dx[2]

    def rotation(self):
        ''' Specifies the rotation matrix for rotations about the z-axis, such that
            "body-fixed coordinates" = rotation x "North-east-down-fixed coordinates" .
        '''
        return np.array([[np.cos(self.yaw_angle), -np.sin(self.yaw_angle), 0],
                         [np.sin(self.yaw_angle), np.cos(self.yaw_angle), 0],
                         [0, 0, 1]])

    def mass_matrix(self):
        return np.array([[self.mass + self.x_du, 0, 0],
                         [0, self.mass + self.y_dv, self.mass * self.x_g],
                         [0, self.mass * self.x_g, self.i_z + self.n_dr]])

    def coriolis_matrix(self):
        return np.array([[0, 0, -self.mass * (self.x_g * self.yaw_rate + self.sideways_speed)],
                         [0, 0, self.mass * self.forward_speed],
                         [self.mass * (self.x_g * self.yaw_rate + self.sideways_speed),
                          -self.mass * self.forward_speed, 0]])

    def coriolis_added_mass_matrix(self, u_r, v_r):
        return np.array([[0, 0, self.y_dv * v_r],
                        [0, 0, -self.x_du * u_r],
                        [-self.y_dv * v_r, self.x_du * u_r, 0]])

    def linear_damping_matrix(self):
        return np.array([[self.mass / self.t_surge, 0, 0],
                      [0, self.mass / self.t_sway, 0],
                      [0, 0, self.i_z / self.t_yaw]])

    def non_linear_damping_matrix(self):
        return np.array([[self.ku * self.forward_speed, 0, 0],
                       [0, self.kv * self.sideways_speed, 0],
                       [0, 0, self.kr * self.yaw_rate]])

    def three_dof_kinetics(self, *args, **kwargs):
        ''' Calculates accelerations of the ship, as a funciton
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        wind_force = self.get_wind_force()
        wave_force = np.array([0, 0, 0])

        # assembling state vector
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.forward_speed - v_c[0]
        v_r = self.sideways_speed - v_c[1]

        # Kinetic equation
        m_inv = np.linalg.inv(self.mass_matrix())
        dx = np.dot(
            m_inv,
            -np.dot(self.coriolis_matrix(), vel)
            - np.dot(self.coriolis_added_mass_matrix(u_r=u_r, v_r=v_r), vel - v_c)
            - np.dot(self.linear_damping_matrix() + self.non_linear_damping_matrix(), vel - v_c)
            + wind_force + wave_force
        )
        self.d_forward_speed = dx[0]
        self.d_sideways_speed = dx[1]
        self.d_yaw_rate = dx[2]

    def update_differentials(self, *args, **kwargs):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        self.three_dof_kinetics()

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.north = self.int.integrate(x=self.north, dx=self.d_north)
        self.east = self.int.integrate(x=self.east, dx=self.d_east)
        self.yaw_angle = self.int.integrate(x=self.yaw_angle, dx=self.d_yaw)
        self.forward_speed = self.int.integrate(x=self.forward_speed, dx=self.d_forward_speed)
        self.sideways_speed = self.int.integrate(x=self.sideways_speed, dx=self.d_sideways_speed)
        self.yaw_rate = self.int.integrate(x=self.yaw_rate, dx=self.d_yaw_rate)

    def ship_snap_shot(self):
        ''' This method is used to store a map-view snap shot of
            the ship at the given north-east position and heading.
            It uses the ShipDraw-class. To plot a map view of the
            n-th ship snap-shot, use:

            plot(ship_drawings[1][n], ship_drawings[0][n])
        '''
        x, y = self.drw.local_coords()
        x_ned, y_ned = self.drw.rotate_coords(x, y, self.yaw_angle)
        x_ned_trans, y_ned_trans = self.drw.translate_coords(x_ned, y_ned, self.north, self.east)
        self.ship_drawings[0].append(x_ned_trans)
        self.ship_drawings[1].append(y_ned_trans)


class ShipModel(BaseShipModel):
    ''' Creates a ship model object that can be used to simulate a ship in transit

        The ships model is propelled by a single propeller and steered by a rudder.
        The propeller is powered by either the main engine, an auxiliary motor
        referred to as the hybrid shaft generator, or both. The model contains the
        following states:
        - North position of ship
        - East position of ship
        - Yaw angle (relative to north axis)
        - Surge velocity (forward)
        - Sway velocity (sideways)
        - Yaw rate
        - Propeller shaft speed

        Simulation results are stored in the instance variable simulation_results
    '''
    def __init__(self, ship_config: ShipConfiguration, simulation_config: SimulationConfiguration,
                 environment_config: EnvironmentConfiguration, machinery_config: MachinerySystemConfiguration,
                 initial_propeller_shaft_speed_rad_per_s):
        super().__init__(ship_config, simulation_config, environment_config)
        self.ship_machinery_model = ShipMachineryModel(
            machinery_config=machinery_config,
            initial_propeller_shaft_speed_rad_per_sec=initial_propeller_shaft_speed_rad_per_s,
            time_step=self.int.dt
        )
        self.simulation_results = defaultdict(list)

    def three_dof_kinetics(self, thrust_force=None, rudder_angle=None, load_percentage=None, *args, **kwargs):
        ''' Calculates accelerations of the ship, as a funciton
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        # Forces acting (replace zero vectors with suitable functions)
        f_rudder_v, f_rudder_r = self.rudder(rudder_angle)

        wind_force = self.get_wind_force()
        wave_force = np.array([0, 0, 0])
        ctrl_force = np.array([thrust_force, f_rudder_v, f_rudder_r])

        # assembling state vector
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.forward_speed - v_c[0]
        v_r = self.sideways_speed - v_c[1]

        # Kinetic equation
        m_inv = np.linalg.inv(self.mass_matrix())
        dx = np.dot(
            m_inv,
            -np.dot(self.coriolis_matrix(), vel)
            - np.dot(self.coriolis_added_mass_matrix(u_r=u_r, v_r=v_r), vel - v_c)
            - np.dot(self.linear_damping_matrix() + self.non_linear_damping_matrix(), vel - v_c)
            + wind_force + wave_force + ctrl_force)
        self.d_forward_speed = dx[0]
        self.d_sideways_speed = dx[1]
        self.d_yaw_rate = dx[2]

    def rudder(self, delta):
        ''' This method takes in the rudder angle and returns
            the force i sway and yaw generated by the rudder.

            args:
            delta (float): The rudder angle in radians

            returs:
            v_force (float): The force in sway-direction generated by the rudder
            r_force (float): The yaw-torque generated by the rudder
        '''
        u_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)[0]
        v_force = -self.ship_machinery_model.c_rudder_v * delta * (self.forward_speed - u_c)
        r_force = -self.ship_machinery_model.c_rudder_r * delta * (self.forward_speed - u_c)
        return v_force, r_force

    def update_differentials(self, engine_throttle=None, rudder_angle=None, *args, **kwargs):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        self.ship_machinery_model.update_shaft_equation(engine_throttle)
        self.three_dof_kinetics(thrust_force=self.ship_machinery_model.thrust(), rudder_angle=rudder_angle)

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.north = self.int.integrate(x=self.north, dx=self.d_north)
        self.east = self.int.integrate(x=self.east, dx=self.d_east)
        self.yaw_angle = self.int.integrate(x=self.yaw_angle, dx=self.d_yaw)
        self.forward_speed = self.int.integrate(x=self.forward_speed, dx=self.d_forward_speed)
        self.sideways_speed = self.int.integrate(x=self.sideways_speed, dx=self.d_sideways_speed)
        self.yaw_rate = self.int.integrate(x=self.yaw_rate, dx=self.d_yaw_rate)
        self.ship_machinery_model.integrate_differentials()

    def store_simulation_data(self, load_perc):
        load_perc_me, load_perc_hsg = self.ship_machinery_model.load_perc(load_perc)
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.north)
        self.simulation_results['east position [m]'].append(self.east)
        self.simulation_results['yaw angle [deg]'].append(self.yaw_angle * 180 / np.pi)
        self.simulation_results['forward speed[m/s]'].append(self.forward_speed)
        self.simulation_results['sideways speed [m/s]'].append(self.sideways_speed)
        self.simulation_results['yaw rate [deg/sec]'].append(self.yaw_rate * 180 / np.pi)
        self.simulation_results['propeller shaft speed [rpm]'].append(self.ship_machinery_model.omega * 30 / np.pi)
        self.simulation_results['commanded load fraction me [-]'].append(load_perc_me)
        self.simulation_results['commanded load fraction hsg [-]'].append(load_perc_hsg)

        load_data = self.ship_machinery_model.mode.distribute_load(
            load_perc=load_perc, hotel_load=self.ship_machinery_model.hotel_load
        )
        self.simulation_results['power me [kw]'].append(load_data.load_on_main_engine / 1000)
        self.simulation_results['available power me [kw]'].append(
            self.ship_machinery_model.mode.main_engine_capacity / 1000
        )
        self.simulation_results['power electrical [kw]'].append(load_data.load_on_electrical / 1000)
        self.simulation_results['available power electrical [kw]'].append(
            self.ship_machinery_model.mode.electrical_capacity / 1000
        )
        self.simulation_results['power [kw]'].append((load_data.load_on_electrical
                                                      + load_data.load_on_main_engine) / 1000)
        self.simulation_results['propulsion power [kw]'].append(
            (load_perc * self.ship_machinery_model.mode.available_propulsion_power) / 1000)
        rate_me, rate_hsg, cons_me, cons_hsg, cons = self.ship_machinery_model.fuel_consumption(load_perc)
        self.simulation_results['fuel rate me [kg/s]'].append(rate_me)
        self.simulation_results['fuel rate hsg [kg/s]'].append(rate_hsg)
        self.simulation_results['fuel rate [kg/s]'].append(rate_me + rate_hsg)
        self.simulation_results['fuel consumption me [kg]'].append(cons_me)
        self.simulation_results['fuel consumption hsg [kg]'].append(cons_hsg)
        self.simulation_results['fuel consumption [kg]'].append(cons)
        self.simulation_results['motor torque [Nm]'].append(self.ship_machinery_model.main_engine_torque(load_perc))
        self.simulation_results['thrust force [kN]'].append(self.ship_machinery_model.thrust() / 1000)


class ShipModelSimplifiedPropulsion(BaseShipModel):
    ''' Creates a ship model object that can be used to simulate a ship in transit

        The ships model is propelled by a single propeller and steered by a rudder.
        The propeller is powered by either the main engine, an auxiliary motor
        referred to as the hybrid shaft generator, or both. The model contains the
        following states:
        - North position of ship
        - East position of ship
        - Yaw angle (relative to north axis)
        - Surge velocity (forward)
        - Sway velocity (sideways)
        - Yaw rate
        - Propeller shaft speed

        Simulation results are stored in the instance variable simulation_results
    '''

    def __init__(self, ship_config: ShipConfiguration,
                 machinery_config: SimplifiedPropulsionMachinerySystemConfiguration,
                 environment_config: EnvironmentConfiguration,
                 simulation_config: SimulationConfiguration):

        super().__init__(ship_config, simulation_config, environment_config)

        # Machinery system params
        rudder_config = RudderConfiguration(
            rudder_angle_to_sway_force_coefficient=machinery_config.rudder_angle_to_sway_force_coefficient,
            rudder_angle_to_yaw_force_coefficient=machinery_config.rudder_angle_to_yaw_force_coefficient,
            max_rudder_angle_degrees=machinery_config.max_rudder_angle_degrees
        )
        self.ship_machinery_model = SimplifiedMachineryModel(
            machinery_config=machinery_config,
            time_step=simulation_config.integration_step,
            initial_thrust_force=0
        )

        self.simulation_results = defaultdict(list)

    def three_dof_kinetics(self, thrust_force=None, load_percentage=None, rudder_angle=None, *args, **kwargs):
        ''' Calculates accelerations of the ship, as a funciton
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        # Forces acting (replace zero vectors with suitable functions)
        f_rudder_v, f_rudder_r = self.rudder(rudder_angle)

        wind_force = self.get_wind_force()
        wave_force = np.array([0, 0, 0])

        ctrl_force = np.array(
            [thrust_force, f_rudder_v, f_rudder_r]
        )

        # assembling state vector
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.forward_speed - v_c[0]
        v_r = self.sideways_speed - v_c[1]

        # Kinetic equation
        m_inv = np.linalg.inv(self.mass_matrix())
        dx = np.dot(
            m_inv,
            -np.dot(self.coriolis_matrix(), vel)
            - np.dot(self.coriolis_added_mass_matrix(u_r=u_r, v_r=v_r), vel - v_c)
            - np.dot(self.linear_damping_matrix() + self.non_linear_damping_matrix(), vel - v_c)
            + wind_force + wave_force + ctrl_force)
        self.d_forward_speed = dx[0]
        self.d_sideways_speed = dx[1]
        self.d_yaw_rate = dx[2]

    def rudder(self, delta):
        ''' This method takes in the rudder angle and returns
            the force i sway and yaw generated by the rudder.

            args:
            delta (float): The rudder angle in radians

            returs:
            v_force (float): The force in sway-direction generated by the rudder
            r_force (float): The yaw-torque generated by the rudder
        '''
        u_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)[0]
        v_force = -self.ship_machinery_model.c_rudder_v * delta * (self.forward_speed - u_c)
        r_force = -self.ship_machinery_model.c_rudder_r * delta * (self.forward_speed - u_c)
        return v_force, r_force

    def update_differentials(self, engine_throttle=None, rudder_angle=None, *args, **kwargs):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        self.ship_machinery_model.update_thrust_force(load_perc=engine_throttle)
        self.three_dof_kinetics(thrust_force=self.ship_machinery_model.thrust, rudder_angle=rudder_angle)

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.north = self.int.integrate(x=self.north, dx=self.d_north)
        self.east = self.int.integrate(x=self.east, dx=self.d_east)
        self.yaw_angle = self.int.integrate(x=self.yaw_angle, dx=self.d_yaw)
        self.forward_speed = self.int.integrate(x=self.forward_speed, dx=self.d_forward_speed)
        self.sideways_speed = self.int.integrate(x=self.sideways_speed, dx=self.d_sideways_speed)
        self.yaw_rate = self.int.integrate(x=self.yaw_rate, dx=self.d_yaw_rate)
        self.ship_machinery_model.integrate_differentials()

    def store_simulation_data(self, load_perc):
        load_perc_me, load_perc_hsg = self.ship_machinery_model.load_perc(load_perc)
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.north)
        self.simulation_results['east position [m]'].append(self.east)
        self.simulation_results['yaw angle [deg]'].append(self.yaw_angle * 180 / np.pi)
        self.simulation_results['forward speed[m/s]'].append(self.forward_speed)
        self.simulation_results['sideways speed [m/s]'].append(self.sideways_speed)
        self.simulation_results['yaw rate [deg/sec]'].append(self.yaw_rate * 180 / np.pi)
        self.simulation_results['commanded load fraction [-]'].append(load_perc)
        self.simulation_results['commanded load fraction me [-]'].append(load_perc_me)
        self.simulation_results['commanded load fraction hsg [-]'].append(load_perc_hsg)

        load_data = self.ship_machinery_model.mode.distribute_load(
            load_perc=load_perc, hotel_load=self.ship_machinery_model.hotel_load
        )
        self.simulation_results['power me [kw]'].append(load_data.load_on_main_engine / 1000)
        self.simulation_results['available power me [kw]'].append(
            self.ship_machinery_model.mode.main_engine_capacity / 1000
        )
        self.simulation_results['power electrical [kw]'].append(load_data.load_on_electrical / 1000)
        self.simulation_results['available power electrical [kw]'].append(
            self.ship_machinery_model.mode.electrical_capacity / 1000
        )
        self.simulation_results['power [kw]'].append((load_data.load_on_electrical
                                                      + load_data.load_on_main_engine) / 1000)
        self.simulation_results['propulsion power [kw]'].append(
            (load_perc * self.ship_machinery_model.mode.available_propulsion_power) / 1000
        )

        rate_me, rate_hsg, cons_me, cons_hsg, cons = self.ship_machinery_model.fuel_consumption(load_perc)
        self.simulation_results['fuel rate me [kg/s]'].append(rate_me)
        self.simulation_results['fuel rate hsg [kg/s]'].append(rate_hsg)
        self.simulation_results['fuel rate [kg/s]'].append(rate_me + rate_hsg)
        self.simulation_results['fuel consumption me [kg]'].append(cons_me)
        self.simulation_results['fuel consumption hsg [kg]'].append(cons_hsg)
        self.simulation_results['fuel consumption [kg]'].append(cons)
        self.ship_machinery_model.fuel_me.append(cons_me)
        self.ship_machinery_model.fuel_hsg.append(cons_hsg)
        self.ship_machinery_model.fuel.append(cons)
        self.simulation_results['thrust force [kN]'].append(self.ship_machinery_model.thrust / 1000)


class ShipModelWithoutPropulsion(BaseShipModel):
    ''' Creates a ship model object that can be used to simulate a ship drifting freely

        The model contains the following states:
        - North position of ship
        - East position of ship
        - Yaw angle (relative to north axis)
        - Surge velocity (forward)
        - Sway velocity (sideways)
        - Yaw rate

        Simulation results are stored in the instance variable simulation_results
    '''

    def __init__(self, ship_config: ShipConfiguration, environment_config: EnvironmentConfiguration,
                 simulation_config: SimulationConfiguration):

        super().__init__(ship_config, simulation_config, environment_config)
        self.simulation_results = defaultdict(list)

    def store_simulation_data(self):
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.north)
        self.simulation_results['east position [m]'].append(self.east)
        self.simulation_results['yaw angle [deg]'].append(self.yaw_angle * 180 / np.pi)
        self.simulation_results['forward speed[m/s]'].append(self.forward_speed)
        self.simulation_results['sideways speed [m/s]'].append(self.sideways_speed)
        self.simulation_results['yaw rate [deg/sec]'].append(self.yaw_rate * 180 / np.pi)
        self.simulation_results['wind speed [m/sec]'].append(self.wind_speed)


class PiController:
    def __init__(self, kp: float, ki: float, time_step: float, initial_integral_error=0):
        self.kp = kp
        self.ki = ki
        self.error_i = initial_integral_error
        self.time_step = time_step

    def pi_ctrl(self, setpoint, measurement, *args):
        ''' Uses a proportional-integral control law to calculate a control
            output. The optional argument is an 2x1 array and will specify lower
            and upper limit for error integration [lower, upper]
        '''
        error = setpoint - measurement
        error_i = self.error_i + error * self.time_step
        if args:
            error_i = self.sat(error_i, args[0], args[1])
        self.error_i = error_i
        return error * self.kp + error_i * self.ki

    @staticmethod
    def sat(val, low, hi):
        ''' Saturate the input val such that it remains
        between "low" and "hi"
        '''
        return max(low, min(val, hi))


class PidController:
    def __init__(self, kp: float, kd: float, ki: float, time_step: float):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.error_i = 0
        self.prev_error = 0
        self.time_step = time_step

    def pid_ctrl(self, setpoint, measurement, *args):
        ''' Uses a proportional-derivative-integral control law to calculate
            a control output. The optional argument is a 2x1 array and will
            specify lower and upper [lower, upper] limit for error integration
        '''
        error = setpoint - measurement
        d_error = (error - self.prev_error) / self.time_step
        error_i = self.error_i + error * self.time_step
        if args:
            error_i = self.sat(error_i, args[0], args[1])
        self.prev_error = error
        self.error_i = error_i
        return error * self.kp + d_error * self.kd + error_i * self.ki

    @staticmethod
    def sat(val, low, hi):
        ''' Saturate the input val such that it remains
        between "low" and "hi"
        '''
        return max(low, min(val, hi))


class ThrottleControllerGains(NamedTuple):
    kp_ship_speed: float
    ki_ship_speed: float
    kp_shaft_speed: float
    ki_shaft_speed: float


class EngineThrottleFromSpeedSetPoint:
    """
    Calculates throttle setpoint for power generation based on the ship´s speed, the propeller shaft speed
    and the desires ship speed.
    """

    def __init__(
            self,
            gains: ThrottleControllerGains,
            max_shaft_speed: float,
            time_step: float,
            initial_shaft_speed_integral_error: float
    ):
        self.ship_speed_controller = PiController(
            kp=gains.kp_ship_speed, ki=gains.ki_ship_speed, time_step=time_step
        )
        self.shaft_speed_controller = PiController(
            kp=gains.kp_shaft_speed,
            ki=gains.ki_shaft_speed,
            time_step=time_step,
            initial_integral_error=initial_shaft_speed_integral_error
        )
        self.max_shaft_speed = max_shaft_speed

    def throttle(self, speed_set_point, measured_speed, measured_shaft_speed):
        desired_shaft_speed = self.ship_speed_controller.pi_ctrl(setpoint=speed_set_point, measurement=measured_speed)
        desired_shaft_speed = self.ship_speed_controller.sat(val=desired_shaft_speed, low=0, hi=self.max_shaft_speed)
        throttle = self.shaft_speed_controller.pi_ctrl(setpoint=desired_shaft_speed, measurement=measured_shaft_speed)
        return self.shaft_speed_controller.sat(val=throttle, low=0, hi=1.1)


class HeadingControllerGains(NamedTuple):
    kp: float
    kd: float
    ki: float


class HeadingByReferenceController:
    def __init__(self, gains: HeadingControllerGains, time_step, max_rudder_angle):
        self.ship_heading_controller = PidController(kp=gains.kp, kd=gains.kd, ki=gains.ki, time_step=time_step)
        self.max_rudder_angle = max_rudder_angle

    def rudder_angle_from_heading_setpoint(self, heading_ref: float, measured_heading: float):
        ''' This method finds a suitable rudder angle for the ship to
            sail with the heading specified by "heading_ref" by using
            PID-controller. The rudder angle is saturated according to
            |self.rudder_ang_max|. The mathod should be called from within
            simulation loop if the user want the ship to follow a specified
            heading reference signal.
        '''
        rudder_angle = -self.ship_heading_controller.pid_ctrl(setpoint=heading_ref, measurement=measured_heading)
        return self.ship_heading_controller.sat(rudder_angle, -self.max_rudder_angle, self.max_rudder_angle)


class HeadingByRouteController:
    def __init__(self, route_name, heading_controller_gains: HeadingControllerGains, time_step: float, max_rudder_angle: float):
        self.heading_controller = HeadingByReferenceController(
            gains=heading_controller_gains, time_step=time_step, max_rudder_angle=max_rudder_angle
        )
        self.navigate = NavigationSystem(route_name)
        self.next_wpt = 1
        self.prev_wpt = 0

    def rudder_angle_from_route(self, north_position, east_position, heading):
        ''' This method finds a suitable rudder angle for the ship to follow
            a predefined route specified in the "navigate"-instantiation of the
            "NavigationSystem"-class.
        '''
        self.next_wpt, self.prev_wpt = self.navigate.next_wpt(self.next_wpt, north_position, east_position)
        psi_d = self.navigate.los_guidance(self.next_wpt, north_position, east_position)
        return self.heading_controller.rudder_angle_from_heading_setpoint(heading_ref=psi_d, measured_heading=heading)


class EulerInt:
    ''' Provides methods relevant for using the
        Euler method to integrate an ODE.

        Usage:

        int=EulerInt()
        while int.time <= int.sim_time:
            dx = f(x)
            int.integrate(x,dx)
            int.next_time
    '''

    def __init__(self):
        self.dt = 0.01
        self.sim_time = 10
        self.time = 0.0
        self.times = []
        self.global_times = []

    def set_dt(self, val):
        ''' Sets the integrator step length
        '''
        self.dt = val

    def set_sim_time(self, val):
        ''' Sets the upper time integration limit
        '''
        self.sim_time = val

    def set_time(self, val):
        ''' Sets the time variable to "val"
        '''
        self.time = val

    def next_time(self, time_shift=0):
        ''' Increment the time variable to the next time instance
            and store in an array
        '''
        self.time = self.time + self.dt
        self.times.append(self.time)
        self.global_times.append(self.time + time_shift)

    def integrate(self, x, dx):
        ''' Performs the Euler integration step
        '''
        return x + dx * self.dt


class ShipDraw:
    ''' This class is used to calculate the coordinates of each
        corner of 80 meter long and 20meter wide ship seen from above,
        and rotate and translate the coordinates according to
        the ship heading and position
    '''

    def __init__(self):
        self.l = 80.0
        self.b = 20.0

    def local_coords(self):
        ''' Here the ship is pointing along the local
            x-axix with its center of origin (midship)
            at the origin
            1 denotes the left back corner
            2 denotes the left starting point of bow curvatiure
            3 denotes the bow
            4 the right starting point of the bow curve
            5 the right back cornier
        '''
        x1, y1 = -self.l / 2, -self.b / 2
        x2, y2 = self.l / 4, -self.b / 2
        x3, y3 = self.l / 2, 0.0
        x4, y4 = self.l / 4, self.b / 2
        x5, y5 = -self.l / 2, self.b / 2

        x = np.array([x1, x2, x3, x4, x5, x1])
        y = np.array([y1, y2, y3, y4, y5, y1])
        return x, y

    def rotate_coords(self, x, y, psi):
        ''' Rotates the ship an angle psi
        '''
        x_t = np.cos(psi) * x - np.sin(psi) * y
        y_t = np.sin(psi) * x + np.cos(psi) * y
        return x_t, y_t

    def translate_coords(self, x_ned, y_ned, north, east):
        ''' Takes in coordinates of the corners of the ship (in the ned-frame)
            and translates them in the north and east direction according to
            "north" and "east"
        '''
        x_t = x_ned + north
        y_t = y_ned + east
        return x_t, y_t


class NavigationSystem:
    ''' This class provides a way of following a predifined route using
        line-og-sight (LOS) guidance law. The path to the textfile where
        the route is specified is given as an argument when calling the
        class. The route text file is formated as follows:
        x1 y1
        x2 y2
        ...
        where (x1,y1) are the coordinates to the first waypoint,
        (x2,y2) to the second, etc.
    '''

    def __init__(self, route):
        self.load_waypoints(route)
        self.ra = 600  # Radius of acceptance for waypoints
        self.r = 450  # Lookahead distance
        self.ki = 0.000007
        self.e_ct_int = 0

    def load_waypoints(self, route):
        ''' Reads the file containing the route and stores it as an
            array of north positions and an array of east positions
        '''
        self.data = np.loadtxt(route)
        self.north = []
        self.east = []
        for i in range(0, (int(np.size(self.data) / 2))):
            self.north.append(self.data[i][0])
            self.east.append(self.data[i][1])

    def next_wpt(self, k, N, E):
        ''' Returns the index of the next and current waypoint. The method, if
            called at each time step, will detect when the ship has arrived
            close enough to a waypoint, to proceed ot the next waypoint. Example
            of usage in the method "rudderang_from_route()" from the ShipDyn-class.
        '''
        if (self.north[k] - N) ** 2 + (
                self.east[k] - E) ** 2 <= self.ra ** 2:  # Check that we are within circle of acceptance
            if len(self.north) > k + 1:  # If number of waypoints are greater than current waypoint index
                return k + 1, k  # Then move on to next waypoint and let current become previous
            else:
                return k, k  # At the end of the route, let the next wpt also be the previous wpt
        else:
            return k, k - 1

    def los_guidance(self, k, x, y):
        ''' Returns the desired heading (i.e. reference signal to
            a ship heading controller). The parameter "k" is the
            index of the next waypoint.
        '''
        dx = self.north[k] - self.north[k - 1]
        dy = self.east[k] - self.east[k - 1]
        alpha_k = math.atan2(dy, dx)
        e_ct = -(x - self.north[k - 1]) * math.sin(alpha_k) + (y - self.east[k - 1]) * math.cos(alpha_k)
        if e_ct ** 2 >= self.r ** 2:
            e_ct = 0.99 * self.r
        delta = math.sqrt(self.r ** 2 - e_ct ** 2)
        self.e_ct_int += e_ct
        chi_r = math.atan(-e_ct / delta + self.ki * self.e_ct_int)
        return alpha_k + chi_r


class StaticObstacle:
    ''' This class is used to define a static obstacle. It can only make
        circular obstacles. The class is instantiated with the following
        input paramters:
        - n_pos: The north coordinate of the center of the obstacle.
        - e_pos: The east coordinate of the center of the obstacle.
        - radius: The radius of the obstacle.
    '''

    def __init__(self, n_pos, e_pos, radius):
        self.n = n_pos
        self.e = e_pos
        self.r = radius

    def distance(self, n_ship, e_ship):
        ''' Returns the distance from a ship with coordinates (north, east)=
            (n_ship, e_ship), to the closest point on the perifery of the
            circular obstacle.
        '''
        rad_2 = (n_ship - self.n) ** 2 + (e_ship - self.e) ** 2
        rad = np.sqrt(abs(rad_2))
        return rad - self.r

    def plot_obst(self, ax):
        ''' This method can be used to plot the obstacle in a
            map-view.
        '''
        # ax = plt.gca()
        ax.add_patch(plt.Circle((self.e, self.n), radius=self.r, fill=True, color='grey'))
