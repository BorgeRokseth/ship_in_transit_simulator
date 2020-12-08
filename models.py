""" This module provides classes that that can be used to setup and
    run simulation models of a ship in transit.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from typing import NamedTuple, List
import random


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
    route_name: str
    initial_north_position_m: float
    initial_east_position_m: float
    initial_yaw_angle_rad: float
    initial_forward_speed_m_per_s: float
    initial_sideways_speed_m_per_s: float
    initial_yaw_rate_rad_per_s: float
    initial_propeller_shaft_speed_rad_per_s: float
    machinery_system_operating_mode: int
    integration_step: float
    simulation_time: float


class SimplifiedPropulsionSimulationConfiguration(NamedTuple):
    route_name: str
    initial_north_position_m: float
    initial_east_position_m: float
    initial_yaw_angle_rad: float
    initial_forward_speed_m_per_s: float
    initial_sideways_speed_m_per_s: float
    initial_yaw_rate_rad_per_s: float
    initial_thrust_force: float
    machinery_system_operating_mode: int
    integration_step: float
    simulation_time: float


class DriftSimulationConfiguration(NamedTuple):
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

class IcebergConfiguration(NamedTuple):
    mass_tonnage: float
    coefficient_of_deadweight_to_displacement: float
    waterlinelength_of_iceberg: float
    width_of_iceberg: float
    height_of_iceberg: float
    shape_of_iceberg: str
    size_of_iceberg: str
    added_mass_coefficient_in_surge: float
    added_mass_coefficient_in_sway: float
    added_mass_coefficient_in_yaw: float
    mass_over_linear_friction_coefficient_in_surge: float
    mass_over_linear_friction_coefficient_in_sway: float
    mass_over_linear_friction_coefficient_in_yaw: float
    nonlinear_friction_coefficient__in_surge: float
    nonlinear_friction_coefficient__in_sway: float
    nonlinear_friction_coefficient__in_yaw: float
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


class MachinerySystemConfiguration(NamedTuple):
    hotel_load: float
    machinery_modes: MachineryModes
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


class SimplifiedPropulsionMachinerySystemConfiguration(NamedTuple):
    hotel_load: float
    machinery_modes: MachineryModes
    thrust_force_dynamic_time_constant: float
    rudder_angle_to_sway_force_coefficient: float
    rudder_angle_to_yaw_force_coefficient: float
    max_rudder_angle_degrees: float


class ShipModel:
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
                 machinery_config: MachinerySystemConfiguration,
                 environment_config: EnvironmentConfiguration,
                 simulation_config: SimulationConfiguration):
        route_name = simulation_config.route_name
        if route_name != 'none':
            # Route following
            self.navigate = NavigationSystem(route_name)
            self.next_wpt = 1
            self.prev_wpt = 0

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


        # Machinery system params
        self.machinery_modes = machinery_config.machinery_modes
        self.hotel_load = machinery_config.hotel_load  # 200000  # 0.2 MW
        self.update_available_propulsion_power()
        mode = simulation_config.machinery_system_operating_mode
        self.mode = self.machinery_modes.list_of_modes[mode]

        #self.p_rated_me = machinery_config.mcr_main_engine  # 2160000  # 2.16 MW
        #self.p_rated_hsg = machinery_config.mcr_hybrid_shaft_generator  # 590000  # 0.59 MW
        self.w_rated_me = machinery_config.rated_speed_main_engine_rpm * np.pi / 30  # 1000 * np.pi / 30  # rated speed
        self.d_me = machinery_config.linear_friction_main_engine  # 68.0  # linear friction for main engine speed
        self.d_hsg = machinery_config.linear_friction_hybrid_shaft_generator  # 57.0  # linear friction for HSG speed
        self.r_me = machinery_config.gear_ratio_between_main_engine_and_propeller  # 0.6  # gear ratio between main engine and propeller
        self.r_hsg = machinery_config.gear_ratio_between_hybrid_shaft_generator_and_propeller  # 0.6  # gear ratio between main engine and propeller
        self.jp = machinery_config.propeller_inertia  # 6000  # propeller inertia
        self.kp = machinery_config.propeller_speed_to_torque_coefficient  # 7.5  # constant relating omega to torque
        self.dp = machinery_config.propeller_diameter  # 3.1  # propeller diameter
        self.kt = machinery_config.propeller_speed_to_thrust_force_coefficient  # 1.7  # constant relating omega to thrust force
        self.shaft_speed_max = 1.1 * self.w_rated_me * self.r_me  # Used for saturation of power sources

        self.c_rudder_v = machinery_config.rudder_angle_to_sway_force_coefficient  # 50000.0  # tuning param for simplified rudder response model
        self.c_rudder_r = machinery_config.rudder_angle_to_yaw_force_coefficient  # 500000.0  # tuning param for simplified rudder response model
        self.rudder_ang_max = machinery_config.max_rudder_angle_degrees * np.pi / 180  # 30 * np.pi / 180  # Maximal rudder angle deflection (both ways)

        # Environmental conditions
        self.vel_c = np.array([environment_config.current_velocity_component_from_north,
                               environment_config.current_velocity_component_from_east,
                               0.0])
        self.wind_dir = environment_config.wind_direction
        self.wind_speed = environment_config.wind_speed

        # Operational parameters used to calculate loading percent on each power source
        self.p_rel_rated_hsg = 0.0
        self.p_rel_rated_me = 0.0

        # Configure machinery system according to self.mso
        #self.mso_mode = simulation_config.machinery_system_operating_mode
        #self.mode_selector(machinery_config.mcr_main_engine,
        #                   machinery_config.mcr_hybrid_shaft_generator)

        # Initial states (can be altered using self.set_state_vector(x))
        self.n = simulation_config.initial_north_position_m
        self.e = simulation_config.initial_east_position_m
        self.psi = simulation_config.initial_yaw_angle_rad
        self.u = simulation_config.initial_forward_speed_m_per_s
        self.v = simulation_config.initial_sideways_speed_m_per_s
        self.r = simulation_config.initial_yaw_rate_rad_per_s
        self.omega = simulation_config.initial_propeller_shaft_speed_rad_per_s
        self.x = self.update_state_vector()
        self.states = np.empty(7)

        # Differentials
        self.d_n = self.d_e = self.d_psi = 0
        self.d_u = self.d_v = self.d_r = 0
        self.d_omega = 0

        # Set up ship control systems
        self.initialize_shaft_speed_controller(kp=0.05, ki=0.005)
        self.initialize_ship_speed_controller(kp=7, ki=0.13)
        self.initialize_ship_heading_controller(kp=4, kd=90, ki=0.005)
        self.initialize_heading_filter(kp=0.5, kd=10, t=5000)

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(simulation_config.integration_step)
        self.int.set_sim_time(simulation_config.simulation_time)

        # Instantiate ship draw plotting
        self.drw = ShipDraw()  # Instantiate the ship drawing class
        self.ship_drawings = [[], []]  # Arrays for storing ship drawing data

        # Fuel
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

        # Wind effect on ship
        self.rho_a = 1.2
        self.h_f = 8.0  # mean height above water seen from the front
        self.h_s = 8.0  # mean height above water seen from the side
        self.proj_area_f = self.w_ship * self.h_f  # Projected are from the front
        self.proj_area_l = self.l_ship * self.h_s  # Projected area from the side
        self.cx = 0.5
        self.cy = 0.7
        self.cn = 0.08

        # Fuel consumption function parameters
        self.a_me = 128.89
        self.b_me = -168.93
        self.c_me = 246.76

        self.a_dg = 180.71
        self.b_dg = -289.90
        self.c_dg = 324.90

        self.simulation_results = defaultdict(list)

    def update_available_propulsion_power(self):
        for mode in self.machinery_modes.list_of_modes:
            mode.update_available_propulsion_power(self.hotel_load)

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

    def mode_selector(self, mode: int):
        self.mode = self.machinery_modes.list_of_modes[mode]

    def spec_fuel_cons_me(self, load_perc):
        """ Calculate fuel consumption rate for the main engine.

            Args:
                load_perc (float): The fraction of the mcr load on the ME
            Returns:
                Number of kilograms of fuel per second used by ME
        """
        rate = self.a_me * load_perc ** 2 + self.b_me * load_perc + self.c_me
        return rate / 3.6e9

    def spec_fuel_cons_dg(self, load_perc):
        """ Calculate fuel consumption rate for a diesel generator.

            Args:
                load_perc (float): The fraction of the mcr load on the DG
            Returns:
                Number of kilograms of fuel per second used by DG
        """
        rate = self.a_dg * load_perc ** 2 + self.b_dg * load_perc + self.c_dg
        return rate / 3.6e9

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
        '''
        if self.mso_mode == 1:
            load_me = load_perc * self.p_rated_me + self.hotel_load
            load_perc_me = load_me / self.p_rated_me
            rate_me = load_me * self.spec_fuel_cons_me(load_perc_me)
            rate_hsg = 0.0
        elif self.mso_mode == 2:
            load_me = load_perc * self.p_rated_me
            load_perc_me = load_me / self.p_rated_me
            load_hsg = self.hotel_load
            load_perc_hsg = load_hsg / self.p_rated_hsg
            rate_me = load_me * self.spec_fuel_cons_me(load_perc_me)
            rate_hsg = load_hsg * self.spec_fuel_cons_dg(load_perc_hsg)
        elif self.mso_mode == 3:
            load_hsg = (load_perc * self.p_rated_hsg + self.hotel_load)
            load_perc_hsg = load_hsg / self.p_rated_hsg
            rate_me = 0.0
            rate_hsg = load_hsg * self.spec_fuel_cons_dg(load_perc_hsg)
        '''
        load_data = self.mode.distribute_load(load_perc=load_perc, hotel_load=self.hotel_load)
        if load_data.load_on_main_engine == 0:
            rate_me = 0
        else:
            rate_me = load_data.load_on_main_engine \
                      * self.spec_fuel_cons_me(load_data.load_percentage_on_main_engine)

        if load_data.load_percentage_on_electrical == 0:
            rate_electrical = 0
        else:
            rate_electrical = load_data.load_on_electrical \
                              * self.spec_fuel_cons_dg(load_data.load_percentage_on_electrical)

        self.fuel_cons_me = self.fuel_cons_me + rate_me * self.int.dt
        self.fuel_cons_electrical = self.fuel_cons_electrical + rate_electrical * self.int.dt
        self.fuel_cons = self.fuel_cons + (rate_me + rate_electrical) * self.int.dt
        return rate_me, rate_electrical, self.fuel_cons_me, self.fuel_cons_electrical, self.fuel_cons

    def get_wind_force(self):
        ''' This method calculates the forces due to the relative
            wind speed, acting on teh ship in surge, sway and yaw
            direction.

            :return: Wind force acting in surge, sway and yaw
        '''
        uw = self.wind_speed * np.cos(self.wind_dir - self.psi)
        vw = self.wind_speed * np.sin(self.wind_dir - self.psi)
        u_rw = uw - self.u
        v_rw = vw - self.v
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

    def update_state_vector(self):
        ''' Update the state vector according to the individual state values
        '''
        return np.array([self.n, self.e, self.psi, self.u, self.v, self.r, self.omega])

    def set_north_pos(self, val):
        ''' Set the north position of the ship and update the state vector
        '''
        self.n = val
        self.x = self.update_state_vector()

    def set_east_pos(self, val):
        ''' Set the east position of the ship and update the state vector
        '''
        self.e = val
        self.x = self.update_state_vector()

    def set_yaw_angle(self, val):
        ''' Set the yaw angle of the ship and update the state vector
        '''
        self.psi = val
        self.x = self.update_state_vector()

    def set_surge_speed(self, val):
        ''' Set the surge speed of the ship and update the state vector
        '''
        self.u = val
        self.x = self.update_state_vector()

    def set_sway_speed(self, val):
        ''' Set the sway speed of the ship and update the state vector
        '''
        self.v = val
        self.x = self.update_state_vector()

    def set_yaw_rate(self, val):
        ''' Set the yaw rate of the ship and update the state vector
        '''
        self.r = val
        self.x = self.update_state_vector()

    def set_shaft_speed(self, val):
        ''' Set the propeller shaft speed and update the state vector
        '''
        self.omega = val
        self.x = self.update_state_vector()

    def initialize_shaft_speed_controller(self, kp, ki):
        ''' This method sets up and configures the shaft speed
            controller of the ship
        '''
        self.shaft_speed_controller = ControllerLib()
        self.shaft_speed_controller.set_kp(kp)
        self.shaft_speed_controller.set_ki(ki)

    def initialize_ship_speed_controller(self, kp, ki):
        ''' This method sets up and configures the ship speed
            controller.
        '''
        self.ship_speed_controller = ControllerLib()
        self.ship_speed_controller.set_kp(kp)
        self.ship_speed_controller.set_ki(ki)

    def initialize_ship_heading_controller(self, kp, kd, ki):
        ''' This method sets up and configures the ship heading
            controller.
        '''
        self.ship_heading_controller = ControllerLib()
        self.ship_heading_controller.set_kp(kp)
        self.ship_heading_controller.set_kd(-kd)
        self.ship_heading_controller.set_ki(ki)

    def initialize_heading_filter(self, kp, kd, t):
        ''' This method sets up and configures a low pass filter
            to smooth the hading setpoint signal for the ship
            heading controller.
        '''
        self.ship_heading_filter = ControllerLib()
        self.ship_heading_filter.set_kp(kp)
        self.ship_heading_filter.set_kd(kd)
        self.ship_heading_filter.set_T(t)

    def loadperc_from_speedref(self, speed_ref):
        ''' Calculates suitable machinery load percentage for the ship to
            track the speed reference signal. The shaft speed controller
            is used to calculate suitable shaft speed to follow the desired
            ship speed and suitable load percentage to follow the calculated
            shaft speed. The load percentage is the fraction of the produced
            power over the total power capacity in the current configuration.
        '''
        ref_shaft_speed = self.ship_speed_controller.pi_ctrl(speed_ref, self.u, self.int.dt, -550, 550)
        ref_shaft_speed = ControllerLib.sat(ref_shaft_speed, 0, self.shaft_speed_max)
        load_perc = self.shaft_speed_controller.pi_ctrl(ref_shaft_speed, self.omega, self.int.dt)
        load_perc = ControllerLib.sat(load_perc, 0, 1.1)
        return load_perc

    def rudderang_from_headingref(self, heading_ref):
        ''' This method finds a suitable rudder angle for the ship to
            sail with the heading specified by "heading_ref" by using
            PID-controller. The rudder angle is saturated according to
            |self.rudder_ang_max|. The mathod should be called from within
            simulation loop if the user want the ship to follow a specified
            heading reference signal.
        '''
        rudder_ang = self.ship_heading_controller.pid_ctrl(heading_ref, self.psi, self.int.dt)
        rudder_ang = ControllerLib.sat(rudder_ang, -self.rudder_ang_max, self.rudder_ang_max)
        return rudder_ang

    def rudderang_from_route(self):
        ''' This method finds a suitable rudder angle for the ship to follow
            a predefined route specified in the "navigate"-instantiation of the
            "NavigationSystem"-class.
        '''
        self.next_wpt, self.prev_wpt = self.navigate.next_wpt(self.next_wpt, self.n, self.e)
        psi_d = self.navigate.los_guidance(self.next_wpt, self.n, self.e)
        return self.rudderang_from_headingref(psi_d)

    def print_next_wpt(self, ship_id):
        ''' Prints a string with the ship identification (ship_id)
            and its next waypoint, if the next waypoint is specified
        '''
        if self.next_wpt != self.navigate.next_wpt(self.next_wpt, self.n, self.e)[0]:
            print('Current target waypoint for ' + ship_id + ' is: ' + str(self.next_wpt))

    def set_next_wpt(self, wpt):
        ''' Sets the next waypoint to "wpt", where "wpt" is the index
            of the waypoint refering to the list of waypoints making
            up the route specified in the instantiation "navigate" of
            the class "NavigationSystem"
        '''
        self.next_wpt = wpt

    def three_dof_kinematics(self):
        ''' Updates the time differientials of the north position, east
            position and yaw angle. Should be called in the simulation
            loop before the integration step.
        '''
        vel = np.array([self.u, self.v, self.r])
        dx = np.dot(self.rotation(), vel)
        self.d_n = dx[0]
        self.d_e = dx[1]
        self.d_psi = dx[2]

    def rotation(self):
        ''' Specifies the rotation matrix for rotations about the z-axis, such that
            "body-fixed coordinates" = rotation x "North-east-down-fixed coordinates" .
        '''
        return np.array([[np.cos(self.psi), -np.sin(self.psi), 0],
                         [np.sin(self.psi), np.cos(self.psi), 0],
                         [0, 0, 1]])

    def three_dof_kinetics(self, f_thrust, rudder_angle):
        ''' Calculates accelerations of the ship, as a funciton
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        # System matrices (did not include added mass yet)
        M_rb = np.array([[self.mass + self.x_du, 0, 0],
                         [0, self.mass + self.y_dv, self.mass * self.x_g],
                         [0, self.mass * self.x_g, self.i_z + self.n_dr]])
        C_rb = np.array([[0, 0, -self.mass * (self.x_g * self.r + self.v)],
                         [0, 0, self.mass * self.u],
                         [self.mass * (self.x_g * self.r + self.v), -self.mass * self.u, 0]])

        D = np.array([[self.mass / self.t_surge, 0, 0],
                      [0, self.mass / self.t_sway, 0],
                      [0, 0, self.i_z / self.t_yaw]])
        D2 = np.array([[self.ku * self.u, 0, 0],
                       [0, self.kv * self.v, 0],
                       [0, 0, self.kr * self.r]])

        # Forces acting (replace zero vectors with suitable functions)
        f_rudder_v, f_rudder_r = self.rudder(rudder_angle)

        F_wind = self.get_wind_force()
        F_waves = np.array([0, 0, 0])
        F_ctrl = np.array([f_thrust, f_rudder_v, f_rudder_r])

        # assembling state vector
        vel = np.array([self.u, self.v, self.r])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.u - v_c[0]
        v_r = self.v - v_c[1]

        C_a = np.array([[0, 0, self.y_dv * v_r],
                        [0, 0, -self.x_du * u_r],
                        [-self.y_dv * v_r, self.x_du * u_r, 0]])

        # Kinetic equation
        M_inv = np.linalg.inv(M_rb)
        dx = np.dot(M_inv, -np.dot(C_rb, vel) - np.dot(C_a, vel - v_c) - np.dot(D + D2, vel - v_c)
                    + F_wind + F_waves + F_ctrl)
        self.d_u = dx[0]
        self.d_v = dx[1]
        self.d_r = dx[2]

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
        v_force = -self.c_rudder_v * delta * (self.u - u_c)
        r_force = -self.c_rudder_r * delta * (self.u - u_c)
        return v_force, r_force

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
        # if self.omega >= 1 * np.pi / 30:
        #    return load_perc * self.p_rel_rated_me / self.omega
        # else:
        #    return 0
        #return min(load_perc * self.p_rel_rated_me / (self.omega + 0.1), self.p_rel_rated_me / 5 * np.pi / 30)
        return min(load_perc * self.mode.available_propulsion_power_main_engine / (self.omega + 0.1),
                   self.mode.available_propulsion_power_main_engine / 5 * np.pi / 30)

    def hsg_torque(self, load_perc):
        ''' Returns the torque of the HSG as a
            function of the load percentage parameter
        '''
        # if self.omega >= 100 * np.pi / 30:
        #    return load_perc * self.p_rel_rated_hsg / self.omega
        # else:
        #    return 0
        # return min(load_perc * self.p_rel_rated_hsg / (self.omega + 0.1), self.p_rel_rated_hsg / 5 * np.pi / 30)
        return min(load_perc * self.mode.available_propulsion_power_electrical / (self.omega + 0.1),
                   self.mode.available_propulsion_power_electrical / 5 * np.pi / 30)

    def update_differentials(self, load_perc, rudder_angle):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        self.shaft_eq(self.main_engine_torque(load_perc), self.hsg_torque(load_perc))
        self.three_dof_kinetics(self.thrust(), rudder_angle)

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.set_north_pos(self.int.integrate(self.n, self.d_n))
        self.set_east_pos(self.int.integrate(self.e, self.d_e))
        self.set_yaw_angle(self.int.integrate(self.psi, self.d_psi))
        self.set_surge_speed(self.int.integrate(self.u, self.d_u))
        self.set_sway_speed(self.int.integrate(self.v, self.d_v))
        self.set_yaw_rate(self.int.integrate(self.r, self.d_r))
        self.set_shaft_speed(self.int.integrate(self.omega, self.d_omega))

    def store_states(self):
        ''' Appends the current value of each state to an array. This
            is convenient when plotting. The method should be called within
            the simulation loop each time step. Then afterwars, an array
            containing for ecample the north-position for each time step
            is obtained as ...states[0]
        '''
        self.states[0].append(self.n)
        self.states[1].append(self.e)
        self.states[2].append(self.psi)
        self.states[3].append(self.u)
        self.states[4].append(self.v)
        self.states[5].append(self.r)
        self.states[6].append(self.omega)

    def ship_snap_shot(self):
        ''' This method is used to store a map-view snap shot of
            the ship at the given north-east position and heading.
            It uses the ShipDraw-class. To plot a map view of the
            n-th ship snap-shot, use:

            plot(ship_drawings[1][n], ship_drawings[0][n])
        '''
        x, y = self.drw.local_coords()
        x_ned, y_ned = self.drw.rotate_coords(x, y, self.psi)
        x_ned_trans, y_ned_trans = self.drw.translate_coords(x_ned, y_ned, self.n, self.e)
        self.ship_drawings[0].append(x_ned_trans)
        self.ship_drawings[1].append(y_ned_trans)

    def store_simulation_data(self, load_perc):
        load_perc_me, load_perc_hsg = self.load_perc(load_perc)
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.n)
        self.simulation_results['east position [m]'].append(self.e)
        self.simulation_results['yaw angle [deg]'].append(self.t_yaw * 180 / np.pi)
        self.simulation_results['forward speed[m/s]'].append(self.u)
        self.simulation_results['sideways speed [m/s]'].append(self.v)
        self.simulation_results['yaw rate [deg/sec]'].append(self.r * 180 / np.pi)
        self.simulation_results['propeller shaft speed [rpm]'].append(self.omega * 30 / np.pi)
        self.simulation_results['commanded load fraction [-]'].append(load_perc)
        self.simulation_results['commanded load fraction me [-]'].append(load_perc_me)
        self.simulation_results['commanded load fraction hsg [-]'].append(load_perc_hsg)

        load_data = self.mode.distribute_load(load_perc=load_perc, hotel_load=self.hotel_load)
        self.simulation_results['power me [kw]'].append(load_data.load_on_main_engine / 1000)
        self.simulation_results['available power me [kw]'].append(self.mode.main_engine_capacity / 1000)
        self.simulation_results['power electrical [kw]'].append(load_data.load_on_electrical / 1000)
        self.simulation_results['available power electrical [kw]'].append(self.mode.electrical_capacity / 1000)
        self.simulation_results['power [kw]'].append((load_data.load_on_electrical
                                                      + load_data.load_on_main_engine) / 1000)
        self.simulation_results['propulsion power [kw]'].append((load_perc
                                                                 * self.mode.available_propulsion_power) / 1000)

        rate_me, rate_hsg, cons_me, cons_hsg, cons = self.fuel_consumption(load_perc)
        self.simulation_results['fuel rate me [kg/s]'].append(rate_me)
        self.simulation_results['fuel rate hsg [kg/s]'].append(rate_hsg)
        self.simulation_results['fuel rate [kg/s]'].append(rate_me + rate_hsg)
        self.simulation_results['fuel consumption me [kg]'].append(cons_me)
        self.simulation_results['fuel consumption hsg [kg]'].append(cons_hsg)
        self.simulation_results['fuel consumption [kg]'].append(cons)
        self.simulation_results['motor torque [Nm]'].append(self.main_engine_torque(load_perc))
        self.simulation_results['thrust force [kN]'].append(self.thrust() / 1000)
        self.fuel_me.append(cons_me)
        self.fuel_hsg.append(cons_hsg)
        self.fuel.append(cons)


class ShipModelSimplifiedPropulsion:
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
                 simulation_config: SimplifiedPropulsionSimulationConfiguration):
        route_name = simulation_config.route_name
        if route_name != 'none':
            # Route following
            self.navigate = NavigationSystem(route_name)
            self.next_wpt = 1
            self.prev_wpt = 0

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

        # Machinery system params
        self.machinery_modes = machinery_config.machinery_modes
        self.hotel_load = machinery_config.hotel_load  # 200000  # 0.2 MW
        self.update_available_propulsion_power()
        mode = simulation_config.machinery_system_operating_mode
        self.mode = self.machinery_modes.list_of_modes[mode]

        self.thrust = simulation_config.initial_thrust_force
        self.d_thrust = 0
        self.k_thrust = 2160 / 790
        self.thrust_time_constant = machinery_config.thrust_force_dynamic_time_constant

        self.c_rudder_v = machinery_config.rudder_angle_to_sway_force_coefficient
        self.c_rudder_r = machinery_config.rudder_angle_to_yaw_force_coefficient  # 500000.0  # tuning param for simplified rudder response model
        self.rudder_ang_max = machinery_config.max_rudder_angle_degrees * np.pi / 180  # 30 * np.pi / 180  # Maximal rudder angle deflection (both ways)

        # Environmental conditions
        self.vel_c = np.array([environment_config.current_velocity_component_from_north,
                               environment_config.current_velocity_component_from_east,
                               0.0])
        self.wind_dir = environment_config.wind_direction
        self.wind_speed = environment_config.wind_speed

        # Operational parameters used to calculate loading percent on each power source
        self.p_rel_rated_hsg = 0.0
        self.p_rel_rated_me = 0.0

        # Configure machinery system according to self.mso
        #self.mso_mode = simulation_config.machinery_system_operating_mode
        #self.mode_selector(machinery_config.mcr_main_engine,
        #                   machinery_config.mcr_hybrid_shaft_generator)

        # Initial states (can be altered using self.set_state_vector(x))
        self.n = simulation_config.initial_north_position_m
        self.e = simulation_config.initial_east_position_m
        self.psi = simulation_config.initial_yaw_angle_rad
        self.u = simulation_config.initial_forward_speed_m_per_s
        self.v = simulation_config.initial_sideways_speed_m_per_s
        self.r = simulation_config.initial_yaw_rate_rad_per_s
        self.x = self.update_state_vector()
        self.states = np.empty(7)

        # Differentials
        self.d_n = self.d_e = self.d_psi = 0
        self.d_u = self.d_v = self.d_r = 0

        # Set up ship control systems
        self.initialize_ship_speed_controller(kp=7, ki=0.13)
        self.initialize_ship_heading_controller(kp=4, kd=90, ki=0.005)
        self.initialize_heading_filter(kp=0.5, kd=10, t=5000)

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(simulation_config.integration_step)
        self.int.set_sim_time(simulation_config.simulation_time)

        # Instantiate ship draw plotting
        self.drw = ShipDraw()  # Instantiate the ship drawing class
        self.ship_drawings = [[], []]  # Arrays for storing ship drawing data

        # Fuel
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

        # Wind effect on ship
        self.rho_a = 1.2
        self.h_f = 8.0  # mean height above water seen from the front
        self.h_s = 8.0  # mean height above water seen from the side
        self.proj_area_f = self.w_ship * self.h_f  # Projected are from the front
        self.proj_area_l = self.l_ship * self.h_s  # Projected area from the side
        self.cx = 0.5
        self.cy = 0.7
        self.cn = 0.08

        # Fuel consumption function parameters
        self.a_me = 128.89
        self.b_me = -168.93
        self.c_me = 246.76

        self.a_dg = 180.71
        self.b_dg = -289.90
        self.c_dg = 324.90

        self.simulation_results = defaultdict(list)

    def update_available_propulsion_power(self):
        for mode in self.machinery_modes.list_of_modes:
            mode.update_available_propulsion_power(self.hotel_load)

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

    def mode_selector(self, mode: int):
        self.mode = self.machinery_modes.list_of_modes[mode]

    def spec_fuel_cons_me(self, load_perc):
        """ Calculate fuel consumption rate for the main engine.

            Args:
                load_perc (float): The fraction of the mcr load on the ME
            Returns:
                Number of kilograms of fuel per second used by ME
        """
        rate = self.a_me * load_perc ** 2 + self.b_me * load_perc + self.c_me
        return rate / 3.6e9

    def spec_fuel_cons_dg(self, load_perc):
        """ Calculate fuel consumption rate for a diesel generator.

            Args:
                load_perc (float): The fraction of the mcr load on the DG
            Returns:
                Number of kilograms of fuel per second used by DG
        """
        rate = self.a_dg * load_perc ** 2 + self.b_dg * load_perc + self.c_dg
        return rate / 3.6e9

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
        '''
        if self.mso_mode == 1:
            load_me = load_perc * self.p_rated_me + self.hotel_load
            load_perc_me = load_me / self.p_rated_me
            rate_me = load_me * self.spec_fuel_cons_me(load_perc_me)
            rate_hsg = 0.0
        elif self.mso_mode == 2:
            load_me = load_perc * self.p_rated_me
            load_perc_me = load_me / self.p_rated_me
            load_hsg = self.hotel_load
            load_perc_hsg = load_hsg / self.p_rated_hsg
            rate_me = load_me * self.spec_fuel_cons_me(load_perc_me)
            rate_hsg = load_hsg * self.spec_fuel_cons_dg(load_perc_hsg)
        elif self.mso_mode == 3:
            load_hsg = (load_perc * self.p_rated_hsg + self.hotel_load)
            load_perc_hsg = load_hsg / self.p_rated_hsg
            rate_me = 0.0
            rate_hsg = load_hsg * self.spec_fuel_cons_dg(load_perc_hsg)
        '''
        load_data = self.mode.distribute_load(load_perc=load_perc, hotel_load=self.hotel_load)
        if load_data.load_on_main_engine == 0:
            rate_me = 0
        else:
            rate_me = load_data.load_on_main_engine \
                      * self.spec_fuel_cons_me(load_data.load_percentage_on_main_engine)

        if load_data.load_percentage_on_electrical == 0:
            rate_electrical = 0
        else:
            rate_electrical = load_data.load_on_electrical \
                              * self.spec_fuel_cons_dg(load_data.load_percentage_on_electrical)

        self.fuel_cons_me = self.fuel_cons_me + rate_me * self.int.dt
        self.fuel_cons_electrical = self.fuel_cons_electrical + rate_electrical * self.int.dt
        self.fuel_cons = self.fuel_cons + (rate_me + rate_electrical) * self.int.dt
        return rate_me, rate_electrical, self.fuel_cons_me, self.fuel_cons_electrical, self.fuel_cons

    def get_wind_force(self):
        ''' This method calculates the forces due to the relative
            wind speed, acting on teh ship in surge, sway and yaw
            direction.

            :return: Wind force acting in surge, sway and yaw
        '''
        uw = self.wind_speed * np.cos(self.wind_dir - self.psi)
        vw = self.wind_speed * np.sin(self.wind_dir - self.psi)
        u_rw = uw - self.u
        v_rw = vw - self.v
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

    def update_state_vector(self):
        ''' Update the state vector according to the individual state values
        '''
        return np.array([self.n, self.e, self.psi, self.u, self.v, self.r])

    def set_north_pos(self, val):
        ''' Set the north position of the ship and update the state vector
        '''
        self.n = val
        self.x = self.update_state_vector()

    def set_east_pos(self, val):
        ''' Set the east position of the ship and update the state vector
        '''
        self.e = val
        self.x = self.update_state_vector()

    def set_yaw_angle(self, val):
        ''' Set the yaw angle of the ship and update the state vector
        '''
        self.psi = val
        self.x = self.update_state_vector()

    def set_surge_speed(self, val):
        ''' Set the surge speed of the ship and update the state vector
        '''
        self.u = val
        self.x = self.update_state_vector()

    def set_sway_speed(self, val):
        ''' Set the sway speed of the ship and update the state vector
        '''
        self.v = val
        self.x = self.update_state_vector()

    def set_yaw_rate(self, val):
        ''' Set the yaw rate of the ship and update the state vector
        '''
        self.r = val
        self.x = self.update_state_vector()

    def initialize_shaft_speed_controller(self, kp, ki):
        ''' This method sets up and configures the shaft speed
            controller of the ship
        '''
        self.shaft_speed_controller = ControllerLib()
        self.shaft_speed_controller.set_kp(kp)
        self.shaft_speed_controller.set_ki(ki)

    def initialize_ship_speed_controller(self, kp, ki):
        ''' This method sets up and configures the ship speed
            controller.
        '''
        self.ship_speed_controller = ControllerLib()
        self.ship_speed_controller.set_kp(kp)
        self.ship_speed_controller.set_ki(ki)

    def initialize_ship_heading_controller(self, kp, kd, ki):
        ''' This method sets up and configures the ship heading
            controller.
        '''
        self.ship_heading_controller = ControllerLib()
        self.ship_heading_controller.set_kp(kp)
        self.ship_heading_controller.set_kd(-kd)
        self.ship_heading_controller.set_ki(ki)

    def initialize_heading_filter(self, kp, kd, t):
        ''' This method sets up and configures a low pass filter
            to smooth the hading setpoint signal for the ship
            heading controller.
        '''
        self.ship_heading_filter = ControllerLib()
        self.ship_heading_filter.set_kp(kp)
        self.ship_heading_filter.set_kd(kd)
        self.ship_heading_filter.set_T(t)

    def loadperc_from_speedref(self, speed_ref):
        ''' Calculates suitable machinery load percentage for the ship to
            track the speed reference signal. The shaft speed controller
            is used to calculate suitable shaft speed to follow the desired
            ship speed and suitable load percentage to follow the calculated
            shaft speed. The load percentage is the fraction of the produced
            power over the total power capacity in the current configuration.
        '''
        ref_shaft_speed = self.ship_speed_controller.pi_ctrl(speed_ref, self.u, self.int.dt, -550, 550)
        ref_shaft_speed = ControllerLib.sat(ref_shaft_speed, 0, self.shaft_speed_max)
        load_perc = self.shaft_speed_controller.pi_ctrl(ref_shaft_speed, self.omega, self.int.dt)
        load_perc = ControllerLib.sat(load_perc, 0, 1.1)
        return load_perc

    def rudderang_from_headingref(self, heading_ref):
        ''' This method finds a suitable rudder angle for the ship to
            sail with the heading specified by "heading_ref" by using
            PID-controller. The rudder angle is saturated according to
            |self.rudder_ang_max|. The mathod should be called from within
            simulation loop if the user want the ship to follow a specified
            heading reference signal.
        '''
        rudder_ang = self.ship_heading_controller.pid_ctrl(heading_ref, self.psi, self.int.dt)
        rudder_ang = ControllerLib.sat(rudder_ang, -self.rudder_ang_max, self.rudder_ang_max)
        return rudder_ang

    def rudderang_from_route(self):
        ''' This method finds a suitable rudder angle for the ship to follow
            a predefined route specified in the "navigate"-instantiation of the
            "NavigationSystem"-class.
        '''
        self.next_wpt, self.prev_wpt = self.navigate.next_wpt(self.next_wpt, self.n, self.e)
        psi_d = self.navigate.los_guidance(self.next_wpt, self.n, self.e)
        return self.rudderang_from_headingref(psi_d)

    def print_next_wpt(self, ship_id):
        ''' Prints a string with the ship identification (ship_id)
            and its next waypoint, if the next waypoint is specified
        '''
        if self.next_wpt != self.navigate.next_wpt(self.next_wpt, self.n, self.e)[0]:
            print('Current target waypoint for ' + ship_id + ' is: ' + str(self.next_wpt))

    def set_next_wpt(self, wpt):
        ''' Sets the next waypoint to "wpt", where "wpt" is the index
            of the waypoint refering to the list of waypoints making
            up the route specified in the instantiation "navigate" of
            the class "NavigationSystem"
        '''
        self.next_wpt = wpt

    def three_dof_kinematics(self):
        ''' Updates the time differientials of the north position, east
            position and yaw angle. Should be called in the simulation
            loop before the integration step.
        '''
        vel = np.array([self.u, self.v, self.r])
        dx = np.dot(self.rotation(), vel)
        self.d_n = dx[0]
        self.d_e = dx[1]
        self.d_psi = dx[2]

    def rotation(self):
        ''' Specifies the rotation matrix for rotations about the z-axis, such that
            "body-fixed coordinates" = rotation x "North-east-down-fixed coordinates" .
        '''
        return np.array([[np.cos(self.psi), -np.sin(self.psi), 0],
                         [np.sin(self.psi), np.cos(self.psi), 0],
                         [0, 0, 1]])

    def three_dof_kinetics(self, load_perc, rudder_angle):
        ''' Calculates accelerations of the ship, as a funciton
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        # System matrices (did not include added mass yet)
        M_rb = np.array([[self.mass + self.x_du, 0, 0],
                         [0, self.mass + self.y_dv, self.mass * self.x_g],
                         [0, self.mass * self.x_g, self.i_z + self.n_dr]])
        C_rb = np.array([[0, 0, -self.mass * (self.x_g * self.r + self.v)],
                         [0, 0, self.mass * self.u],
                         [self.mass * (self.x_g * self.r + self.v), -self.mass * self.u, 0]])

        D = np.array([[self.mass / self.t_surge, 0, 0],
                      [0, self.mass / self.t_sway, 0],
                      [0, 0, self.i_z / self.t_yaw]])
        D2 = np.array([[self.ku * self.u, 0, 0],
                       [0, self.kv * self.v, 0],
                       [0, 0, self.kr * self.r]])

        # Forces acting (replace zero vectors with suitable functions)
        f_rudder_v, f_rudder_r = self.rudder(rudder_angle)
        self.update_thrust(load_perc)

        F_wind = self.get_wind_force()
        F_waves = np.array([0, 0, 0])
        F_ctrl = np.array([self.thrust, f_rudder_v, f_rudder_r])

        # assembling state vector
        vel = np.array([self.u, self.v, self.r])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.u - v_c[0]
        v_r = self.v - v_c[1]

        C_a = np.array([[0, 0, self.y_dv * v_r],
                        [0, 0, -self.x_du * u_r],
                        [-self.y_dv * v_r, self.x_du * u_r, 0]])

        # Kinetic equation
        M_inv = np.linalg.inv(M_rb)
        dx = np.dot(M_inv, -np.dot(C_rb, vel) - np.dot(C_a, vel - v_c) - np.dot(D + D2, vel - v_c)
                    + F_wind + F_waves + F_ctrl)
        self.d_u = dx[0]
        self.d_v = dx[1]
        self.d_r = dx[2]

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
        v_force = -self.c_rudder_v * delta * (self.u - u_c)
        r_force = -self.c_rudder_r * delta * (self.u - u_c)
        return v_force, r_force

    def update_thrust(self, load_perc):
        ''' Updates the thrust force based on engine power
        '''
        power = load_perc * (self.mode.available_propulsion_power_main_engine
                             + self.mode.available_propulsion_power_electrical)
        self.d_thrust = (-self.k_thrust * self.thrust + power) / self.thrust_time_constant

        self.thrust = self.thrust + self.int.dt * self.d_thrust

    def update_differentials(self, load_perc, rudder_angle):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        self.three_dof_kinetics(load_perc=load_perc, rudder_angle=rudder_angle)

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.set_north_pos(self.int.integrate(self.n, self.d_n))
        self.set_east_pos(self.int.integrate(self.e, self.d_e))
        self.set_yaw_angle(self.int.integrate(self.psi, self.d_psi))
        self.set_surge_speed(self.int.integrate(self.u, self.d_u))
        self.set_sway_speed(self.int.integrate(self.v, self.d_v))
        self.set_yaw_rate(self.int.integrate(self.r, self.d_r))

    def store_states(self):
        ''' Appends the current value of each state to an array. This
            is convenient when plotting. The method should be called within
            the simulation loop each time step. Then afterwars, an array
            containing for ecample the north-position for each time step
            is obtained as ...states[0]
        '''
        self.states[0].append(self.n)
        self.states[1].append(self.e)
        self.states[2].append(self.psi)
        self.states[3].append(self.u)
        self.states[4].append(self.v)
        self.states[5].append(self.r)
        self.states[6].append(self.omega)

    def ship_snap_shot(self):
        ''' This method is used to store a map-view snap shot of
            the ship at the given north-east position and heading.
            It uses the ShipDraw-class. To plot a map view of the
            n-th ship snap-shot, use:

            plot(ship_drawings[1][n], ship_drawings[0][n])
        '''
        x, y = self.drw.local_coords()
        x_ned, y_ned = self.drw.rotate_coords(x, y, self.psi)
        x_ned_trans, y_ned_trans = self.drw.translate_coords(x_ned, y_ned, self.n, self.e)
        self.ship_drawings[0].append(x_ned_trans)
        self.ship_drawings[1].append(y_ned_trans)

    def store_simulation_data(self, load_perc):
        load_perc_me, load_perc_hsg = self.load_perc(load_perc)
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.n)
        self.simulation_results['east position [m]'].append(self.e)
        self.simulation_results['yaw angle [deg]'].append(self.t_yaw * 180 / np.pi)
        self.simulation_results['forward speed[m/s]'].append(self.u)
        self.simulation_results['sideways speed [m/s]'].append(self.v)
        self.simulation_results['yaw rate [deg/sec]'].append(self.r * 180 / np.pi)
        self.simulation_results['commanded load fraction [-]'].append(load_perc)
        self.simulation_results['commanded load fraction me [-]'].append(load_perc_me)
        self.simulation_results['commanded load fraction hsg [-]'].append(load_perc_hsg)

        load_data = self.mode.distribute_load(load_perc=load_perc, hotel_load=self.hotel_load)
        self.simulation_results['power me [kw]'].append(load_data.load_on_main_engine / 1000)
        self.simulation_results['available power me [kw]'].append(self.mode.main_engine_capacity / 1000)
        self.simulation_results['power electrical [kw]'].append(load_data.load_on_electrical / 1000)
        self.simulation_results['available power electrical [kw]'].append(self.mode.electrical_capacity / 1000)
        self.simulation_results['power [kw]'].append((load_data.load_on_electrical
                                                      + load_data.load_on_main_engine) / 1000)
        self.simulation_results['propulsion power [kw]'].append((load_perc
                                                                 * self.mode.available_propulsion_power) / 1000)

        rate_me, rate_hsg, cons_me, cons_hsg, cons = self.fuel_consumption(load_perc)
        self.simulation_results['fuel rate me [kg/s]'].append(rate_me)
        self.simulation_results['fuel rate hsg [kg/s]'].append(rate_hsg)
        self.simulation_results['fuel rate [kg/s]'].append(rate_me + rate_hsg)
        self.simulation_results['fuel consumption me [kg]'].append(cons_me)
        self.simulation_results['fuel consumption hsg [kg]'].append(cons_hsg)
        self.simulation_results['fuel consumption [kg]'].append(cons)
        self.fuel_me.append(cons_me)
        self.fuel_hsg.append(cons_hsg)
        self.fuel.append(cons)
        self.simulation_results['thrust force [kN]'].append(self.thrust / 1000)


class ShipModelWithoutPropulsion:
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

    def __init__(self, ship_config: ShipConfiguration,
                 environment_config: EnvironmentConfiguration,
                 simulation_config: DriftSimulationConfiguration):
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

        # Initial states (can be altered using self.set_state_vector(x))
        self.n = simulation_config.initial_north_position_m
        self.e = simulation_config.initial_east_position_m
        self.psi = simulation_config.initial_yaw_angle_rad
        self.u = simulation_config.initial_forward_speed_m_per_s
        self.v = simulation_config.initial_sideways_speed_m_per_s
        self.r = simulation_config.initial_yaw_rate_rad_per_s
        self.x = self.update_state_vector()
        self.states = np.empty(6)

        # Differentials
        self.d_n = self.d_e = self.d_psi = 0
        self.d_u = self.d_v = self.d_r = 0
        self.hello = 'Hello'

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(simulation_config.integration_step)
        self.int.set_sim_time(simulation_config.simulation_time)

        # Instantiate ship draw plotting
        self.drw = ShipDraw()  # Instantiate the ship drawing class
        self.ship_drawings = [[], []]  # Arrays for storing ship drawing data

        # Wind effect on ship
        self.rho_a = 1.2
        self.h_f = 8.0  # mean height above water seen from the front
        self.h_s = 8.0  # mean height above water seen from the side
        self.proj_area_f = self.w_ship * self.h_f  # Projected are from the front
        self.proj_area_l = self.l_ship * self.h_s  # Projected area from the side
        self.cx = 0.5
        self.cy = 0.7
        self.cn = 0.08

        self.simulation_results = defaultdict(list)

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
        uw = self.wind_speed * np.cos(self.wind_dir - self.psi)
        vw = self.wind_speed * np.sin(self.wind_dir - self.psi)
        u_rw = uw - self.u
        v_rw = vw - self.v
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

    def update_state_vector(self):
        ''' Update the state vector according to the individual state values
        '''
        return np.array([self.n, self.e, self.psi, self.u, self.v, self.r])

    def set_north_pos(self, val):
        ''' Set the north position of the ship and update the state vector
        '''
        self.n = val
        self.x = self.update_state_vector()

    def set_east_pos(self, val):
        ''' Set the east position of the ship and update the state vector
        '''
        self.e = val
        self.x = self.update_state_vector()

    def set_yaw_angle(self, val):
        ''' Set the yaw angle of the ship and update the state vector
        '''
        self.psi = val
        self.x = self.update_state_vector()

    def set_surge_speed(self, val):
        ''' Set the surge speed of the ship and update the state vector
        '''
        self.u = val
        self.x = self.update_state_vector()

    def set_sway_speed(self, val):
        ''' Set the sway speed of the ship and update the state vector
        '''
        self.v = val
        self.x = self.update_state_vector()

    def set_yaw_rate(self, val):
        ''' Set the yaw rate of the ship and update the state vector
        '''
        self.r = val
        self.x = self.update_state_vector()

    def three_dof_kinematics(self):
        ''' Updates the time differientials of the north position, east
            position and yaw angle. Should be called in the simulation
            loop before the integration step.
        '''
        vel = np.array([self.u, self.v, self.r])
        dx = np.dot(self.rotation(), vel)
        self.d_n = dx[0]
        self.d_e = dx[1]
        self.d_psi = dx[2]

    def rotation(self):
        ''' Specifies the rotation matrix for rotations about the z-axis, such that
            "body-fixed coordinates" = rotation x "North-east-down-fixed coordinates" .
        '''
        return np.array([[np.cos(self.psi), -np.sin(self.psi), 0],
                         [np.sin(self.psi), np.cos(self.psi), 0],
                         [0, 0, 1]])

    def three_dof_kinetics(self):
        ''' Calculates accelerations of the ship, as a funciton
            of wind forces and the states in the previous time-step.
        '''
        # System matrices (did not include added mass yet)
        M_rb = np.array([[self.mass + self.x_du, 0, 0],
                         [0, self.mass + self.y_dv, self.mass * self.x_g],
                         [0, self.mass * self.x_g, self.i_z + self.n_dr]])
        C_rb = np.array([[0, 0, -self.mass * (self.x_g * self.r + self.v)],
                         [0, 0, self.mass * self.u],
                         [self.mass * (self.x_g * self.r + self.v), -self.mass * self.u, 0]])

        D = np.array([[self.mass / self.t_surge, 0, 0],
                      [0, self.mass / self.t_sway, 0],
                      [0, 0, self.i_z / self.t_yaw]])
        D2 = np.array([[self.ku * self.u, 0, 0],
                       [0, self.kv * self.v, 0],
                       [0, 0, self.kr * self.r]])

        F_wind = self.get_wind_force()
        F_waves = np.array([0, 0, 0])

        # assembling state vector
        vel = np.array([self.u, self.v, self.r])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.u - v_c[0]
        v_r = self.v - v_c[1]

        C_a = np.array([[0, 0, self.y_dv * v_r],
                        [0, 0, -self.x_du * u_r],
                        [-self.y_dv * v_r, self.x_du * u_r, 0]])

        # Kinetic equation
        M_inv = np.linalg.inv(M_rb)
        dx = np.dot(M_inv, -np.dot(C_rb, vel) - -np.dot(C_a, vel - v_c) - np.dot(D + D2, vel - v_c)
                    + F_wind)
        self.d_u = dx[0]
        self.d_v = dx[1]
        self.d_r = dx[2]

    def update_differentials(self):
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
        self.set_north_pos(self.int.integrate(self.n, self.d_n))
        self.set_east_pos(self.int.integrate(self.e, self.d_e))
        self.set_yaw_angle(self.int.integrate(self.psi, self.d_psi))
        self.set_surge_speed(self.int.integrate(self.u, self.d_u))
        self.set_sway_speed(self.int.integrate(self.v, self.d_v))
        self.set_yaw_rate(self.int.integrate(self.r, self.d_r))

    def store_states(self):
        ''' Appends the current value of each state to an array. This
            is convenient when plotting. The method should be called within
            the simulation loop each time step. Then afterwars, an array
            containing for ecample the north-position for each time step
            is obtained as ...states[0]
        '''
        self.states[0].append(self.n)
        self.states[1].append(self.e)
        self.states[2].append(self.psi)
        self.states[3].append(self.u)
        self.states[4].append(self.v)
        self.states[5].append(self.r)

    def ship_snap_shot(self):
        ''' This method is used to store a map-view snap shot of
            the ship at the given north-east position and heading.
            It uses the ShipDraw-class. To plot a map view of the
            n-th ship snap-shot, use:

            plot(ship_drawings[1][n], ship_drawings[0][n])
        '''
        x, y = self.drw.local_coords()
        x_ned, y_ned = self.drw.rotate_coords(x, y, self.psi)
        x_ned_trans, y_ned_trans = self.drw.translate_coords(x_ned, y_ned, self.n, self.e)
        self.ship_drawings[0].append(x_ned_trans)
        self.ship_drawings[1].append(y_ned_trans)

    def store_simulation_data(self):
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.n)
        self.simulation_results['east position [m]'].append(self.e)
        self.simulation_results['yaw angle [deg]'].append(self.t_yaw * 180 / np.pi)
        self.simulation_results['forward speed[m/s]'].append(self.u)
        self.simulation_results['sideways speed [m/s]'].append(self.v)
        self.simulation_results['yaw rate [deg/sec]'].append(self.r * 180 / np.pi)
        self.simulation_results['wind speed [m/sec]'].append(self.wind_speed)


class ControllerLib:
    ''' This class offers the following set of controllers :
        - P-controller
        - PI-controller
        - PD-controller
        - PID-controller
        - A second order filter
        - Signal saturation
    '''

    def __init__(self):
        self.kp = 1.0
        self.ki = 1.0
        self.kd = 1.0
        self.t = 1.0
        self.prev_error = 0.0
        self.error_i = 0.0

    def set_error_i(self, val):
        ''' Reset/set the value of the error-integral to "val".
            Useful for PI and PID.controllers
        '''
        self.error_i = val

    def set_kp(self, val):
        ''' Set the proportional gain constant
        '''
        self.kp = val

    def set_kd(self, val):
        ''' Set the gain constant for the derivative term
        '''
        self.kd = val

    def set_ki(self, val):
        ''' Set the gain constant for the integral term
        '''
        self.ki = val

    def set_T(self, val):
        ''' Set the time constant. Only relevant
            for the low pass filter
        '''
        self.T = val

    def p_ctrl(self, ref, meas):
        ''' Uses a proportional control law to calculate a control output
        '''
        error = ref - meas
        return self.kp * error

    def pi_ctrl(self, ref, meas, dt, *args):
        ''' Uses a proportional-integral control law to calculate a control
            output. The optional argument is an 2x1 array and will specify lower
            and upper limit for error integration [lower, upper]
        '''
        error = ref - meas
        error_i = self.error_i + error * dt
        if args:
            error_i = self.sat(error_i, args[0], args[1])
        self.error_i = error_i
        return error * self.kp + error_i * self.ki

    def pd_ctrl(self, ref, meas, dt):
        ''' Uses a proportional-derivative control law to calculate a control
            output
        '''
        error = ref - meas
        d_error = (error - self.prev_error) / dt
        self.prev_error = error
        return error * self.kp - d_error * self.kd

    def pid_ctrl(self, ref, meas, dt, *args):
        ''' Uses a proportional-derivative-integral control law to calculate
            a control output. The optional argument is a 2x1 array and will
            specify lower and upper [lower, upper] limit for error integration
        '''
        error = ref - meas
        d_error = (error - self.prev_error) / dt
        error_i = self.error_i + error * dt
        if args:
            error_i = self.sat(error_i, args[0], args[1])
        self.prev_error = error
        self.error_i = error_i
        return error * self.kp - d_error * self.kd + error_i * self.ki

    def filter_2(self, ref, x, v):
        ''' Calculates the two time differentials dx and dv which may be
            integrated to "smooth out" the reference signal "ref"
        '''
        dx = v
        dv = (self.kp * (ref - x) - self.kd * v) / self.t
        return dx, dv

    @staticmethod
    def sat(val, low, hi):
        ''' Saturate the input val such that it remains
        between "low" and "hi"
        '''
        return max(low, min(val, hi))


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
        chi_r = math.atan(-e_ct / delta)
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
class Zones:
    def __init__(self, n_pos, e_pos, object_radius, coll_radius, excl_radius, zone1_radius, zone2_radius, zone3_radius,
                 iceberg_config:IcebergConfiguration):
        self.n = n_pos
        self.e = e_pos
        self.r = coll_radius
        self.r0 = excl_radius
        self.r1 = zone1_radius
        self.r2 = zone2_radius
        self.r3 = zone3_radius
        self.collimargin = 0.5*(iceberg_config.waterlinelength_of_iceberg + object_radius)

    def distance(self, n_iceberg, e_iceberg):
        ''' Returns the distance from a ship with coordinates (north, east)=
            (n_ship, e_ship), to the closest point on the perifery of the
            circular obstacle.
        '''
        rad_2 = (n_iceberg - self.n) ** 2 + (e_iceberg - self.e) ** 2
        rad = np.sqrt(abs(rad_2))
        return rad

    def d_to_north(self, n_iceberg):
        rad = abs(n_iceberg - self.n)
        return rad

    def d_to_east(self, e_iceberg):
        rad = abs(e_iceberg - self.e)
        return rad

    def cpa_zone(self, d_to_s):
        """to calculate which zone the cpa (closest point of approach). d_to_s is the distance between the iceberg center and zone center"""
        if d_to_s - self.collimargin-self.r <= 0:
            cpazone = -1 #"Collision Zone"
        elif d_to_s -self.collimargin -self.r0 <= 0:
            cpazone = 0# "Exclusion Zone"
        elif d_to_s - self.collimargin - self.r1 <= 0:
            cpazone = 1 # "Zone 1"
        elif d_to_s - self.collimargin - self.r2 <= 0:
            cpazone = 2  # "Zone 2"
        elif d_to_s - self.collimargin - self.r3 <= 0:
            cpazone = 3  # "Zone 3"
        else:cpazone = 4 #"outside all zones"
        return cpazone

    def d_to_exclusion(self, n_iceberg, e_iceberg):
        ''' Returns the distance from a ship with coordinates (north, east)=
            (n_ship, e_ship), to the closest point on the perifery of the
            circular obstacle.
        '''
        rad_2 = (n_iceberg - self.n) ** 2 + (e_iceberg - self.e) ** 2
        rad = np.sqrt(abs(rad_2))
        return rad - self.r0-self.collimargin

    def colli_event(self, n_iceberg, e_iceberg):
        rad_2 = (n_iceberg - self.n) ** 2 + (e_iceberg - self.e) ** 2
        rad = np.sqrt(abs(rad_2))
        if rad - self.collimargin-self.r <= 0:
            return 1
        else:
            return 0

    def breach_exclusion(self, n_iceberg, e_iceberg):
        rad_2 = (n_iceberg - self.n) ** 2 + (e_iceberg - self.e) ** 2
        rad = np.sqrt(abs(rad_2))
        if rad - self.collimargin+self.r0 <= 0:
            return 1
        else:
            return 0

    def plot_coll(self):
        ''' This method can be used to plot the obstacle in a
            map-view.
        '''
        # ax = plt.gca()
        return plt.Circle((self.e, self.n), radius=self.r+self.collimargin, fill=False, color='red')

    def plot_excl(self):
        ''' This method can be used to plot the obstacle in a
            map-view.
        '''
        # ax = plt.gca()
        return plt.Circle((self.e, self.n), radius=self.r0+self.collimargin, fill=False, color='red')

    def plot_zone1(self):
        ''' This method can be used to plot the obstacle in a
            map-view.
        '''
        # ax = plt.gca()
        return plt.Circle((self.e, self.n), radius=self.r1+self.collimargin, fill=False, color='orange')

    def plot_zone2(self):
        ''' This method can be used to plot the obstacle in a
            map-view.
        '''
        # ax = plt.gca()
        return plt.Circle((self.e, self.n), radius=self.r2+self.collimargin, fill=False, color='blue')

    def plot_zone3(self):
        ''' This method can be used to plot the obstacle in a
            map-view.
        '''
        # ax = plt.gca()
        return plt.Circle((self.e, self.n), radius=self.r3+self.collimargin, fill=False, color='green')
class IcebergDraw:
    ''' This class is used to calculate the coordinates of each
        corner of 80 meter long and 20meter wide ship seen from above,
        and rotate and translate the coordinates according to
        the ship heading and position
    '''

    def __init__(self,iceberg_config:IcebergConfiguration):
        self.l = iceberg_config.waterlinelength_of_iceberg
        self.b = iceberg_config.width_of_iceberg

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
class IcebergDriftingModel1:
    ''' Creates a iceberg model object that can be used to simulate a iceberg drifting freely

        The model contains the following states:
        - North position of iceberg
        - East position of iceberg
        - Yaw angle (relative to north axis)
        - Surge velocity (forward)
        - Sway velocity (sideways)
        - Yaw rate

        Simulation results are stored in the instance variable simulation_results
    '''

    def __init__(self, iceberg_config: IcebergConfiguration,
                 environment_config: EnvironmentConfiguration,
                 simulation_config: DriftSimulationConfiguration):
        payload = 0.9 * (iceberg_config.mass_tonnage)
        lsw = iceberg_config.mass_tonnage / iceberg_config.coefficient_of_deadweight_to_displacement \
              - iceberg_config.mass_tonnage
        self.mass = lsw + payload

        self.l_iceberg = iceberg_config.waterlinelength_of_iceberg  # 80
        self.w_iceberg = iceberg_config.width_of_iceberg  # 16.0
        self.x_g = 0
        self.i_z = self.mass * (self.l_iceberg ** 2 + self.w_iceberg ** 2) / 12

        # zero-frequency added mass
        self.x_du, self.y_dv, self.n_dr = self.set_added_mass(iceberg_config.added_mass_coefficient_in_surge,
                                                              iceberg_config.added_mass_coefficient_in_sway,
                                                              iceberg_config.added_mass_coefficient_in_yaw)

        self.t_surge = iceberg_config.mass_over_linear_friction_coefficient_in_surge
        self.t_sway = iceberg_config.mass_over_linear_friction_coefficient_in_sway
        self.t_yaw = iceberg_config.mass_over_linear_friction_coefficient_in_yaw
        self.ku = iceberg_config.nonlinear_friction_coefficient__in_surge  # 2400.0  # non-linear friction coeff in surge
        self.kv = iceberg_config.nonlinear_friction_coefficient__in_sway  # 4000.0  # non-linear friction coeff in sway
        self.kr = iceberg_config.nonlinear_friction_coefficient__in_yaw  # 400.0  # non-linear friction coeff in yaw

        # Environmental conditions
        self.vel_c = np.array([environment_config.current_velocity_component_from_north,
                               environment_config.current_velocity_component_from_east,
                               0.0])
        self.wind_dir = environment_config.wind_direction
        self.wind_speed = environment_config.wind_speed

        # Initial states (can be altered using self.set_state_vector(x))
        self.n = simulation_config.initial_north_position_m
        self.e = simulation_config.initial_east_position_m
        self.psi = simulation_config.initial_yaw_angle_rad
        self.u = simulation_config.initial_forward_speed_m_per_s
        self.v = simulation_config.initial_sideways_speed_m_per_s
        self.r = simulation_config.initial_yaw_rate_rad_per_s
        self.x = self.update_state_vector()
        self.states = np.empty(6)

        # Differentials
        self.d_n = self.d_e = self.d_psi = 0
        self.d_u = self.d_v = self.d_r = 0

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(simulation_config.integration_step)
        self.int.set_sim_time(simulation_config.simulation_time)

        # Instantiate ship draw plotting
        self.drw = IcebergDraw(iceberg_config)  # Instantiate the ship drawing class
        self.iceberg_drawings = [[], []]  # Arrays for storing ship drawing data

        # Wind effect on ship
        self.rho_a = 1.2
        self.h_f = 8.0  # mean height above water seen from the front
        self.h_s = 8.0  # mean height above water seen from the side
        self.proj_area_f = self.w_iceberg * self.h_f  # Projected are from the front
        self.proj_area_l = self.l_iceberg * self.h_s  # Projected area from the side
        self.cx = 0.5
        self.cy = 0.7
        self.cn = 0.08

        self.simulation_results = defaultdict(list)

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
        uw = self.wind_speed * np.cos(self.wind_dir - self.psi)
        vw = self.wind_speed * np.sin(self.wind_dir - self.psi)
        u_rw = uw - self.u
        v_rw = vw - self.v
        gamma_rw = -np.arctan2(v_rw, u_rw)
        wind_rw2 = u_rw ** 2 + v_rw ** 2
        c_x = -self.cx * np.cos(gamma_rw)
        c_y = self.cy * np.sin(gamma_rw)
        c_n = self.cn * np.sin(2 * gamma_rw)
        tau_coeff = 0.5 * self.rho_a * wind_rw2
        tau_u = tau_coeff * c_x * self.proj_area_f
        tau_v = tau_coeff * c_y * self.proj_area_l
        tau_n = tau_coeff * c_n * self.proj_area_l * self.l_iceberg
        return np.array([tau_u, tau_v, tau_n])

    def update_state_vector(self):
        ''' Update the state vector according to the individual state values
        '''
        return np.array([self.n, self.e, self.psi, self.u, self.v, self.r])

    def set_north_pos(self, val):
        ''' Set the north position of the iceberg and update the state vector
        '''
        self.n = val
        self.x = self.update_state_vector()

    def set_east_pos(self, val):
        ''' Set the east position of the iceberg and update the state vector
        '''
        self.e = val
        self.x = self.update_state_vector()

    def set_yaw_angle(self, val):
        ''' Set the yaw angle of the iceberg and update the state vector
        '''
        self.psi = val
        self.x = self.update_state_vector()

    def set_surge_speed(self, val):
        ''' Set the surge speed of the iceberg and update the state vector
        '''
        self.u = val
        self.x = self.update_state_vector()

    def set_sway_speed(self, val):
        ''' Set the sway speed of the iceberg and update the state vector
        '''
        self.v = val
        self.x = self.update_state_vector()

    def set_yaw_rate(self, val):
        ''' Set the yaw rate of the iceberg and update the state vector
        '''
        self.r = val
        self.x = self.update_state_vector()

    def three_dof_kinematics(self):
        ''' Updates the time differientials of the north position, east
            position and yaw angle. Should be called in the simulation
            loop before the integration step.
        '''
        vel = np.array([self.u, self.v, self.r])
        dx = np.dot(self.rotation(), vel)
        self.d_n = dx[0]
        self.d_e = dx[1]
        self.d_psi = dx[2]

    def rotation(self):
        ''' Specifies the rotation matrix for rotations about the z-axis, such that
            "body-fixed coordinates" = rotation x "North-east-down-fixed coordinates" .
        '''
        return np.array([[np.cos(self.psi), -np.sin(self.psi), 0],
                         [np.sin(self.psi), np.cos(self.psi), 0],
                         [0, 0, 1]])

    def three_dof_kinetics(self):
        ''' Calculates accelerations of the iceberg, as a funciton
            of wind forces and the states in the previous time-step.
        '''
        # System matrices (did not include added mass yet)
        M_rb = np.array([[self.mass + self.x_du, 0, 0],
                         [0, self.mass + self.y_dv, self.mass * self.x_g],
                         [0, self.mass * self.x_g, self.i_z + self.n_dr]])
        C_rb = np.array([[0, 0, -self.mass * (self.x_g * self.r + self.v)],
                         [0, 0, self.mass * self.u],
                         [self.mass * (self.x_g * self.r + self.v), -self.mass * self.u, 0]])

        D = np.array([[self.mass / self.t_surge, 0, 0],
                      [0, self.mass / self.t_sway, 0],
                      [0, 0, self.i_z / self.t_yaw]])
        D2 = np.array([[self.ku * self.u, 0, 0],
                       [0, self.kv * self.v, 0],
                       [0, 0, self.kr * self.r]])

        F_wind = self.get_wind_force()
        F_waves = np.array([0, 0, 0])

        # assembling state vector
        vel = np.array([self.u, self.v, self.r])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.u - v_c[0]
        v_r = self.v - v_c[1]

        C_a = np.array([[0, 0, self.y_dv * v_r],
                        [0, 0, -self.x_du * u_r],
                        [-self.y_dv * v_r, self.x_du * u_r, 0]])

        # Kinetic equation
        M_inv = np.linalg.inv(M_rb)
        dx = np.dot(M_inv, -np.dot(C_rb, vel) - -np.dot(C_a, vel - v_c) - np.dot(D + D2, vel - v_c)
                    + F_wind)
        self.d_u = dx[0]
        self.d_v = dx[1]
        self.d_r = dx[2]

    def update_differentials(self):
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
        self.set_north_pos(self.int.integrate(self.n, self.d_n))
        self.set_east_pos(self.int.integrate(self.e, self.d_e))
        self.set_yaw_angle(self.int.integrate(self.psi, self.d_psi))
        self.set_surge_speed(self.int.integrate(self.u, self.d_u))
        self.set_sway_speed(self.int.integrate(self.v, self.d_v))
        self.set_yaw_rate(self.int.integrate(self.r, self.d_r))

    def store_states(self):
        ''' Appends the current value of each state to an array. This
            is convenient when plotting. The method should be called within
            the simulation loop each time step. Then afterwars, an array
            containing for ecample the north-position for each time step
            is obtained as ...states[0]
        '''
        self.states[0].append(self.n)
        self.states[1].append(self.e)
        self.states[2].append(self.psi)
        self.states[3].append(self.u)
        self.states[4].append(self.v)
        self.states[5].append(self.r)

    def iceberg_snap_shot(self):
        ''' This method is used to store a map-view snap shot of
            the ship at the given north-east position and heading.
            It uses the ShipDraw-class. To plot a map view of the
            n-th ship snap-shot, use:

            plot(ship_drawings[1][n], ship_drawings[0][n])
        '''
        x, y = self.drw.local_coords()
        x_ned, y_ned = self.drw.rotate_coords(x, y, self.psi)
        x_ned_trans, y_ned_trans = self.drw.translate_coords(x_ned, y_ned, self.n, self.e)
        self.iceberg_drawings[0].append(x_ned_trans)
        self.iceberg_drawings[1].append(y_ned_trans)

    def store_simulation_data(self):
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.n)
        self.simulation_results['east position [m]'].append(self.e)
        self.simulation_results['yaw angle [deg]'].append(self.t_yaw * 180 / np.pi)
        self.simulation_results['forward speed[m/s]'].append(self.u)
        self.simulation_results['sideways speed [m/s]'].append(self.v)
        self.simulation_results['yaw rate [deg/sec]'].append(self.r * 180 / np.pi)
        self.simulation_results['wind speed [m/sec]'].append(self.wind_speed)
        self.simulation_results['wind direction [radius]'].append(self.wind_dir)

    def restore_to_intial(self,simulation_config:DriftSimulationConfiguration):
        self.n = simulation_config.initial_north_position_m
        self.e = simulation_config.initial_east_position_m
        self.psi = simulation_config.initial_yaw_angle_rad
        self.u = simulation_config.initial_forward_speed_m_per_s
        self.v = simulation_config.initial_sideways_speed_m_per_s
        self.r = simulation_config.initial_yaw_rate_rad_per_s


class DistanceSimulation:
    """his class is for simulate drift multiple times to get a distribution of
    collision event and
    time to collision,
    closest point of approach,
    impact point in case of collision,
    when and where iceberg breach the exclusion zone,
    when and where the iceberg breach zone 1
    when and where the iceberg breach zone 2
    when and where the iceberg beach zone 3"""
    def __init__(self, iceberg:IcebergDriftingModel1, zones_config:Zones):
        self.distance_results = defaultdict(list)
        self.iceberg = iceberg
        self.zones_config = zones_config
        self.cpa_point = np.empty(4)
        self.col_point = np.empty(3)
        self.exc_point = np.empty(3)
        self.zone1_point = np.empty(3)
        self.zone2_point = np.empty(3)
        self.zone3_point = np.empty(3)
        self.breach_event = np.empty(5)

    def simulation(self):
        max_wind_speed = 25
        self.distance_results.clear()
        self.iceberg.simulation_results.clear()
        self.iceberg.int.time = 0
        continue_simulation = True
        while self.iceberg.int.time <= self.iceberg.int.sim_time and continue_simulation:
            #self.iceberg.wind_speed = random.random()*max_wind_speed
            self.iceberg.update_differentials()
            self.iceberg.integrate_differentials()
            self.iceberg.store_simulation_data()
            col = self.zones_config.colli_event(self.iceberg.n, self.iceberg.e)

            d = self.zones_config.distance(self.iceberg.n, self.iceberg.e)
            d_to_exc = d-self.zones_config.r0-self.zones_config.collimargin
            d_to_zone1 = d - self.zones_config.r1 - self.zones_config.collimargin
            d_to_zone2 = d - self.zones_config.r2 - self.zones_config.collimargin
            d_to_zone3 = d - self.zones_config.r3 - self.zones_config.collimargin

            dn = self.zones_config.d_to_north(self.iceberg.n)
            de = self.zones_config.d_to_east(self.iceberg.e)

            t = self.iceberg.int.time

            self.distance_results['Time [s]'].append(t)
            self.distance_results['Distance between iceberg and structure [m]'].append(d)
            self.distance_results['Distance between iceberg and structure in north direction [m]'].append(dn)
            self.distance_results['Distance between iceberg and structure in east direction [m]'].append(de)
            self.distance_results['Distance to exclusion zone'].append(d_to_exc)
            self.distance_results['Distance to zone 1'].append(d_to_zone1)
            self.distance_results['Distance to zone 2'].append(d_to_zone2)
            self.distance_results['Distance to zone 3'].append(d_to_zone3)
            self.distance_results['Collision event'].append(col)

            if col == 1:

                continue_simulation = False
                col_time=self.iceberg.int.time
                print('Collision occur at: ', self.iceberg.int.time, 's')
                print("Closest point to Structure:", self.zones_config.distance(self.iceberg.n, self.iceberg.e), 'm', "CPA:", self.iceberg.n, self.iceberg.e)
            elif self.iceberg.n > self.zones_config.r3 + self.zones_config.n:
                continue_simulation = False
            self.iceberg.int.next_time()
    def cpa(self):
        """distance_list = self.distance_results['Distance between iceberg and structure [m]']
        time_list = self.distance_results['Time [s]']
        d_north_list = distance_results['Distance between iceberg and structure in north direction [m]']
        d_east_list = distance_results['Distance between iceberg and structure in east direction [m]']"""

        #cpaf = pd.DataFrame().from_dict(distance_list).min()
        #cpa_d = pd.DataFrame(cpaf).to_numpy()[0, 0]
        #cpaf_idx = pd.DataFrame().from_dict(distance_list).idxmin()
        #cpa_idx = pd.DataFrame(cpaf_idx).to_numpy()[0, 0]
        #cpa_time = pd.DataFrame().from_dict(time_list).loc(cpa_idx)

        cpa_d = min(self.distance_results['Distance between iceberg and structure [m]'])
        cpa_idx = self.distance_results['Distance between iceberg and structure [m]'].index(cpa_d)
        cpa_time = self.distance_results['Time [s]'][cpa_idx]
        cpa_loc = np.empty(2).tolist()
        cpa_loc[0] = self.iceberg.simulation_results['north position [m]'][cpa_idx]
        cpa_loc[1] = self.iceberg.simulation_results['east position [m]'][cpa_idx]
        cpazone = self.zones_config.cpa_zone(cpa_d)
        self.cpa_point = [cpa_d, cpazone, cpa_time, cpa_loc]
        self.col_point = np.empty(3).tolist()
        self.exc_point = np.empty(3).tolist()
        self.zone1_point = np.empty(3).tolist()
        self.zone2_point = np.empty(3).tolist()
        self.zone3_point = np.empty(3).tolist()

        if cpazone == -1:
            col = 1
            exc_breach = 1
            zone1_breach = 1
            zone2_breach = 1
            zone3_breach = 1
            self.col_point = [cpa_time, self.iceberg.simulation_results['north position [m]'][cpa_idx],
                         self.iceberg.simulation_results['east position [m]'][cpa_idx]]
            d_to_exc = self.distance_results['Distance to exclusion zone']
            exc_idx = list(map(lambda i: i <= 0, d_to_exc)).index(True)
            self.exc_point = [self.iceberg.simulation_results['time [s]'][exc_idx],
                         self.iceberg.simulation_results['north position [m]'][exc_idx],
                         self.iceberg.simulation_results['east position [m]'][exc_idx]]
            d_to_zone1 = self.distance_results['Distance to zone 1']
            zone1_idx = list(map(lambda i: i <= 0, d_to_zone1)).index(True)
            self.zone1_point = [self.iceberg.simulation_results['time [s]'][zone1_idx],
                           self.iceberg.simulation_results['north position [m]'][zone1_idx],
                           self.iceberg.simulation_results['east position [m]'][zone1_idx]]
            d_to_zone2 = self.distance_results['Distance to zone 2']
            zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
            self.zone2_point = [self.iceberg.simulation_results['time [s]'][zone2_idx],
                           self.iceberg.simulation_results['north position [m]'][zone2_idx],
                           self.iceberg.simulation_results['east position [m]'][zone2_idx]]
            d_to_zone3 = self.distance_results['Distance to zone 3']
            zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
            self.zone3_point = [self.iceberg.simulation_results['time [s]'][zone3_idx],
                           self.iceberg.simulation_results['north position [m]'][zone3_idx],
                           self.iceberg.simulation_results['east position [m]'][zone3_idx]]
        elif cpazone == 0:
            col = 0
            exc_breach = 1
            zone1_breach = 1
            zone2_breach = 1
            zone3_breach = 1
            d_to_exc = self.distance_results['Distance to exclusion zone']
            exc_idx = list(map(lambda i: i <= 0, d_to_exc)).index(True)
            self.exc_point = [self.iceberg.simulation_results['time [s]'][exc_idx],
                         self.iceberg.simulation_results['north position [m]'][exc_idx],
                         self.iceberg.simulation_results['east position [m]'][exc_idx]]
            d_to_zone1 = self.distance_results['Distance to zone 1']
            zone1_idx = list(map(lambda i: i <= 0, d_to_zone1)).index(True)
            self.zone1_point = [self.iceberg.simulation_results['time [s]'][zone1_idx],
                           self.iceberg.simulation_results['north position [m]'][zone1_idx],
                           self.iceberg.simulation_results['east position [m]'][zone1_idx]]
            d_to_zone2 =self.distance_results['Distance to zone 2']
            zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
            self.zone2_point = [self.iceberg.simulation_results['time [s]'][zone2_idx],
                           self.iceberg.simulation_results['north position [m]'][zone2_idx],
                           self.iceberg.simulation_results['east position [m]'][zone2_idx]]
            d_to_zone3 = self.distance_results['Distance to zone 3']
            zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
            self.zone3_point = [self.iceberg.simulation_results['time [s]'][zone3_idx],
                           self.iceberg.simulation_results['north position [m]'][zone3_idx],
                           self.iceberg.simulation_results['east position [m]'][zone3_idx]]
        elif cpazone == 1:
            col = 0
            exc_breach = 0
            zone1_breach = 1
            zone2_breach = 1
            zone3_breach = 1
            d_to_zone1 = self.distance_results['Distance to zone 1']
            zone1_idx = list(map(lambda i: i <= 0, d_to_zone1)).index(True)
            self.zone1_point = [self.iceberg.simulation_results['time [s]'][zone1_idx],
                           self.iceberg.simulation_results['north position [m]'][zone1_idx],
                           self.iceberg.simulation_results['east position [m]'][zone1_idx]]
            d_to_zone2 = self.distance_results['Distance to zone 2']
            zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
            self.zone2_point = [self.iceberg.simulation_results['time [s]'][zone2_idx],
                           self.iceberg.simulation_results['north position [m]'][zone2_idx],
                           self.iceberg.simulation_results['east position [m]'][zone2_idx]]
            d_to_zone3 = self.distance_results['Distance to zone 3']
            zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
            self.zone3_point = [self.iceberg.simulation_results['time [s]'][zone3_idx],
                           self.iceberg.simulation_results['north position [m]'][zone3_idx],
                           self.iceberg.simulation_results['east position [m]'][zone3_idx]]
        elif cpazone == 2:
            col = 0
            exc_breach = 0
            zone1_breach = 0
            zone2_breach = 1
            zone3_breach = 1
            d_to_zone2 = self.distance_results['Distance to zone 2']
            zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
            self.zone2_point = [self.iceberg.simulation_results['time [s]'][zone2_idx],
                           self.iceberg.simulation_results['north position [m]'][zone2_idx],
                           self.iceberg.simulation_results['east position [m]'][zone2_idx]]
            d_to_zone3 = self.distance_results['Distance to zone 3']
            zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
            self.zone3_point = (self.iceberg.simulation_results['time [s]'][zone3_idx],
                           self.iceberg.simulation_results['north position [m]'][zone3_idx],
                           self.iceberg.simulation_results['east position [m]'][zone3_idx])
        elif cpazone == 3:
            col = 0
            exc_breach = 0
            zone1_breach = 0
            zone2_breach = 0
            zone3_breach = 1
            d_to_zone3 = self.distance_results['Distance to zone 3']
            zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
            self.zone3_point = [self.iceberg.simulation_results['time [s]'][zone3_idx],
                           self.iceberg.simulation_results['north position [m]'][zone3_idx],
                           self.iceberg.simulation_results['east position [m]'][zone3_idx]]
        else:
            col = 0
            exc_breach = 0
            zone1_breach = 0
            zone2_breach = 0
            zone3_breach = 0
        self.breach_event = [col, exc_breach, zone1_breach, zone2_breach, zone3_breach]

class MultiSimulation:
    """this class is for
    1) store data for each simulation,
    2) calculate the distribution of cpa, tcpa etc. for all simulations"""
    def __init__(self, sim_round, dsim_config:DistanceSimulation):
        self.n = sim_round
        self.round_results = defaultdict(list)
        self.cpa_point = dsim_config.cpa_point
        self.breach_event = dsim_config.breach_event
        self.col_point = dsim_config.col_point
        self.exc_point = dsim_config.exc_point
        self.zone1_point = dsim_config.zone1_point
        self.zone2_point = dsim_config.zone2_point
        self.zone3_point = dsim_config.zone3_point
        self.distance_results=dsim_config.distance_results
        self.simulation_results = dsim_config.iceberg.simulation_results
        self.dis_lists = np.empty(self.n, dtype=object)
        self.sim_lists = np.empty(self.n, dtype=object)
        self.sim = dsim_config

    def multsim(self):
        n = 1
        self.round_results.clear()
        while n <= self.n:
            self.sim.simulation()
            self.sim.cpa()
            self.sim_lists[n - 1] = self.sim.iceberg.simulation_results
            self.dis_lists[n - 1] = self.sim.distance_results
            self.round_results['simulation round'].append(n)
            self.round_results['distance between the closest point of approach (cpa) and the structure'].append(self.cpa_point[0])
            self.round_results['zone of closest point of approach (cpa)'].append(self.cpa_point[1])
            self.round_results['time when iceberg reaches the closest point of approach (cpa)'].append(self.cpa_point[2])
            self.round_results['location of the closest point of approach (cpa)'].append(self.cpa_point[3])
            self.round_results['breach event'].append(self.breach_event)
            self.round_results['where the iceberg breach the collision zone'].append(self.col_point[1:3])
            self.round_results['when the iceberg breach the collision zone'].append(self.col_point[0])
            self.round_results['where the iceberg breach the exclusion zone'].append(self.exc_point[1:3])
            self.round_results['when the iceberg breach the exclusion zone'].append(self.exc_point[0])
            self.round_results['where the iceberg breach the zone 1'].append(self.zone1_point[1:3])
            self.round_results['when the iceberg breach the zone 1'].append(self.zone1_point[0])
            self.round_results['where the iceberg breach the zone 2'].append(self.zone2_point[1::3])
            self.round_results['when the iceberg breach the zone 2'].append(self.zone2_point[0])
            self.round_results['where the iceberg breach the zone 3'].append(self.zone3_point[1:3])
            self.round_results['when the iceberg breach the zone 3'].append(self.zone3_point[0])
            n += 1


