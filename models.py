import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict

from typing import NamedTuple

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


class MachinerySystemConfiguration(NamedTuple):
    hotel_load: float
    mcr_main_engine: float
    mcr_hybrid_shaft_generator: float
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


class ShipModel:
    ''' A 7-state simulation model for a ship in transit. The ships is
        propelled by a single propeller and steered by a rudder. The propeller
        is powered by either the main engine, an auxiliary motor referred to
        as the hybrid shaft generator, or both. The seven states are:
        - North position of ship
        - East position of ship
        - Yaw angle (relative to north axis)
        - Surge velocity (forward)
        - Sway velocity (sideways)
        - Yaw rate
        - Propeller shaft speed
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
        #self.set_added_mass(0.4, 0.4, 0.4)
        self.set_added_mass(ship_config.added_mass_coefficient_in_surge,
                            ship_config.added_mass_coefficient_in_sway,
                            ship_config.added_mass_coefficient_in_yaw)

        self.t_surge = ship_config.mass_over_linear_friction_coefficient_in_surge
        self.t_sway = ship_config.mass_over_linear_friction_coefficient_in_sway
        self.t_yaw = ship_config.mass_over_linear_friction_coefficient_in_yaw
        self.ku = ship_config.nonlinear_friction_coefficient__in_surge  # 2400.0  # non-linear friction coeff in surge
        self.kv = ship_config.nonlinear_friction_coefficient__in_sway  # 4000.0  # non-linear friction coeff in sway
        self.kr = ship_config.nonlinear_friction_coefficient__in_yaw  # 400.0  # non-linear friction coeff in yaw

        # Machinery system params
        self.hotel_load = machinery_config.hotel_load  #200000  # 0.2 MW
        self.p_rated_me = machinery_config.mcr_main_engine  # 2160000  # 2.16 MW
        self.p_rated_hsg = machinery_config.mcr_hybrid_shaft_generator # 590000  # 0.59 MW
        self.w_rated_me = machinery_config.rated_speed_main_engine_rpm * np.pi / 30  # 1000 * np.pi / 30  # rated speed
        self.d_me = machinery_config.linear_friction_main_engine  #68.0  # linear friction for main engine speed
        self.d_hsg = machinery_config.linear_friction_hybrid_shaft_generator  # 57.0  # linear friction for HSG speed
        self.r_me = machinery_config.gear_ratio_between_main_engine_and_propeller  #0.6  # gear ratio between main engine and propeller
        self.r_hsg = machinery_config.gear_ratio_between_hybrid_shaft_generator_and_propeller  # 0.6  # gear ratio between main engine and propeller
        self.jp = machinery_config.propeller_inertia  # 6000  # propeller inertia
        self.kp = machinery_config.propeller_speed_to_torque_coefficient  # 7.5  # constant relating omega to torque
        self.dp = machinery_config.propeller_diameter  #3.1  # propeller diameter
        self.kt = machinery_config.propeller_speed_to_thrust_force_coefficient  #1.7  # constant relating omega to thrust force
        self.shaft_speed_max = 1.1 * self.w_rated_me * self.r_me  # Used for saturation of power sources

        self.c_rudder_v = machinery_config.rudder_angle_to_sway_force_coefficient  # 50000.0  # tuning param for simplified rudder response model
        self.c_rudder_r = machinery_config.rudder_angle_to_yaw_force_coefficient  # 500000.0  # tuning param for simplified rudder response model
        self.rudder_ang_max = machinery_config.max_rudder_angle_degrees * np.pi / 180 #30 * np.pi / 180  # Maximal rudder angle deflection (both ways)

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
        self.mso_mode = simulation_config.machinery_system_operating_mode
        self.mode_selector(machinery_config.mcr_main_engine,
                           machinery_config.mcr_hybrid_shaft_generator)

        # Initial states (can be altered using self.set_state_vector(x))
        self.n = simulation_config.initial_north_position_m
        self.e = simulation_config.initial_east_position_m
        self.psi = simulation_config.initial_yaw_angle_rad
        self.u = simulation_config.initial_forward_speed_m_per_s
        self.v = simulation_config.initial_sideways_speed_m_per_s
        self.r = simulation_config.initial_yaw_rate_rad_per_s
        self.omega = simulation_config.initial_propeller_shaft_speed_rad_per_s
        self.x = self.update_state_vector()
        self.states = np.ndarray(shape=7)

        # Differentials
        self.d_n = self.d_e = self.d_psi = 0
        self.d_u = self.d_v = self.d_r = 0
        self.d_omega = 0

        # Set up ship control systems
        self.initialize_shaft_speed_controller(kp=0.1, ki=0.005)
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
        self.fuel_cons_hsg = 0.0  # Initial fuel cons for HSG
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

    def set_added_mass(self, surge_coeff, sway_coeff, yaw_coeff):
        ''' Sets the added mass in surge due to surge motion, sway due
            to sway motion and yaw due to yaw motion according to given coeffs
        '''
        self.x_du = self.mass * surge_coeff
        self.y_dv = self.mass * sway_coeff
        self.n_dr = self.i_z * yaw_coeff

    def mode_selector(self, mcr_me, mcr_hsg):
        ''' Select mode between mode 1, 2 or 3
        '''
        if self.mso_mode is 1:
            # PTO
            # ME is online and loaded with hotel loads
            self.p_rel_rated_me = mcr_me
            self.p_rel_rated_hsg = 0.0
            self.p_rated_me = mcr_me
            self.p_rated_hsg = 0.0
        elif self.mso_mode is 2:
            # Mechanical
            # ME is responsible for only propulsion
            self.p_rel_rated_me = mcr_me
            self.p_rel_rated_hsg = 0
            self.p_rated_me = mcr_me
            self.p_rated_hsg = mcr_hsg
        elif self.mso_mode is 3:
            # PTI with one DG
            # HSG is responsible for propulsion and hotel
            self.p_rel_rated_me = 0
            self.p_rel_rated_hsg = 2 * mcr_hsg
            self.p_rated_me = 0.0
            self.p_rated_hsg = 2 * mcr_hsg

    def spec_fuel_cons_me(self, load_perc):
        rate = self.a_me * load_perc ** 2 + self.b_me * load_perc + self.c_me
        return rate / 3.6e9

    def spec_fuel_cons_dg(self, load_perc):
        rate = self.a_dg * load_perc ** 2 + self.b_dg * load_perc + self.c_dg
        return rate/3.6e9

    def load_perc(self, load_perc):
        if self.mso_mode == 1:
            load_me = load_perc * self.p_rated_me + self.hotel_load
            load_perc_me = load_me / self.p_rated_me
            load_perc_hsg = 0.0
        elif self.mso_mode == 2:
            load_me = load_perc * self.p_rated_me
            load_perc_me = load_me / self.p_rated_me
            load_hsg = self.hotel_load
            load_perc_hsg = load_hsg / self.p_rated_hsg
        elif self.mso_mode == 3:
            load_hsg = (load_perc * self.p_rated_hsg + self.hotel_load)
            load_perc_me = 0.0
            load_perc_hsg = load_hsg / self.p_rated_hsg
        return load_perc_me, load_perc_hsg

    def fuel_consumption(self, load_perc):
        '''
            :param load_perc: The fraction of produced power over the online
            power production capacity.
            :return: fuel consumption rate for ME and HSG, and
            accumulated fuel consumption for ME, HSG and total
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

        self.fuel_cons_me = self.fuel_cons_me + rate_me * self.int.dt
        self.fuel_cons_hsg = self.fuel_cons_hsg + rate_hsg * self.int.dt
        self.fuel_cons = self.fuel_cons + (rate_me + rate_hsg) * self.int.dt
        return rate_me, rate_hsg, self.fuel_cons_me, self.fuel_cons_hsg, self.fuel_cons

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
                         [0, self.mass * self.x_g, self.i_z+ self.n_dr]])
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
        dx = np.dot(M_inv, -np.dot(C_rb, vel) - -np.dot(C_a, vel - v_c) - np.dot(D + D2, vel - v_c)
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
        if self.omega >= 100:
            return load_perc * self.p_rel_rated_me / self.omega
        else:
            return 0

    def hsg_torque(self, load_perc):
        ''' Returns the torque of the HSG as a
            function of the load percentage parameter
        '''
        if self.omega >= 100:
            return load_perc * self.p_rel_rated_hsg / self.omega
        else:
            return 0

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
        if self.mso_mode == 1:
            power_me = load_perc * self.p_rated_me + self.hotel_load
            power_hsg = load_perc * self.p_rated_hsg
            self.simulation_results['power me [kw]'].append(power_me / 1000)
            self.simulation_results['rated power me [kw]'].append(self.p_rated_me / 1000)
            self.simulation_results['power hsg [kw]'].append(power_hsg / 1000)
            self.simulation_results['rated power hsg [kw]'].append(self.p_rated_hsg / 1000)
            self.simulation_results['power [kw]'].append((power_me + power_hsg) / 1000)
            self.simulation_results['propulsion power [kw]'].append((load_perc * self.p_rated_me) / 1000)
        elif self.mso_mode == 2:
            power_me = load_perc * self.p_rated_me
            power_hsg = self.hotel_load
            self.simulation_results['power me [kw]'].append(power_me / 1000)
            self.simulation_results['rated power me [kw]'].append(self.p_rated_me / 1000)
            self.simulation_results['power hsg [kw]'].append(power_hsg / 1000)
            self.simulation_results['rated power hsg [kw]'].append(self.p_rated_hsg / 1000)
            self.simulation_results['power [kw]'].append((power_me + power_hsg) / 1000)
            self.simulation_results['propulsion power [kw]'].append(power_me / 1000)
        elif self.mso_mode == 3:
            power_me = load_perc * self.p_rated_me
            power_hsg = load_perc * self.p_rated_hsg + self.hotel_load
            self.simulation_results['power me [kw]'].append(power_me / 1000)
            self.simulation_results['rated power me [kw]'].append(self.p_rated_me / 1000)
            self.simulation_results['power hsg [kw]'].append(power_hsg / 1000)
            self.simulation_results['rated power hsg [kw]'].append(self.p_rated_hsg / 1000)
            self.simulation_results['power [kw]'].append((power_me + power_hsg) / 1000)
            self.simulation_results['propulsion power [kw]'].append(load_perc * self.p_rated_hsg / 1000)
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

    def plot_obst(self):
        ''' This method can be used to plot the obstacle in a
            map-view.
        '''
        ax = plt.gca()
        ax.add_patch(plt.Circle((self.e, self.n), radius=self.r, fill=True, color='grey'))
