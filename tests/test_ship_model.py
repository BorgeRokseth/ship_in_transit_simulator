from unittest import TestCase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import ShipConfiguration, \
    MachinerySystemConfiguration, \
    EnvironmentConfiguration, \
    SimulationConfiguration, ShipDraw, ShipModel

class TestShipModel(TestCase):
    def make_example_ship(self, route_name: str,
                          initial_yaw_angle: float,
                          initial_forward_speed: float,
                          initial_propeller_shaft_rpm: float):
        ship_config = ShipConfiguration(
            coefficient_of_deadweight_to_displacement=0.7,
            bunkers=200000,
            ballast=200000,
            length_of_ship=80,
            width_of_ship=16,
            added_mass_coefficient_in_surge=0.4,
            added_mass_coefficient_in_sway=0.4,
            added_mass_coefficient_in_yaw=0.4,
            dead_weight_tonnage=3850000,
            mass_over_linear_friction_coefficient_in_surge=130,
            mass_over_linear_friction_coefficient_in_sway=18,
            mass_over_linear_friction_coefficient_in_yaw=90,
            nonlinear_friction_coefficient__in_surge=2400,
            nonlinear_friction_coefficient__in_sway=4000,
            nonlinear_friction_coefficient__in_yaw=400
        )
        env_config = EnvironmentConfiguration(
            current_velocity_component_from_north=0,
            current_velocity_component_from_east=0,
            wind_speed=0,
            wind_direction=0
        )

        machinery_config = MachinerySystemConfiguration(
            mcr_main_engine=2.16e6,
            mcr_hybrid_shaft_generator=0.51e6,
            linear_friction_main_engine=68,
            linear_friction_hybrid_shaft_generator=57,
            gear_ratio_between_main_engine_and_propeller=0.6,
            gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
            propeller_inertia=6000,
            propeller_diameter=3.1,
            propeller_speed_to_torque_coefficient=7.5,
            propeller_speed_to_thrust_force_coefficient=1.7,
            hotel_load=200000,
            rated_speed_main_engine_rpm=1000,
            rudder_angle_to_sway_force_coefficient=50e3,
            rudder_angle_to_yaw_force_coefficient=500e3,
            max_rudder_angle_degrees=30
        )
        simulation_setup = SimulationConfiguration(
            route_name=route_name,
            initial_north_position_m=0,
            initial_east_position_m=0,
            initial_yaw_angle_rad=initial_yaw_angle * np.pi / 180,
            initial_forward_speed_m_per_s=initial_forward_speed,
            initial_sideways_speed_m_per_s=0,
            initial_yaw_rate_rad_per_s=0,
            initial_propeller_shaft_speed_rad_per_s=initial_propeller_shaft_rpm * np.pi / 30,
            machinery_system_operating_mode=1,
            integration_step=1,
            simulation_time=2,
            integral_error_speed_controller=0,
            integral_error_shaft_speed_controller=0
        )
        return ShipModel(ship_config, machinery_config, env_config, simulation_setup)

    def basic_simulation_results(self, initial_forward_speed: float,
                                 initial_propeller_shaft_rpm: float,
                                 initial_yaw_angle: float):
        ship = self.make_example_ship(route_name='none', initial_yaw_angle=initial_yaw_angle,
                                      initial_forward_speed=initial_forward_speed,
                                      initial_propeller_shaft_rpm=initial_propeller_shaft_rpm)
        desired_speed = 8.0
        desired_heading_angle = 15 * np.pi / 180
        while ship.int.time < ship.int.sim_time:
            engine_load_percentage_setpoint = ship.loadperc_from_speedref(desired_speed)
            rudder_angle = ship.rudderang_from_headingref(desired_heading_angle)
            ship.update_differentials(engine_load_percentage_setpoint, rudder_angle)
            ship.integrate_differentials()
            ship.store_simulation_data(load_perc=engine_load_percentage_setpoint)
            ship.int.next_time()
        return pd.DataFrame().from_dict(ship.simulation_results)

    def route_following_simulation_results(self, route_name: str,
                                 initial_forward_speed: float,
                                 initial_propeller_shaft_rpm: float,
                                 initial_yaw_angle: float):
        ship = self.make_example_ship(route_name=route_name, initial_yaw_angle=initial_yaw_angle,
                                      initial_forward_speed=initial_forward_speed,
                                      initial_propeller_shaft_rpm=initial_propeller_shaft_rpm)
        desired_speed = 8.0
        ship.int.set_sim_time(np.sqrt(2 * 1000 ** 2) / 7)
        ship.int.set_dt(0.5)
        while ship.int.time < ship.int.sim_time:
            engine_load_percentage_setpoint = ship.loadperc_from_speedref(desired_speed)
            rudder_angle = -ship.rudderang_from_route()
            ship.update_differentials(engine_load_percentage_setpoint, rudder_angle)
            ship.integrate_differentials()
            ship.store_simulation_data(load_perc=engine_load_percentage_setpoint)
            ship.int.next_time()
        return pd.DataFrame().from_dict(ship.simulation_results)


    def simulation_with_ship_drawing(self, route_name: str,
                                     initial_forward_speed: float,
                                     initial_propeller_shaft_rpm: float,
                                     initial_yaw_angle: float):
        ship = self.make_example_ship(route_name=route_name, initial_yaw_angle=initial_yaw_angle,
                                      initial_forward_speed=initial_forward_speed,
                                      initial_propeller_shaft_rpm=initial_propeller_shaft_rpm)
        desired_speed = 8.0
        ship.int.set_sim_time(np.sqrt(2 * 1000 ** 2) / 7)
        ship.int.set_dt(0.5)
        draw_time = ship.int.sim_time / 2
        draw = True
        while ship.int.time < ship.int.sim_time:
            engine_load_percentage_setpoint = ship.loadperc_from_speedref(desired_speed)
            rudder_angle = -ship.rudderang_from_route()
            ship.update_differentials(engine_load_percentage_setpoint, rudder_angle)
            ship.integrate_differentials()
            ship.store_simulation_data(load_perc=engine_load_percentage_setpoint)

            if ship.int.time > draw_time and draw:
                ship.ship_snap_shot()
                draw = False

            ship.int.next_time()
        return ship.ship_drawings


    def test_time_progression(self):
        sim_results = self.basic_simulation_results(initial_forward_speed=0,
                                                    initial_propeller_shaft_rpm=0,
                                                    initial_yaw_angle=0)
        self.assertEqual(sim_results['time [s]'][0], 0)
        self.assertEqual(sim_results['time [s]'][1], 1)

    def test_route_follow(self):
        sim_results = self.route_following_simulation_results(route_name='route.txt',
                                                              initial_forward_speed=7,
                                                              initial_propeller_shaft_rpm=400,
                                                              initial_yaw_angle=35)

        for n, e in zip(sim_results['north position [m]'], sim_results['east position [m]']):
            self.assertAlmostEqual(n, e, delta=100)


    def test_ship_draw(self):
        ship_drawings = self.simulation_with_ship_drawing(route_name='route.txt',
                                                          initial_forward_speed=7,
                                                          initial_propeller_shaft_rpm=400,
                                                          initial_yaw_angle=40)
        # Check that data has been stored in the ship drawing
        self.assertFalse(len(ship_drawings[0]) == 0)
        self.assertFalse(len(ship_drawings[1]) == 0)

