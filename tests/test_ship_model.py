from unittest import TestCase
import pandas as pd
import numpy as np
from models import ShipConfiguration, \
    MachinerySystemConfiguration, \
    EnvironmentConfiguration, \
    SimulationConfiguration,\
    ShipDraw, ShipModel,\
    MachineryModes,\
    MachineryModeParams, SimplifiedPropulsionMachinerySystemConfiguration,\
    MachineryMode, SpecificFuelConsumptionWartila6L26, \
    SpecificFuelConsumptionBaudouin6M26Dot3, FuelConsumptionCoefficients, SimplifiedMachineryModel

class TestMachinerySystem(TestCase):
    def set_up_machinery(self, me_rating, dg_rating, operating_mode, hotel_load):
        hybrid_shaft_gen_as_generator = 'GEN'
        hybrid_shaft_gen_as_motor = 'MOTOR'
        hybrid_shaft_gen_as_offline = 'OFF'
        pto_mode_params = MachineryModeParams(
            main_engine_capacity=me_rating,
            electrical_capacity=0,
            shaft_generator_state=hybrid_shaft_gen_as_generator
        )
        pto_mode = MachineryMode(params=pto_mode_params)

        mec_mode_params = MachineryModeParams(
            main_engine_capacity=me_rating,
            electrical_capacity=dg_rating,
            shaft_generator_state=hybrid_shaft_gen_as_offline
        )
        mec_mode = MachineryMode(params=mec_mode_params)

        pti_mode_params = MachineryModeParams(
            main_engine_capacity=0,
            electrical_capacity=2 * dg_rating,
            shaft_generator_state=hybrid_shaft_gen_as_motor
        )
        pti_mode = MachineryMode(params=pti_mode_params)

        mso_modes = MachineryModes(
            [pto_mode,
             mec_mode,
             pti_mode]
        )
        fuel_curves_me = FuelConsumptionCoefficients(a=0, b=1, c=0)
        fuel_curves_dg = FuelConsumptionCoefficients(a=0, b=1, c=0)
        self.fuel_curves_me = fuel_curves_me
        self.fuel_curves_dg = fuel_curves_dg
        machinery_config = SimplifiedPropulsionMachinerySystemConfiguration(
            hotel_load=hotel_load,
            machinery_modes=mso_modes,
            machinery_operating_mode=operating_mode,
            max_rudder_angle_degrees=30,
            rudder_angle_to_yaw_force_coefficient=500e3,
            rudder_angle_to_sway_force_coefficient=50e3,
            specific_fuel_consumption_coefficients_me=fuel_curves_me,
            specific_fuel_consumption_coefficients_dg=fuel_curves_dg,
            thrust_force_dynamic_time_constant=30
        )
        return SimplifiedMachineryModel(
            machinery_config=machinery_config,
            time_step=1,
            initial_thrust_force=0
        )
    def test_load_perc_pto(self):
        machinery = self.set_up_machinery(
            me_rating=1000,
            dg_rating=500,
            operating_mode=0,
            hotel_load=100
        )
        load_perc_on_me, load_perc_on_el = machinery.load_perc(load_perc=1)
        self.assertEqual(1, load_perc_on_me)
        self.assertEqual(0, load_perc_on_el)

    def test_load_perc_mec(self):
        machinery = self.set_up_machinery(
            me_rating=1000,
            dg_rating=500,
            operating_mode=1,
            hotel_load=100
        )
        load_perc_on_me, load_perc_on_el = machinery.load_perc(load_perc=1)
        electrical_load = 100 / 500  # Due to hotel load
        self.assertEqual(1, load_perc_on_me)
        self.assertEqual(electrical_load, load_perc_on_el)

    def test_load_perc_pti(self):
        machinery = self.set_up_machinery(
            me_rating=1000,
            dg_rating=500,
            operating_mode=2,
            hotel_load=100
        )
        load_perc_on_me, load_perc_on_el = machinery.load_perc(load_perc=1)

        self.assertEqual(0, load_perc_on_me)
        self.assertEqual(1, load_perc_on_el)

    def test_fuel_consumption_pto(self):
        machinery = self.set_up_machinery(
            me_rating=1000,
            dg_rating=500,
            operating_mode=0,
            hotel_load=100
        )
        load_perc_on_me, load_perc_on_el = machinery.load_perc(load_perc=1)

        # ME will generate 1000kW. 900 of these will be for propulsion, the rest for hotel
        self.assertEqual(900, machinery.mode.available_propulsion_power_main_engine)
        fuel_cons_rate = 1000 * load_perc_on_me / 3.6e9
        rate_me, rate_el, fuel_cons_me, fuel_cons_el, fuel_cons = machinery.fuel_consumption(
            load_perc=1
        )
        self.assertEqual(fuel_cons_rate, rate_me)

    def test_fuel_consumption_mec(self):
        machinery = self.set_up_machinery(
            me_rating=1000,
            dg_rating=500,
            operating_mode=0,
            hotel_load=100
        )
        load_perc_on_me, load_perc_on_el = machinery.load_perc(load_perc=1)

        # ME will generate 1000kW and dg will take case of hotel.
        # 1000 of these will be for propulsion
        self.assertEqual(1000, machinery.mode.available_propulsion_power_main_engine)
        fuel_cons_rate_me = 1000 * load_perc_on_me / 3.6e9
        fuel_cons_rate_dg = 100 * load_perc_on_el / 3.6e9
        rate_me, rate_el, fuel_cons_me, fuel_cons_el, fuel_cons = machinery.fuel_consumption(
            load_perc=1
        )
        self.assertEqual(fuel_cons_rate_me, rate_me)
        self.assertEqual(fuel_cons_rate_dg, rate_el)
        self.assertEqual(fuel_cons_rate_dg+fuel_cons_rate_me, fuel_cons)



class TestShipModel(TestCase):
    @staticmethod
    def make_example_ship(route_name: str,
                          initial_yaw_angle: float,
                          initial_forward_speed: float,
                          initial_propeller_shaft_rpm: float):

        main_engine_capacity = 1000e3
        diesel_gen_capacity = 500e3
        hybrid_shaft_gen_as_generator = 'GEN'
        hybrid_shaft_gen_as_motor = 'MOTOR'
        hybrid_shaft_gen_as_offline = 'OFF'

        time_step = 0.5

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

        pto_mode_params = MachineryModeParams(
            main_engine_capacity=main_engine_capacity,
            electrical_capacity=0,
            shaft_generator_state=hybrid_shaft_gen_as_generator
        )
        pto_mode = MachineryMode(params=pto_mode_params)

        mec_mode_params = MachineryModeParams(
            main_engine_capacity=main_engine_capacity,
            electrical_capacity=diesel_gen_capacity,
            shaft_generator_state=hybrid_shaft_gen_as_offline
        )
        mec_mode = MachineryMode(params=mec_mode_params)

        pti_mode_params = MachineryModeParams(
            main_engine_capacity=0,
            electrical_capacity=2 * diesel_gen_capacity,
            shaft_generator_state=hybrid_shaft_gen_as_motor
        )
        pti_mode = MachineryMode(params=pti_mode_params)

        mso_modes = MachineryModes(
            [pto_mode,
             mec_mode,
             pti_mode]
        )
        fuel_curves_me = SpecificFuelConsumptionWartila6L26()
        fuel_curves_dg = SpecificFuelConsumptionBaudouin6M26Dot3()

        machinery_config = MachinerySystemConfiguration(
            hotel_load=200e3,
            machinery_modes=mso_modes,
            machinery_operating_mode=1,
            rated_speed_main_engine_rpm=1000,
            linear_friction_main_engine=68,
            linear_friction_hybrid_shaft_generator=57,
            gear_ratio_between_main_engine_and_propeller=0.6,
            gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
            propeller_inertia=6000,
            propeller_diameter=3.1,
            propeller_speed_to_torque_coefficient=7.5,
            propeller_speed_to_thrust_force_coefficient=1.7,
            max_rudder_angle_degrees=30,
            rudder_angle_to_yaw_force_coefficient=500e3,
            rudder_angle_to_sway_force_coefficient=50e3,
            specific_fuel_consumption_coefficients_me=fuel_curves_me.fuel_consumption_coefficients(),
            specific_fuel_consumption_coefficients_dg=fuel_curves_dg.fuel_consumption_coefficients()
        )
        simulation_setup = SimulationConfiguration(
            initial_north_position_m=0,
            initial_east_position_m=0,
            initial_yaw_angle_rad=10 * np.pi / 180,
            initial_forward_speed_m_per_s=7,
            initial_sideways_speed_m_per_s=0,
            initial_yaw_rate_rad_per_s=0,
            integration_step=time_step,
            simulation_time=300
        )

        return ShipModel(
            ship_config=ship_config,
            machinery_config=machinery_config,
            environment_config=env_config,
            simulation_config=simulation_setup,
            initial_propeller_shaft_speed_rad_per_s=200 * np.pi / 30
        )

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

    def test_available_power_for_propulsion_pto(self):
        ship = self.make_example_ship(route_name='none', initial_yaw_angle=0,
                                      initial_forward_speed=0,
                                      initial_propeller_shaft_rpm=0)
        ship.mode_selector(0)
        available_propulsion_power_in_pto = ship.mode.main_engine_capacity - ship.hotel_load
        ship.mode.update_available_propulsion_power(hotel_load=ship.hotel_load)
        assert(ship.mode.available_propulsion_power == available_propulsion_power_in_pto)

    def test_available_power_for_propulsion_mec(self):
        ship = self.make_example_ship(route_name='none', initial_yaw_angle=0,
                                      initial_forward_speed=0,
                                      initial_propeller_shaft_rpm=0)
        ship.mode_selector(1)
        available_propulsion_power_in_mec = ship.mode.main_engine_capacity
        ship.mode.update_available_propulsion_power(hotel_load=ship.hotel_load)
        assert(ship.mode.available_propulsion_power == available_propulsion_power_in_mec)

    def test_available_power_for_propulsion_pti(self):
        ship = self.make_example_ship(route_name='none', initial_yaw_angle=0,
                                      initial_forward_speed=0,
                                      initial_propeller_shaft_rpm=0)
        ship.mode_selector(2)
        available_propulsion_power_in_pti = ship.mode.electrical_capacity - ship.hotel_load
        ship.mode.update_available_propulsion_power(hotel_load=ship.hotel_load)
        assert(ship.mode.available_propulsion_power == available_propulsion_power_in_pti)



