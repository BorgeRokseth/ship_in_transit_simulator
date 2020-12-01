from unittest import TestCase
import pandas as pd
import numpy as np
from models import ShipConfiguration, \
    SimplifiedPropulsionMachinerySystemConfiguration, \
    EnvironmentConfiguration, \
    SimplifiedPropulsionSimulationConfiguration, \
    ShipModel,\
    MachineryModes,\
    MachineryModeParams,\
    MachineryMode, SimulationConfiguration, MachinerySystemConfiguration, ShipModelSimplifiedPropulsion

class TestShipModel(TestCase):
    @staticmethod
    def make_example_ship_w_simplified_propulsion(mso_mode: int):
        main_engine_capacity = 2160e3
        diesel_gen_capacity = 510e3
        hybrid_shaft_gen_as_generator = 'GEN'
        hybrid_shaft_gen_as_motor = 'MOTOR'
        hybrid_shaft_gen_as_offline = 'OFF'

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

        machinery_config = SimplifiedPropulsionMachinerySystemConfiguration(
            hotel_load=200e3,
            machinery_modes=mso_modes,
            max_rudder_angle_degrees=30,
            rudder_angle_to_yaw_force_coefficient=500e3,
            rudder_angle_to_sway_force_coefficient=50e3,
            thrust_force_dynamic_damping=0.5,
            thrust_force_dynamic_time_constant=30
        )

        simulation_setup = SimplifiedPropulsionSimulationConfiguration(
            route_name='none',
            initial_north_position_m=0,
            initial_east_position_m=0,
            initial_yaw_angle_rad=0 * np.pi / 180,
            initial_forward_speed_m_per_s=0,
            initial_sideways_speed_m_per_s=0,
            initial_yaw_rate_rad_per_s=0,
            initial_thrust_force=0,
            machinery_system_operating_mode=mso_mode,
            integration_step=0.5,
            simulation_time=500
        )

        return ShipModelSimplifiedPropulsion(ship_config, machinery_config, env_config, simulation_setup)

    @staticmethod
    def make_example_ship(mso_mode: int):

        main_engine_capacity = 2160e3
        diesel_gen_capacity = 510e3
        hybrid_shaft_gen_as_generator = 'GEN'
        hybrid_shaft_gen_as_motor = 'MOTOR'
        hybrid_shaft_gen_as_offline = 'OFF'

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

        machinery_config = MachinerySystemConfiguration(
            hotel_load=200e3,
            machinery_modes=mso_modes,
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
            rudder_angle_to_sway_force_coefficient=50e3
        )

        simulation_setup = SimulationConfiguration(
            route_name='none',
            initial_north_position_m=0,
            initial_east_position_m=0,
            initial_yaw_angle_rad=0 * np.pi / 180,
            initial_forward_speed_m_per_s=0,
            initial_sideways_speed_m_per_s=0,
            initial_yaw_rate_rad_per_s=0,
            initial_propeller_shaft_speed_rad_per_s=0 * np.pi / 30,
            machinery_system_operating_mode=mso_mode,
            integration_step=0.5,
            simulation_time=500
        )
        return ShipModel(ship_config, machinery_config, env_config, simulation_setup)

    def step_response_w_simplified_propulsion(self, mso_mode: int):
        ship = self.make_example_ship_w_simplified_propulsion(mso_mode=mso_mode)
        desired_heading_angle = 0 * np.pi / 180
        while ship.int.time < ship.int.sim_time:
            if ship.int.time < 20:
                engine_load = 0
            else:
                engine_load = 1.0
            rudder_angle = ship.rudderang_from_headingref(desired_heading_angle)
            ship.update_differentials(engine_load, rudder_angle)
            ship.integrate_differentials()
            ship.store_simulation_data(load_perc=engine_load)
            ship.int.next_time()
        return pd.DataFrame().from_dict(ship.simulation_results)

    def step_response(self, mso_mode: int):
        ship = self.make_example_ship(mso_mode=mso_mode)
        desired_heading_angle = 0 * np.pi / 180
        while ship.int.time < ship.int.sim_time:
            if ship.int.time < 20:
                engine_load = 0
            else:
                engine_load = 1.0
            rudder_angle = ship.rudderang_from_headingref(desired_heading_angle)
            ship.update_differentials(engine_load, rudder_angle)
            ship.integrate_differentials()
            ship.store_simulation_data(load_perc=engine_load)
            ship.int.next_time()
        return pd.DataFrame().from_dict(ship.simulation_results)

    def test_power_speed_step_is_equal(self):
        step_resp_simplified = self.step_response_w_simplified_propulsion(mso_mode=1)
        step_resp_regular = self.step_response(mso_mode=1)
        for u_simpl, u in zip(step_resp_simplified['forward speed[m/s]'],
                                      step_resp_regular['forward speed[m/s]']):
            self.assertAlmostEqual(u_simpl, u, delta=0.5)

    def test_power_pos_step_is_equal(self):
        step_resp_simplified = self.step_response_w_simplified_propulsion(mso_mode=1)
        step_resp_regular = self.step_response(mso_mode=1)
        for n_simpl, n in zip(step_resp_simplified['north position [m]'],
                                      step_resp_regular['north position [m]']):
            self.assertAlmostEqual(n_simpl, n, delta=200)

    def test_max_thrust_comparison_pto(self):
        step_resp_simplified = self.step_response_w_simplified_propulsion(mso_mode=0)
        step_resp_regular = self.step_response(mso_mode=0)
        max_thrust_on_simplified_model = step_resp_simplified['thrust force [kN]'][500]
        max_thrust_on_regular_model = step_resp_regular['thrust force [kN]'][500]
        self.assertAlmostEqual(max_thrust_on_simplified_model,
                               max_thrust_on_regular_model, delta=50)

    def test_max_thrust_comparison_mec(self):
        step_resp_simplified = self.step_response_w_simplified_propulsion(mso_mode=1)
        step_resp_regular = self.step_response(mso_mode=1)
        max_thrust_on_simplified_model = step_resp_simplified['thrust force [kN]'][500]
        max_thrust_on_regular_model = step_resp_regular['thrust force [kN]'][500]
        self.assertAlmostEqual(max_thrust_on_simplified_model,
                               max_thrust_on_regular_model, delta=50)

    def test_max_thrust_comparison_pti(self):
        step_resp_simplified = self.step_response_w_simplified_propulsion(mso_mode=2)
        step_resp_regular = self.step_response(mso_mode=2)
        max_thrust_on_simplified_model = step_resp_simplified['thrust force [kN]'][500]
        max_thrust_on_regular_model = step_resp_regular['thrust force [kN]'][500]
        self.assertAlmostEqual(max_thrust_on_simplified_model,
                               max_thrust_on_regular_model, delta=100)




