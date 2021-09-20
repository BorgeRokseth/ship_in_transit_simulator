from models import ShipModel, ShipConfiguration, EnvironmentConfiguration, \
    MachinerySystemConfiguration, SimulationConfiguration, MachineryModes, \
    MachineryMode, MachineryModeParams
import numpy as np
import pandas as pd
from typing import List



def make_fuel_estimation_lookup_table(
        ship_config: ShipConfiguration,
        machinery_config: MachinerySystemConfiguration,
        speed_setpoints: List[float],
        relative_current_directions: List[float],
        current_magnitudes: List[float],
        relative_wind_directions: List[float],
        wind_magnitudes: List[float],
        lookup_table_file_name: str
):
    mso_mode_list = []
    speed_list = []
    relative_current_direction_list = []
    current_magnitude_list = []
    relative_wind_direction_list = []
    wind_magnitude_list = []
    fuel_consumption_list = []
    for mso_mode_index, mso_mode in enumerate(machinery_config.machinery_modes.list_of_modes):
        for speed in speed_setpoints:
            for rel_current_dir in relative_current_directions:
                for current_magnitude in current_magnitudes:
                    for rel_wind_dir in relative_wind_directions:
                        for wind_magnitude in wind_magnitudes:
                            mso_mode_list.append(mso_mode.name)
                            speed_list.append(speed)
                            relative_current_direction_list.append(rel_current_dir)
                            relative_wind_direction_list.append(rel_wind_dir)
                            current_magnitude_list.append(current_magnitude)
                            wind_magnitude_list.append(wind_magnitude)

                            simulation_setup = SimulationConfiguration(
                                route_name='none',
                                initial_north_position_m=0,
                                initial_east_position_m=0,
                                initial_yaw_angle_rad=0,
                                initial_forward_speed_m_per_s=speed,
                                initial_sideways_speed_m_per_s=0,
                                initial_yaw_rate_rad_per_s=0,
                                initial_propeller_shaft_speed_rad_per_s=200 * np.pi / 30,
                                machinery_system_operating_mode=mso_mode_index,
                                integration_step=0.5,
                                simulation_time=200
                            )
                            env_config = EnvironmentConfiguration(
                                current_velocity_component_from_north=current_magnitude * np.sin(rel_current_dir),
                                current_velocity_component_from_east=current_magnitude * np.cos(rel_current_dir),
                                wind_speed=wind_magnitude,
                                wind_direction=rel_wind_dir
                            )

                            ship_model = ShipModel(ship_config=ship_config,
                                                   machinery_config=machinery_config,
                                                   environment_config=env_config,
                                                   simulation_config=simulation_setup)

                            desired_heading_radians = 0
                            desired_forward_speed_meters_per_second = speed
                            while ship_model.int.time < ship_model.int.sim_time:
                                # Find appropriate rudder angle and engine throttle
                                rudder_angle = -ship_model.rudderang_from_headingref(desired_heading_radians)
                                engine_load = ship_model.loadperc_from_speedref(desired_forward_speed_meters_per_second)

                                # Update and integrate differential equations for current time step
                                _, _, _, _, fuel_consumption = ship_model.fuel_consumption(engine_load)
                                ship_model.update_differentials(load_perc=engine_load, rudder_angle=rudder_angle)
                                ship_model.integrate_differentials()

                                # Progress time variable to the next time step
                                ship_model.int.next_time()

                            fuel_consumption_list.append(fuel_consumption)
    fuel_consumption_data_dict = {
        "Fuel consumption": fuel_consumption_list,
        "MSO-mode": mso_mode_list,
        "Speed": speed_list,
        "Current direction": relative_current_direction_list,
        "Current magnitude": current_magnitude_list,
        "Wind direction": relative_wind_direction_list,
        "Wind magnitude": wind_magnitude_list
    }
    fuel_table = pd.DataFrame.from_dict(fuel_consumption_data_dict)
    fuel_table.index.name = "index"
    fuel_table.to_csv(lookup_table_file_name)

if __name__ == "__main__":
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
    to_mode_params = MachineryModeParams(
        main_engine_capacity=main_engine_capacity,
        electrical_capacity=0,
        shaft_generator_state=hybrid_shaft_gen_as_generator
    )
    pto_mode_params = MachineryModeParams(
        main_engine_capacity=main_engine_capacity,
        electrical_capacity=0,
        shaft_generator_state=hybrid_shaft_gen_as_generator
    )
    pto_mode = MachineryMode(params=pto_mode_params, name="PTO")

    mec_mode_params = MachineryModeParams(
        main_engine_capacity=main_engine_capacity,
        electrical_capacity=diesel_gen_capacity,
        shaft_generator_state=hybrid_shaft_gen_as_offline
    )
    mec_mode = MachineryMode(params=mec_mode_params, name="MEC")

    pti_mode_params = MachineryModeParams(
        main_engine_capacity=0,
        electrical_capacity=2 * diesel_gen_capacity,
        shaft_generator_state=hybrid_shaft_gen_as_motor
    )
    pti_mode = MachineryMode(params=pti_mode_params, name="PTI")

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

    speeds = [5.0, 7.5, 10]
    current_dirs = [
        0,
        90 * np.pi / 180,
        180 * np.pi / 180,
        270 * np.pi / 180,
    ]
    current_magnitudes = [0, 2]
    wind_dirs = [0]
    wind_magnitudes = [2, 7]
    make_fuel_estimation_lookup_table(
        ship_config=ship_config,
        machinery_config=machinery_config,
        speed_setpoints=speeds,
        relative_current_directions=current_dirs,
        current_magnitudes=current_magnitudes,
        relative_wind_directions=wind_dirs,
        wind_magnitudes=wind_magnitudes,
        lookup_table_file_name="fuel-tables/test.csv"
    )