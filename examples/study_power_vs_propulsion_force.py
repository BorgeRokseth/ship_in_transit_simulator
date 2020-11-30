from models import ShipModel, ShipConfiguration, EnvironmentConfiguration, \
    MachinerySystemConfiguration, SimulationConfiguration, MachineryModes, \
    MachineryMode, MachineryModeParams
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

ship_models = []
results = []
power_samples = []
force_samples = []
early_power_samples = []
early_force_samples = []

n = 20

for i in range(0,n):
    simulation_setup = SimulationConfiguration(
        route_name='none',
        initial_north_position_m=0,
        initial_east_position_m=0,
        initial_yaw_angle_rad=2*i * np.pi / 180,
        initial_forward_speed_m_per_s=0,
        initial_sideways_speed_m_per_s=0,
        initial_yaw_rate_rad_per_s=0,
        initial_propeller_shaft_speed_rad_per_s=0 * np.pi / 30,
        machinery_system_operating_mode=1,
        integration_step=0.1,
        simulation_time=200
    )

    ship_models.append(
        ShipModel(
            ship_config=ship_config,
            machinery_config=machinery_config,
            environment_config=env_config,
            simulation_config=simulation_setup)
    )

    desired_heading_radians = 2 * i * np.pi / 180

    while ship_models[-1].int.time < ship_models[-1].int.sim_time:
        # Find appropriate rudder angle and engine throttle
        rudder_angle = -ship_models[-1].rudderang_from_headingref(desired_heading_radians)
        if ship_models[-1].int.time < 20:
            engine_load = 0
        else:
            engine_load = (n - i) / n
        # Update and integrate differential equations for current time step
        ship_models[-1].store_simulation_data(engine_load)
        ship_models[-1].update_differentials(load_perc=engine_load, rudder_angle=rudder_angle)
        ship_models[-1].integrate_differentials()
        ship_models[-1].int.next_time()

    # Store the simulation results in a pandas dataframe
    results.append(pd.DataFrame().from_dict(ship_models[-1].simulation_results))


power_fig, (response_power_ax, response_force_ax) = plt.subplots(2,1)
pow_fig, pow_ax = plt.subplots()

for i in range(0, n):
    power_samples.append(results[i]['power me [kw]'][50*10])
    force_samples.append(results[i]['thrust force [kN]'][50*10])

results[0].plot(x='time [s]', y='power me [kw]', ax=response_power_ax)
results[0].plot(x='time [s]', y='thrust force [kN]', ax=response_force_ax)

results[5].plot(x='time [s]', y='power me [kw]', ax=response_power_ax)
results[5].plot(x='time [s]', y='thrust force [kN]', ax=response_force_ax)

results[10].plot(x='time [s]', y='power me [kw]', ax=response_power_ax)
results[10].plot(x='time [s]', y='thrust force [kN]', ax=response_force_ax)

results[15].plot(x='time [s]', y='power me [kw]', ax=response_power_ax)
results[15].plot(x='time [s]', y='thrust force [kN]', ax=response_force_ax)

pow_ax.scatter(power_samples, force_samples)

pow_ax.set_xlabel('Power')
pow_ax.set_ylabel('Thrust force')





plt.show()
