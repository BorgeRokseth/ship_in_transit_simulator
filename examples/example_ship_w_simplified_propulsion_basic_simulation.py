from models import ShipModelSimplifiedPropulsion, ShipConfiguration, EnvironmentConfiguration, \
    SimplifiedPropulsionMachinerySystemConfiguration ,SimplifiedPropulsionSimulationConfiguration, \
    MachineryModes, MachineryMode, MachineryModeParams
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

machinery_config = SimplifiedPropulsionMachinerySystemConfiguration(
    hotel_load=200e3,
    machinery_modes=mso_modes,
    max_rudder_angle_degrees=30,
    rudder_angle_to_yaw_force_coefficient=500e3,
    rudder_angle_to_sway_force_coefficient=50e3,
    thrust_force_dynamic_time_constant=30
)

simulation_setup = SimplifiedPropulsionSimulationConfiguration(
    route_name='none',
    initial_north_position_m=0,
    initial_east_position_m=0,
    initial_yaw_angle_rad=10 * np.pi / 180,
    initial_forward_speed_m_per_s=0,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    initial_thrust_force=0,
    machinery_system_operating_mode=1,
    integration_step=0.5,
    simulation_time=500
)

ship_model = ShipModelSimplifiedPropulsion(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=simulation_setup)

desired_heading_radians = 45 * np.pi / 180
desired_forward_speed_meters_per_second = 7
time_since_last_ship_drawing = 30

while ship_model.int.time < ship_model.int.sim_time:
    # Find appropriate rudder angle and engine throttle
    rudder_angle = -ship_model.rudderang_from_headingref(desired_heading_radians)
    if ship_model.int.time < 100:
        engine_load = 0
    else:
        engine_load = ship_model.loadperc_from_speedref(desired_forward_speed_meters_per_second)

    # Update and integrate differential equations for current time step
    ship_model.store_simulation_data(engine_load)
    ship_model.update_differentials(load_perc=engine_load, rudder_angle=rudder_angle)
    ship_model.integrate_differentials()

    # Store data for the current time step
    # ship_model.store_simulation_data(engine_load)

    # Make a drawing of the ship from above every 20 second
    if time_since_last_ship_drawing > 30:
        ship_model.ship_snap_shot()
        time_since_last_ship_drawing = 0
    time_since_last_ship_drawing += ship_model.int.dt
    # Progress time variable to the next time step
    ship_model.int.next_time()

# Store the simulation results in a pandas dataframe
results = pd.DataFrame().from_dict(ship_model.simulation_results)

# Example on how a map-view can be generated
map_fig, map_ax = plt.subplots()
map_ax.plot(results['east position [m]'], results['north position [m]'])
for x, y in zip(ship_model.ship_drawings[1], ship_model.ship_drawings[0]):
    map_ax.plot(x, y, color='black')
map_ax.set_aspect('equal')

# Example on plotting time series
speed_fig, speed_ax = plt.subplots()
results.plot(x='time [s]', y='forward speed[m/s]', ax=speed_ax)

force_response_fig, (power_ax, force_response_ax) = plt.subplots(2,1)
results.plot(x='time [s]', y='power me [kw]', ax=power_ax)
results.plot(x='time [s]', y='thrust force [kN]', ax=force_response_ax)

load_fig, (load_ax, int_ax) = plt.subplots(2,1)
results.plot(x="time [s]", y='commanded load fraction me [-]', ax=load_ax)
results.plot(x="time [s]", y='speed controller integrator', ax=int_ax)
plt.show()
