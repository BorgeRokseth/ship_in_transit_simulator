from models import ShipModel, ShipConfiguration, EnvironmentConfiguration, \
    MachinerySystemConfiguration, SimulationConfiguration, StaticObstacle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configure the simulation
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
    nonlinear_friction_coefficient__in_yaw=400,

)
env_config = EnvironmentConfiguration(
    current_velocity_component_from_north=0,
    current_velocity_component_from_east=1,
    wind_speed=5,
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
    route_name='route.txt',
    initial_north_position_m=0,
    initial_east_position_m=0,
    initial_yaw_angle_rad=45 * np.pi / 180,
    initial_forward_speed_m_per_s=7,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    initial_propeller_shaft_speed_rad_per_s=400 * np.pi / 30,
    machinery_system_operating_mode=1,
    integration_step=0.5,
    simulation_time=600,
    integral_error_shaft_speed_controller=60,
    integral_error_speed_controller=400
)

ship_model = ShipModel(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=simulation_setup)

desired_forward_speed_meters_per_second = 8.5
time_since_last_ship_drawing = 30

# Place obstacles
obstacle_data = np.loadtxt('obstacles.txt')
list_of_obstacles = []
for obstacle in obstacle_data:
    list_of_obstacles.append(StaticObstacle(obstacle[0],obstacle[1], obstacle[2]))

while ship_model.int.time < ship_model.int.sim_time:
    # Find appropriate rudder angle and engine throttle
    rudder_angle = -ship_model.rudderang_from_route()
    engine_load = ship_model.loadperc_from_speedref(desired_forward_speed_meters_per_second)

    # Update and integrate differential equations for current time step
    ship_model.update_differentials(load_perc=engine_load, rudder_angle=rudder_angle)
    ship_model.integrate_differentials()

    # Store data for the current time step
    ship_model.store_simulation_data(engine_load)

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
map_ax.scatter(ship_model.navigate.east, ship_model.navigate.north, marker='x', color='green')  # Plot the waypoints
for x, y in zip(ship_model.ship_drawings[1], ship_model.ship_drawings[0]):
    map_ax.plot(x, y, color='black')
for obstacle in list_of_obstacles:
    obstacle.plot_obst(ax=map_ax)

map_ax.set_aspect('equal')


# Example on plotting time series
fuel_ifg, fuel_ax = plt.subplots()
results.plot(x='time [s]', y='power [kw]', ax=fuel_ax)

plt.show()