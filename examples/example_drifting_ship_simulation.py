from models import ShipModelWithoutPropulsion, \
    DriftSimulationConfiguration, \
    EnvironmentConfiguration, \
    ShipConfiguration
import pandas as pd
import matplotlib.pyplot as plt
import random


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
    current_velocity_component_from_north=3,
    current_velocity_component_from_east=2,
    wind_direction=40,
    wind_speed=15
)

simulation_config = DriftSimulationConfiguration(
    initial_north_position_m=0,
    initial_east_position_m=0,
    initial_yaw_angle_rad=0,
    initial_forward_speed_m_per_s=0.5,
    initial_sideways_speed_m_per_s=0.2,
    initial_yaw_rate_rad_per_s=0,
    simulation_time=1000,
    integration_step=5.0
)

ship = ShipModelWithoutPropulsion(ship_config=ship_config,
                                  environment_config=env_config,
                                  simulation_config=simulation_config)
continue_simulation = True
max_wind_speed = 25

time_since_last_ship_drawing = 0


while ship.int.time <= ship.int.sim_time and continue_simulation:

    ship.wind_speed = random.random() * max_wind_speed

    ship.update_differentials()
    ship.integrate_differentials()

    if time_since_last_ship_drawing > 100:
        ship.ship_snap_shot()
        time_since_last_ship_drawing = 0
    time_since_last_ship_drawing += ship.int.dt

    ship.store_simulation_data()

    if ship.n > 1000:
        continue_simulation = False
        print('Simulation stopped at: ', ship.int.time)


    ship.int.next_time()

results = pd.DataFrame().from_dict(ship.simulation_results)
fig, (ax_1, ax_2) = plt.subplots(1, 2)
results.plot(x='east position [m]', y='north position [m]', ax=ax_1)
for x, y in zip(ship.ship_drawings[1], ship.ship_drawings[0]):
    ax_1.plot(x, y, color='black')

ax_1.set_aspect('equal')

results.plot(x='time [s]', y='wind speed [m/sec]', ax=ax_2)



plt.show()