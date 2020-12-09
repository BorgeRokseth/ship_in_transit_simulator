from models import IcebergDriftingModel1, \
    DriftSimulationConfiguration, \
    EnvironmentConfiguration, \
    IcebergConfiguration,\
    Zones,\
    ZonesConfiguration
import pandas as pd
import matplotlib.pyplot as plt
import random

iceberg_config = IcebergConfiguration(
    shape_of_iceberg="tabular",
    coefficient_of_deadweight_to_displacement=0.7,
    size_of_iceberg="medium",
    waterlinelength_of_iceberg=400,
    height_of_iceberg=80,
    width_of_iceberg=16,
    added_mass_coefficient_in_surge=0.4,
    added_mass_coefficient_in_sway=0.4,
    added_mass_coefficient_in_yaw=0.4,
    mass_tonnage=3850000,
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
    wind_direction=39,
    wind_speed=15,
)

zones_config = ZonesConfiguration(
    n_pos=0,
    e_pos=0,
    object_radius=100,
    coll_radius=50,
    excl_radius=500,
    zone1_radius=2000,
    zone2_radius=5000,
    zone3_radius=10000,
)
simulation_config = DriftSimulationConfiguration(
    initial_north_position_m=-50000,
    initial_east_position_m=-20000,
    initial_yaw_angle_rad=0,
    initial_forward_speed_m_per_s=0.5,
    initial_sideways_speed_m_per_s=0.2,
    initial_yaw_rate_rad_per_s=0,
    simulation_time=100000,
    integration_step=5.0
)

iceberg = IcebergDriftingModel1(iceberg_config=iceberg_config,
                                  environment_config=env_config,
                                  simulation_config=simulation_config
                                  )

continue_simulation = True
max_wind_speed = 25
countcol = 0
count_enter_excl = 0

while iceberg.int.time <= iceberg.int.sim_time and continue_simulation:

    #iceberg.wind_speed = random.random() * max_wind_speed


    iceberg.update_differentials()
    iceberg.integrate_differentials()
    countcol = zones_config.colli_event(iceberg.n,iceberg.e)

    iceberg.store_simulation_data()

    if countcol==1:
        continue_simulation = False
        print('Collision occur at: ', iceberg.int.time, 's')
        print("Closest point to Structure:",zones_config.distance(iceberg.n,iceberg.e), 'm')
        countcol = 1

    elif iceberg.n >zones_config.r3+zones_config.n :
        continue_simulation = False
        print('Iceberg passed away the structure and Simulation stopped at: ', iceberg.int.time)
        countcol = 0

    iceberg.int.next_time()

results = pd.DataFrame().from_dict(iceberg.simulation_results)
circle0 = zones_config.plot_coll()
circle1 = zones_config.plot_excl()
circle2 = zones_config.plot_zone1()
circle3 = zones_config.plot_zone2()
circle4 = zones_config.plot_zone3()

fig, axs = plt.subplots(3)
plt.xlim(-100000+zones_config.e, 100000+zones_config.e)
plt.ylim(-100000+zones_config.n, 100000+zones_config.n)
results.plot(x='east position [m]', y='north position [m]', ax=axs[0])
axs[0].set_aspect('equal')
axs[0].add_artist(circle0)
axs[0].add_artist(circle1)
axs[0].add_artist(circle2)
axs[0].add_artist(circle3)
axs[0].add_artist(circle4)
axs[0].set_xlim(-100000+zones_config.e, 100000+zones_config.e)
axs[0].set_ylim(-100000+zones_config.n, 100000+zones_config.n)
results.plot(x='time [s]', y='wind speed [m/sec]', ax=axs[1])
results.plot(x='time [s]', y='wind direction [radius]', ax=axs[2])

plt.show()