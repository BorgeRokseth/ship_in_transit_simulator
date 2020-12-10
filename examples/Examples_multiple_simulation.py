from models import IcebergDriftingModel1, \
    DriftSimulationConfiguration, \
    EnvironmentConfiguration, \
    IcebergConfiguration, \
    ZonesConfiguration, \
    Zones, \
    ShipConfiguration, \
    DistanceSimulation, \
    Cost, \
    IceCost,\
    SimulationPools
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

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

z_config = ZonesConfiguration(
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
    integration_step=10.0
)

iceberg = IcebergDriftingModel1(iceberg_config=iceberg_config,
                                environment_config=env_config,
                                simulation_config=simulation_config
                                )
dsim = DistanceSimulation(100, iceberg_config=iceberg_config,
                          simulation_config=simulation_config,
                          environment_config=env_config,
                          z_config=z_config
                          )

ice_cost_config = IceCost(
    disconnect_cost=10,
    light_col_cost=2,
    medium_col_cost=30,
    severe_col_cost=1000,
    towing_cost=1,
    disconnect_time_cost=3600,  # unit is second, equal to 60 minutes.
    towing_time_cost=14400,  # unit is second, equal to 4 hours.
    Ki_lowerbound_severe=50000000,
    Ki_lowerbound_medium=20000000,
    Ki_lowerbound_light=10000000,
)
cost_calculation = Cost(multi_simulation=dsim,
                        ice_cost_config=ice_cost_config,
                        env_config=env_config)

pool_sim = SimulationPools(100, dsim=dsim, cost=cost_calculation)


#dsim.multsim()
#print(dsim.col_pro())
#print(dsim.exc_pro())
#print(dsim.zone1_pro())

pool_sim.pool_sim()
plt.hist(pool_sim.col_prob_list)
plt.hist(pool_sim.exc_prob_list)
plt.hist(pool_sim.zone1_prob_list)

#dsim.multsim()
#print(cost_calculation.cost_msim())

# print(dsim.round_results)
#for dis in dsim.d_zone1_lists:
#    distancePlot = plt.plot(dis)
#plt.show()
#cpa_zonePlot = plt.hist(dsim.round_results['zone of closest point of approach (cpa)'])
#plt.show()

#cpa_d_Plot = plt.hist(dsim.round_results['distance between the closest point of approach (cpa) and the structure'])
#plt.show()
#cpa_t_Plot = plt.hist(dsim.round_results['time when iceberg reaches the closest point of approach (cpa)'])
#plt.show()
#cpa_loc_Plot = plt.plot(dsim.round_results['location of the closest point of approach (cpa)'])
#plt.show()
