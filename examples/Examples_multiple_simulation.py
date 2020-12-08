from models import IcebergDriftingModel1, \
    DriftSimulationConfiguration, \
    EnvironmentConfiguration, \
    IcebergConfiguration,\
    Zones,\
    ShipConfiguration,\
    DistanceSimulation,\
    MultiSimulation
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


ship_config=ShipConfiguration(
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

zones_config = Zones(
    n_pos=0,
    e_pos=0,
    object_radius=100,
    coll_radius=50,
    excl_radius=500,
    zone1_radius=2000,
    zone2_radius=5000,
    zone3_radius=10000,
    iceberg_config=iceberg_config
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
dsim = DistanceSimulation(iceberg=iceberg, zones_config=zones_config)
simulation_round = 1
round = 2
cpa_loc = np.empty(2)
dis_matrix = np.empty(round,dtype=object)
#dis_matrix = defaultdict(list)
while simulation_round <= round:
    i=simulation_round-1
    iceberg.restore_to_intial(simulation_config=simulation_config)
    dsim.simulation()
    #print(dsim.distance_results['Distance between iceberg and structure [m]'])
    dis_matrix[simulation_round-1] = dsim.distance_results['Distance between iceberg and structure [m]']

    cpa_d = min(dsim.distance_results['Distance between iceberg and structure [m]'])
    cpa_idx = dsim.distance_results['Distance between iceberg and structure [m]'].index(cpa_d)
    cpa_time = dsim.distance_results['Time [s]'][cpa_idx]
    cpa_loc[0] = dsim.iceberg.simulation_results['north position [m]'][cpa_idx]
    cpa_loc[1] = dsim.iceberg.simulation_results['east position [m]'][cpa_idx]
    cpazone = dsim.zones_config.cpa_zone(cpa_d)
    col_point = np.empty(3)
    exc_point = np.empty(3)
    zone1_point = np.empty(3)
    zone2_point = np.empty(3)
    zone3_point = np.empty(3)

    if cpazone == -1:
        col = 1
        exc_breach = 1
        zone1_breach = 1
        zone2_breach = 1
        zone3_breach = 1
        col_point = [cpa_time, dsim.iceberg.simulation_results['north position [m]'][cpa_idx], dsim.iceberg.simulation_results['east position [m]'][cpa_idx]]
        d_to_exc = dsim.distance_results['Distance to exclusion zone']
        exc_idx = list(map(lambda i: i <= 0, d_to_exc)).index(True)
        exc_point = [dsim.iceberg.simulation_results['time [s]'][exc_idx],\
                     dsim.iceberg.simulation_results['north position [m]'][exc_idx],\
                     dsim.iceberg.simulation_results['east position [m]'][exc_idx]]
        d_to_zone1 = dsim.distance_results['Distance to zone 1']
        zone1_idx = list(map(lambda i: i <= 0, d_to_zone1)).index(True)
        zone1_point = [dsim.iceberg.simulation_results['time [s]'][zone1_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone1_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone1_idx]]
        d_to_zone2 = dsim.distance_results['Distance to zone 2']
        zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
        zone2_point = [dsim.iceberg.simulation_results['time [s]'][zone2_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone2_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone2_idx]]
        d_to_zone3 = dsim.distance_results['Distance to zone 3']
        zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone3_idx]]
    elif cpazone == 0:
        col = 0
        exc_breach = 1
        zone1_breach = 1
        zone2_breach = 1
        zone3_breach = 1
        col_point = np.empty(3)
        d_to_exc = dsim.distance_results['Distance to exclusion zone']
        exc_idx = list(map(lambda i: i <= 0, d_to_exc)).index(True)
        exc_point = [dsim.iceberg.simulation_results['time [s]'][exc_idx],\
                     dsim.iceberg.simulation_results['north position [m]'][exc_idx],\
                     dsim.iceberg.simulation_results['east position [m]'][exc_idx]]
        d_to_zone1 = dsim.distance_results['Distance to zone 1']
        zone1_idx = list(map(lambda i: i <= 0, d_to_zone1)).index(True)
        zone1_point = [dsim.iceberg.simulation_results['time [s]'][zone1_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone1_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone1_idx]]
        d_to_zone2 = dsim.distance_results['Distance to zone 2']
        zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
        zone2_point = [dsim.iceberg.simulation_results['time [s]'][zone2_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone2_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone2_idx]]
        d_to_zone3 = dsim.distance_results['Distance to zone 3']
        zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone3_idx]]
    elif cpazone == 1:
        col = 0
        exc_breach = 0
        zone1_breach = 1
        zone2_breach = 1
        zone3_breach = 1
        col_point = np.empty(3)
        exc_point = np.empty(3)
        d_to_zone1 = dsim.distance_results['Distance to zone 1']
        zone1_idx = list(map(lambda i: i <= 0, d_to_zone1)).index(True)
        zone1_point = [dsim.iceberg.simulation_results['time [s]'][zone1_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone1_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone1_idx]]
        d_to_zone2 = dsim.distance_results['Distance to zone 2']
        zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
        zone2_point = [dsim.iceberg.simulation_results['time [s]'][zone2_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone2_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone2_idx]]
        d_to_zone3 = dsim.distance_results['Distance to zone 3']
        zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone3_idx]]
    elif cpazone == 2:
        col = 0
        exc_breach = 0
        zone1_breach = 0
        zone2_breach = 1
        zone3_breach = 1
        col_point = np.empty(3)
        exc_point = np.empty(3)
        zone1_point = np.empty(3)
        d_to_zone2 = dsim.distance_results['Distance to zone 2']
        zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
        zone2_point = [dsim.iceberg.simulation_results['time [s]'][zone2_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone2_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone2_idx]]
        d_to_zone3 = dsim.distance_results['Distance to zone 3']
        zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone3_idx]]
    elif cpazone == 3:
        col = 0
        exc_breach = 0
        zone1_breach = 0
        zone2_breach = 0
        zone3_breach = 1
        col_point = np.empty(3)
        exc_point = np.empty(3)
        zone1_point = np.empty(3)
        zone2_point = np.empty(3)
        d_to_zone3 = dsim.distance_results['Distance to zone 3']
        zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],\
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],\
                       dsim.iceberg.simulation_results['east position [m]'][zone3_idx]]
    else:
        col = 0
        exc_breach = 0
        zone1_breach = 0
        zone2_breach = 0
        zone3_breach = 0
    breach_event = [col, exc_breach, zone1_breach, zone2_breach, zone3_breach]

    print(cpazone, cpa_d, cpa_idx, cpa_time, cpa_loc)
    print(zone3_point)
    print(zone2_point)
    print(zone1_point)
    print(exc_point)
    print(col_point)
    dsim.cpa()
    print(dsim.breach_event)

    simulation_round += 1
print(dis_matrix)
for dis in dis_matrix:
    distancePlot = plt.plot(dis)
plt.show()

multisim=MultiSimulation(4, dsim)
multisim.multsim()
print(multisim.round_results)
