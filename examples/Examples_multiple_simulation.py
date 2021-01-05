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
    SimulationPools,\
    PlotEverything,\
    EntropyCalculation,\
    DataFit
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.stats as st
from math import log, e

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
    initial_north_position_m=-20000,
    initial_east_position_m=-8500,
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
dsim = DistanceSimulation(40, iceberg_config=iceberg_config,
                          simulation_config=simulation_config,
                          environment_config=env_config,
                          z_config=z_config
                          )

ice_cost_config = IceCost(
    disconnect_cost=10,
    light_col_cost=5,
    medium_col_cost=300,
    severe_col_cost=1000,
    towing_cost=1,
    disconnect_time_cost=3600,  # unit is second, equal to 60 minutes.
    towing_time_cost=14400,  # unit is second, equal to 4 hours.
    Ki_lowerbound_severe=50000000,
    Ki_lowerbound_medium=20000000,
    Ki_lowerbound_light=5000000,
)
cost_calculation = Cost(multi_simulation=dsim,
                        ice_cost_config=ice_cost_config,
                        env_config=env_config)
zone = Zones(z_config=z_config, iceberg_config=iceberg_config)
pool_sim = SimulationPools(10, dsim=dsim, cost=cost_calculation)
datafit = DataFit()
plotall = PlotEverything()
pool_sim.pool_sim()

data = np.asarray(pool_sim.cpa_d_list)
#DataFit.get_best_distribution(data=labels)
#best_dis = DataFit.get_best_distribution(data=labels)[0]
#params = DataFit.get_best_distribution(data=labels)[2]
#print(DataFit.get_entropy(dist_name=best_dis, parameters=params))
#print(st.entropy(pool_sim.cpa_d_list))
plotall.plot_icebergpos_zones(zone=zone, sim=pool_sim)
print(data.mean())
print(data.var())
datafit.data_fit_comparision(data=data)
#plt.show()
#ax=plt.hist(data, bins=50, density=True, label='Data')
# Find best fit distribution
#best_fit_name, best_fit_params = DataFit.best_fit_distribution(data, 50, ax=None)
#best_dist = getattr(st, best_fit_name)

# Update plots
#ax.set_ylim(dataYLim)
#ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
#ax.set_xlabel(u'Temp (°C)')
#ax.set_ylabel('Frequency')

# Make PDF with best params
#pdf = DataFit.make_pdf(best_dist, best_fit_params)
#print(DataFit.get_entropy(best_dist, best_fit_params))
# Display
#plt.figure(figsize=(12, 8))
#plt.hist(data, bins=50, density=True, label='Data')
#plt.legend()
#ax = pdf.plot(lw=2, label='PDF', legend=True)
#param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
#param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
#dist_str = '{}({})'.format(best_fit_name, param_str)
#ax.set_title('CPA with best fit distribution \n' + dist_str)
#ax.set_xlabel('CPA')
#ax.set_ylabel('Probability Density')

#plt.show()

