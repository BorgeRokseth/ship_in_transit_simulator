import sys
import os
from functools import reduce

# allow imports when running script from within project dir
[sys.path.append(i) for i in ['.', '..']]

# allow imports when running script from project dir parent dirs
l = []
script_path = os.path.split(sys.argv[0])
for i in range(len(script_path)):
  sys.path.append( reduce(os.path.join, script_path[:i+1]) )

from models import ShipModel, ShipConfiguration, EnvironmentConfiguration, \
    MachinerySystemConfiguration, SimulationConfiguration, MachineryModes, \
    MachineryMode, MachineryModeParams, HeadingControllerGains, HeadingByReferenceController, \
    EngineThrottleFromSpeedSetPoint, ThrottleControllerGains, SpecificFuelConsumptionWartila6L26, \
    SpecificFuelConsumptionBaudouin6M26Dot3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


main_engine_capacity = 2160e3
diesel_gen_capacity = 510e3
hybrid_shaft_gen_as_generator = 'GEN'
hybrid_shaft_gen_as_motor = 'MOTOR'
hybrid_shaft_gen_as_offline = 'OFF'

time_step = 0.5

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
fuel_curves_me = SpecificFuelConsumptionWartila6L26()
fuel_curves_dg = SpecificFuelConsumptionBaudouin6M26Dot3()
machinery_config = MachinerySystemConfiguration(
    hotel_load=200e3,
    machinery_modes=mso_modes,
    machinery_operating_mode=1,
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
    rudder_angle_to_sway_force_coefficient=50e3,
    specific_fuel_consumption_coefficients_me=fuel_curves_me.fuel_consumption_coefficients(),
    specific_fuel_consumption_coefficients_dg=fuel_curves_dg.fuel_consumption_coefficients()
)
simulation_setup = SimulationConfiguration(
    initial_north_position_m=0,
    initial_east_position_m=0,
    initial_yaw_angle_rad=10 * np.pi / 180,
    initial_forward_speed_m_per_s=7,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    integration_step=time_step,
    simulation_time=300
)

fuel_curves_me = SpecificFuelConsumptionWartila6L26()
fuel_curves_dg = SpecificFuelConsumptionBaudouin6M26Dot3()

ship_model = ShipModel(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=simulation_setup,
                       initial_propeller_shaft_speed_rad_per_s=200 * np.pi / 30)

desired_heading_radians = 45 * np.pi / 180
desired_forward_speed_meters_per_second = 8.5
time_since_last_ship_drawing = 30

# Control system setup
heading_controller_gains = HeadingControllerGains(kp=4, kd=90, ki=0.01)
heading_controller = HeadingByReferenceController(
    gains=heading_controller_gains, time_step=time_step,
    max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180
)
throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=7, ki_ship_speed=0.13, kp_shaft_speed=0.05, ki_shaft_speed=0.005
)
throttle_controller = EngineThrottleFromSpeedSetPoint(
    gains=throttle_controller_gains,
    max_shaft_speed=ship_model.ship_machinery_model.shaft_speed_max,
    time_step=time_step,
    initial_shaft_speed_integral_error=114
)


while ship_model.int.time < ship_model.int.sim_time:
    # "Measure" heading and speed
    heading_measurement_radians = ship_model.yaw_angle
    measured_speed_m_per_s = ship_model.forward_speed
    measured_shaft_speed_rad_per_s = ship_model.ship_machinery_model.omega

    # Find appropriate rudder angle and engine throttle
    rudder_angle = heading_controller.rudder_angle_from_heading_setpoint(
        heading_ref=desired_heading_radians,
        measured_heading=heading_measurement_radians
    )
    throttle = throttle_controller.throttle(
        speed_set_point=desired_forward_speed_meters_per_second,
        measured_speed=measured_speed_m_per_s,
        measured_shaft_speed=measured_shaft_speed_rad_per_s
    )

    # Update and integrate differential equations for current time step
    ship_model.store_simulation_data(throttle)
    ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
    ship_model.integrate_differentials()

    # Make a drawing of the ship from above every 30 second
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
speed_fig, (rpm_ax, speed_ax) = plt.subplots(2, 1)
results.plot(x='time [s]', y='propeller shaft speed [rpm]', ax=rpm_ax)
results.plot(x='time [s]', y='forward speed[m/s]', ax=speed_ax)
eng_fig, (torque_ax, power_ax) = plt.subplots(2, 1)
results.plot(x='time [s]', y='motor torque [Nm]', ax=torque_ax)
results.plot(x='time [s]', y='power me [kw]', ax=power_ax)
plt.show()
