# Python3 code to demonstrate
# to find index of first element just
# greater than K
# using map() + index()

# initializing list
test_list = [0.4, 0.5, 11.2, 8.4, 10.4]

# printing original list
print("The original list is : " + str(test_list))

# using map() + index()
# to find index of first element just
# greater than 0.6
res = list(map(lambda i: i > 0.6, test_list)).index(True)

# printing result
print("The index of element just greater than 0.6 : "
      + str(res))

simulation_round = 1
round = 10
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
        exc_point = [dsim.iceberg.simulation_results['time [s]'][exc_idx],
                     dsim.iceberg.simulation_results['north position [m]'][exc_idx],
                     dsim.iceberg.simulation_results['east position [m]'][exc_idx]]
        d_to_zone1 = dsim.distance_results['Distance to zone 1']
        zone1_idx = list(map(lambda i: i <= 0, d_to_zone1)).index(True)
        zone1_point = [dsim.iceberg.simulation_results['time [s]'][zone1_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone1_idx],
                       dsim.iceberg.simulation_results['east position [m]'][zone1_idx]]
        d_to_zone2 = dsim.distance_results['Distance to zone 2']
        zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
        zone2_point = [dsim.iceberg.simulation_results['time [s]'][zone2_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone2_idx],
                       dsim.iceberg.simulation_results['east position [m]'][zone2_idx]]
        d_to_zone3 = dsim.distance_results['Distance to zone 3']
        zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],
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
        exc_point = [dsim.iceberg.simulation_results['time [s]'][exc_idx],
                     dsim.iceberg.simulation_results['north position [m]'][exc_idx],
                     dsim.iceberg.simulation_results['east position [m]'][exc_idx]]
        d_to_zone1 = dsim.distance_results['Distance to zone 1']
        zone1_idx = list(map(lambda i: i <= 0, d_to_zone1)).index(True)
        zone1_point = [dsim.iceberg.simulation_results['time [s]'][zone1_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone1_idx],
                       dsim.iceberg.simulation_results['east position [m]'][zone1_idx]]
        d_to_zone2 = dsim.distance_results['Distance to zone 2']
        zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
        zone2_point = [dsim.iceberg.simulation_results['time [s]'][zone2_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone2_idx],
                       dsim.iceberg.simulation_results['east position [m]'][zone2_idx]]
        d_to_zone3 = dsim.distance_results['Distance to zone 3']
        zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],
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
        zone1_point = [dsim.iceberg.simulation_results['time [s]'][zone1_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone1_idx],
                       dsim.iceberg.simulation_results['east position [m]'][zone1_idx]]
        d_to_zone2 = dsim.distance_results['Distance to zone 2']
        zone2_idx = list(map(lambda i: i <= 0, d_to_zone2)).index(True)
        zone2_point = [dsim.iceberg.simulation_results['time [s]'][zone2_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone2_idx],
                       dsim.iceberg.simulation_results['east position [m]'][zone2_idx]]
        d_to_zone3 = dsim.distance_results['Distance to zone 3']
        zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],
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
        zone2_point = [dsim.iceberg.simulation_results['time [s]'][zone2_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone2_idx],
                       dsim.iceberg.simulation_results['east position [m]'][zone2_idx]]
        d_to_zone3 = dsim.distance_results['Distance to zone 3']
        zone3_idx = list(map(lambda i: i <= 0, d_to_zone3)).index(True)
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],
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
        zone3_point = [dsim.iceberg.simulation_results['time [s]'][zone3_idx],
                       dsim.iceberg.simulation_results['north position [m]'][zone3_idx],
                       dsim.iceberg.simulation_results['east position [m]'][zone3_idx]]
    else:
        col = 0
        exc_breach = 0
        zone1_breach = 0
        zone2_breach = 0
        zone3_breach = 0
    breach_event = [col, exc_breach, zone1_breach, zone2_breach, zone3_breach]

    #print(cpazone, cpa_d, cpa_idx, cpa_time, cpa_loc)
    #print(zone3_point)
    #print(zone2_point)
    #print(zone1_point)
    #print(exc_point)
    #print(col_point)
    #dsim.cpa()
    #print(dsim.breach_event)

    simulation_round += 1
print(dis_matrix)
for dis in dis_matrix:
    distancePlot = plt.plot(dis)
plt.show()
