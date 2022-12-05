




The "ship-in-transit-simulator" is a set of 
python classes that can be used to build and run s
imulations  a ship in transit. 

# Simulation model


## Ship
The ship model describes the motion of a ship in three 
degrees of freedom (surge, sway and yaw),
 and contains the 6 states: 
* North position [meters] 
* East position [meters]
* Yaw angle [radians]
* Surge speed [meters per second]
* Sway speed [meters per second]
* Turn rate [radians per second]

The equations of motion are given as: 

![] (https://latex.codecogs.com/gif.latex?%5Cdot%7B%5Cboldsymbol%7Bx%7D%7D%20%3D%20%5Cboldsymbol%7BR%7D_z%28%5Cpsi%29%20%5Cboldsymbol%7Bv%7D)

Additionally, models including machinery system dynamics will have 
in addition on of the following states
* Propeller shaft angular velocity [radians per second] (advanced machinery model)
* Thrust forces [Newton] (simplified machinery system)

The equations of motion includes inertial forces,
forces induced by the added mass effect, 
Coriolis forces, linear and non-linear friction 
forces, environmental forces (described later) and 
control forces for the main propeller and rudder
if a machinery system is included. 

## Machinery system
The ship model has a single propeller shaft and a 
single rudder. The propeller shaft can be powered 
either by the main power source (typically a diesel
engine) or by a hybrid shaft generator. The hybrid 
shaft generator is powered (when running
as a motor) by an electrical distribution. The 
electrical distribution is typically powered by a 
set of diesel generators. 

The simulator offers two models for this propulsion system.
One with propeller shaft dynamics included and one simplified model 
where the thrust force is modelled as a second order transfer 
function. 

The machinery system can be operated in three 
operating modes:
* **PTO (Power Take Out)**: The main power source is 
responsible for both the propulsion load and the
hotel loads. Hotel loads are served via the 
hybrid shaft generator acting as a generator.
* **MEC (Mechanical)**: The main power source is 
only responsible for the propulsion load while the 
electrical distribution serves the hotel loads.
* **PTI (Power Take In)**: The main power source is 
offline and both propulsion and hotel loads are served 
by the electrical distribution. Propulsion power is
provided via the hybrid shaft generator acting as a 
motor. 

The model can be run as a "single engine"-ship by
selecting to use only the main engine. Using PTO-mode, 
the hotel loads and propulsion loads will be served by 
the main power source.  
  
## Navigation
The model offers a heading-by-heading-reference controller 
and a controller that can follow a list of waypoints 
(see provided examples)

## Environmental forces
The ship is subject to wind forces (constant wind speed), 
and current forces (constant current speed).

## Speed control
Speed control can be achieved by:
* The simulator takes a "engine trottle"-signal which 
represents the load percentage on the available power 
for propulsion to be used for generating thrust
* A speed controller that can controls propeller shaft
speed according to ship speed setpoint is provided. 


# Usage


## Installation
Import models.py into your code. 

## Using the simulator
Examples on how to setup and use to simulator can 
be seen in the files under the "examples" folder. 

# Contributors
Børge Rokseth ([borge.rokseth@gmail.com](mailto:borge.rokseth@ntnu.com))

# Licence
This project is published under the [MIT](https://choosealicense.com/licenses/mit/) license.