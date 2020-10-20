##ship-in-transit-simulator
The "ship-in-transit-simulator" is a set of python classes that can be used to build and run simulations  a ship in transit. 

##The simulation model
### Ship
The ship model describes the motion of a ship three 
degrees of freedom (surge, sway and yaw),
 and contains the following 7 states: 
* North position [meters] 
* East position [meters]
* Yaw angle [radians]
* Surge speed [meters per second]
* Sway speed [meters per second]
* Turn rate [radians per second]
* Propeller shaft angular velocity [radians per second]

The equations of motion includes inertial forces,
forces induced by the added mass effect, 
Coriolis forces, linear and non-linear friction 
forces, environmental forces (described later) and 
control forces from the main propeller and rudder. 

### Machinery system
The ship model has a single propeller shaft and a 
single rudder. The propeller shaft can be powered 
either by the main power source (typically a diesel
engine) or by a hybrid shaft generator. The hybrid 
shaft generator is powered (when running
as a motor) by an electrical distribution. The 
electrical distribution is typically powered by a 
set of diesel generators. 

The dynamics of the propeller shaft is modelled. The 
power source and the hybrid shaft generator is modelled
as input sources (and can be set as a input signal 
(load percentage) or determined by the speed 
controller (to be discussed later).

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
selecting to use only MSO-mode 1 (PTO). The the 
hotel loads and propulsion loads will be served by 
the main power source.  
  
### Navigation
Navigation can be achieved in one of three ways: 
* Providing a rudder angle signal 
* Providing a set point signal for the built-in 
heading controller
* Providing a text file specifying a route to 
 follow (described later). 
### Environmental forces
The ship is subject to wind forces (constant wind speed), 
and current forces (constant current speed).

### Speed control
Speed control can be achieved by:
* Providing a signal representing the load (0-1) 
on the available power sources.
* Using the built-in speed controller to generate
a load signal. 

### Installation
Import models.py into your code. 

### Using the simulator
Examples on how to setup and use to simulator can 
be seen in the files under the "examples" folder. 

