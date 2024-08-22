"""

Frenet optimal trajectory generator

Written By: Atsushi Sakai (@Atsushi_twi)
Modified By: Yide Tao

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

############################################
# Section 1: Import the Necessary Libraries#
############################################
# Deal with the System OS
import sys
import os

# Set the directory root as the local directory
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

# Pip Libraries
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import pandas as pd
# Classes from python robotics: https://github.com/AtsushiSakai/PythonRobotics
from QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from CubicSpline import cubic_spline_planner


######################################
# Section 2: Define Helper Functions #
######################################      
def generate_target_course(x, y):
        """
        Author: Python Robotics
        Description: Generate Cublic Spline Between two points
        Last Modified: 20th Jul 2024 
        """
        csp = cubic_spline_planner.CubicSpline2D(x, y)
        s = np.arange(0, csp.s[-1], 0.1)

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))

        return rx, ry, ryaw, rk, csp
    
def find_projection_point(wx, wy, x, y):
    """
    Author: ChatGPT
    Description: Project x and y coordinate of the vehicle onto the reference path
    Last Modified: 20th Jul 2024 
    """
    # Calculate distances from (x, y) to each point on the path
    distances = np.hypot(wx - x, wy - y)
    # Find the index of the closest point
    min_idx = np.argmin(distances)
    return min_idx, wx[min_idx], wy[min_idx]


def calculate_frenet_coordinates(wx, wy, x, y):
    """
    Author: ChatGPT
    Description: Project x and y coordinate of the vehicle onto the reference path
    Last Modified: 20th Jul 2024 
    """
    # Find the projection point
    min_idx, x_p, y_p = find_projection_point(wx, wy, x, y)
    # Longitudinal coordinate s (sum of distances along the path)
    s = np.sum(np.hypot(np.diff(wx[:min_idx+1]), np.diff(wy[:min_idx+1])))
    # Lateral coordinate d (perpendicular distance to the path)
    dx = x - x_p
    dy = y - y_p
    yaw = np.arctan2(wy[min_idx+1] - wy[min_idx], wx[min_idx+1] - wx[min_idx])
    d = np.hypot(dx, dy) * np.sign(dx * np.cos(yaw) + dy * np.sin(yaw))
    return s, d

class ObstacleManager:
    """
    Author: Yide
    Description: This is the class for defining a list of obstacles.
    Last Modified: 20th Jul 2024 
    """
    def __init__(self, 
                 obs_list: np.array = np.array([])): # A list of n by 3 numpy arrays. 

        # Check if the list of obstacle inputed is correc
        if obs_list.size == 0:
            self.obsticle_arr = np.array([])
        elif obs_list.shape[1] != 3:
            raise ValueError('The obstacles in the list has more than 3 values (x, y r, ?)')
        else:
            self.obsticle_arr = obs_list
    
    # Add New Obsticles to the List
    def add_obs(self, newobs):
        """
        Author: Yide
        Description: This Function adds either:
            1. a list of obsticles of format [x1,y1,r1,x2,y2,r2,x3,y3,r3...] to the existing obsticle manager.
            2. a numpy array of shape n by 3 to the obsticle manager.
        Last Modified: 20th Jul 2024 
        """
        # Check if the New Obsticles added is in the List format
        if isinstance(newobs, list):
            if len(newobs) % 3 != 0:
                raise ValueError('List length must be a multiple of 3 (x, y, r)')
            try:
                new_obs_array = np.array(newobs).reshape(-1, 3)
                self.obsticle_arr = np.vstack((self.obstacle_list, new_obs_array))
            except Exception as e:
                raise ValueError('Error converting list to array: ' + str(e))
        # Check if the New Obsticles added is in the Numpy
        elif isinstance(newobs, np.ndarray):
            if newobs.shape[1] != 3:
                raise ValueError('Array must have shape (n, 3) for obstacles')
            self.obsticle_arr = np.vstack((self.obstacle_list, newobs))
        else:
        # If not in both, raise Type Error.
            raise TypeError('Add obs function can only accept a list of length 3n or a (n, 3) array')
    
    # Return the Obsticle List.
    def show_obs(self):
        """
        Description: THe function will return a list of obstacles
        """
        return self.obstacle_list
    
    def vehicle_obs(self, vehicle: 'VehicleController'):
        """
        Author: Yide
        Description: This function calculates the global coordinates of the circular representation
        of the vehicle and adds them to the obstacle list. (This code need to be simplified using rotational matrix if have time)
        Last Modified: 20th Jul 2024 
        """
        r = vehicle.vehicle_cir_r  # Define the radius of the obstacles
        obs_len = len(vehicle.vehicle_cir_relx)  # Define the number of circles
        
        self.obstacle_arr = np.zeros([obs_len, 3])  # Define the empty obstacle array
        self.obstacle_arr[:, 2] = r  # Set all of the last column to a constant radius
        
        # Apply Transformation
        for i in range(obs_len):
            relx_global = vehicle.vehicle_cir_relx[i] 
            self.obstacle_arr[i, 0] = relx_global * np.cos(vehicle.yaw) + vehicle.x
            self.obstacle_arr[i, 1] = relx_global * np.sin(vehicle.yaw) + vehicle.y
        
        return self.obstacle_arr

class QuarticPolynomial:
    """
    Author: Python Robotics
    Description: Quartic Polynomial Function used for the Frenet Frame
    Last Modified: 20th Jul 2024 
    """
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

#######################################
# Section 3: Define the Vehicle Class #
#######################################      

class VehicleController:
    """
    Author: Yide
    Description: This is the class for the Vehicles around the course.
    Last Modified: 20th Jul 2024 
    """
    
    def __init__(self, 
                 width:float, # Width of the Vehicle
                 length:float, # Length of the Vehicle
                 x:float = 0, # x coord of the Vehicle
                 y:float = 0, # y coord of the Vehicle
                 yaw:float = 0, # the vehicle's yaw
                 lin_v:float = 0): # Linear Velocity of the vehicle Note: Needed to create dynamic obsticle predictions.
        # Vehicle Location
        self.x = x
        self.y = y
        self.yaw = yaw

        # Vehicle Body
        self.vehicle_width = width
        self.vehicle_length = length
        self.vehicle_cir_relx, self.vehicle_cir_r = self.__convert_cir_represent__()
        self.vehicle_cir_n = len(self.vehicle_cir_relx)
        self.vehicle_obs_arr = self.__generate_obs_xy__()
        
        
        # Vehicle Kinematics
        self.lin_v = lin_v
        self.__input_check__()

    def change_position(self, x, y, yaw):
        """
        Author: Yide
        Description: This function plot will change the position of the Vehicle
        Last Modified: 20th Jul 2024
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vehicle_obs_arr = self.__generate_obs_xy__()

    def drive_step(self, dt, v = None):
        if v != None:
            self.lin_v = v
        self.x = self.x + dt*self.lin_v*np.cos(self.yaw)
        self.y = self.y + dt*self.lin_v*np.sin(self.yaw)
        self.change_position(self.x, self.y, self.yaw)

    def plot_vehicle(self, plt):
        """
        Author: Yide
        Description: This function plots the vehicle using its circular representation. (Mostly For Testing)
        Last Modified: 20th Jul 2024
        """
        #print(self.vehicle_obs_arr)
        for circle in self.vehicle_obs_arr:
            circle_plot = plt.Circle((circle[0], circle[1]), circle[2], color='b', alpha=0.5)
            plt.gca().add_patch(circle_plot)
    
    def calculate_body_xy(self, x, y, yaw, i):
        """
        Author: Yide
        Description: This function calculates the xy position of the one body (circle) of the vehicle
        Last Modified: 20th Jul 2024
        """
        body_xy = np.zeros(2)
        if i < self.vehicle_cir_n and i >= 0:
            relx_global = self.vehicle_cir_relx[i] 
            body_xy[0] = relx_global * np.cos(yaw) + x
            body_xy[1] = relx_global * np.sin(yaw) + y
            return body_xy
        else:
            raise ValueError('Index provided is not suitable for computation')

    def __convert_cir_represent__(self):
        """
        Author: Yide
        Description: This function covert a vehicle representation from square to circular representations.
        Note that this code is not that efficient you can probably optimize it if you want.
        Last Modified: 20th Jul 2024 
        """
        body_circles_list =[] 
        robot_radius = 0.5*(pow((2*self.vehicle_width**2),0.5)) 
        n = np.ceil(self.vehicle_length/self.vehicle_width)
        d = (n*self.vehicle_width - self.vehicle_length)/(2*(n-1))
        for i_c in range(int(n)):
            i = i_c + 1
            d_i = d/(n-1)
            body_rel_pos = self.vehicle_width*i - 2*d_i*(i) - self.vehicle_width/2 - self.vehicle_length/2
            body_circles_list.append(body_rel_pos)
        return body_circles_list, robot_radius
    
    def __generate_obs_xy__(self):
        """
        Author: Yide
        Description: This function takes in the x, y, yaw of the vehicle as well as the circular representation of the vehicle, 
        using them it caculates the circular representation of the vehicle.
        Last Modified: 20th Jul 2024 
        """
        vehicle_obs_mg = ObstacleManager()
        vehicle_obs_arr = vehicle_obs_mg.vehicle_obs(self)
        #print(vehicle_obs_arr)
        return vehicle_obs_arr
    
    def __input_check__(self):
        if self.vehicle_width > self.vehicle_length:
            raise ValueError('The Length of the vehicle can not be shorter than the Width of the vehicle')
        
    

##############################################################
# Section 4: Define the Quintic Planner (Frenet Frame) Class #
##############################################################

class FrenetPath:
    """
    Author: Python Robotics
    Modified: Yide
    Description: The Base Variable Storage Class For Frenet Frame
    Last Modified: 20th Jul 2024 
    """
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = [] # Jerk
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = [] # Jerk
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []




class FrenetPathPlanner():
    """
    Author: Python Robotics
    Modified: Yide
    Description: The Planner Class for computing the Frenet Path of the Vehicle using the Quintic Polynomical Method
    Last Modified: 20th Jul 2024 
    """
    def __init__(self, 
                CONTROL_VEHICLE :'VehicleController', #  the vehicle being currently controlled
                MAX_SPEED=5,  # maximum speed [m/s]
                MAX_ACCEL=2.87,  # maximum acceleration [m/ss]
                MAX_YAW = 10,
                MAX_CURVATURE=30.0,  # maximum curvature [1/m]
                MAX_ROAD_WIDTH=2,  # maximum road width [m]
                DT=0.1,  # time tick [s]
                MAX_T=3.0,  # max prediction time [m]
                MIN_T=2.0,  # min prediction time [m]
                TARGET_SPEED=5,  # target speed [m/s]
                D_T_S=10,  # target speed sampling length [m/s]
                N_S_SAMPLE=5,  # sampling number of target speed
                K_J=1,  # weight for jerk
                K_T=1,  # weight for time
                K_D=1,  # weight for distance
                K_LAT=1.0,  # weight for lateral error
                K_LON=1.0):  # weight for longitudinal error
        # Define the vehicle to be controlled
        self.vehicle = CONTROL_VEHICLE
        # Define the Planner Tunning Variable
        self.MAX_SPEED = MAX_SPEED
        self.MAX_ACCEL = MAX_ACCEL
        self.MAX_YAW_RATE =  MAX_YAW
        self.MAX_NACCEL = -7.06
        self.MAX_CURVATURE = MAX_CURVATURE
        self.MAX_ROAD_WIDTH = MAX_ROAD_WIDTH
        self.D_ROAD_W = self.MAX_ROAD_WIDTH/10
        self.DT = DT
        self.MAX_T = MAX_T
        self.MIN_T = MIN_T
        self.TARGET_SPEED = TARGET_SPEED
        self.D_T_S = D_T_S
        self.N_S_SAMPLE = N_S_SAMPLE
        self.K_J = K_J
        self.K_T = K_T
        self.K_D = K_D
        self.K_LAT = K_LAT
        self.K_LON = K_LON
        

        # Define the initial state
        self.c_speed = 0.0 # current speed [m/s]
        self.c_accel = 0.0 # current accerleration [m/ss]
        self.c_d = 0.0 # current lateral position [m]
        self.c_d_d = 0.0 # current lateral speed [m/s]
        self.c_d_dd = 0.0 # current lateral accerlation [m/s]
        self.s0 = 0.0 # current course position
        self.yaw0 = 0.0
        
        #Set True
        self.FIRST_RUN = False

        # Define the path:
        self.frenet_path = FrenetPath()
        
        # Helper Variables
        self.previous_stop_exist = False
        self.vehicle_stopped = False
    
    def set_initial_variables(self, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, yaw0 = 0.326795):
        """
        Author: Yide Tao
        Description: Function for Setting the Initial Variables for the Planner
        Last Modified: 20th Jul 2024 
        """
        self.c_speed = c_speed # Initial Speed 
        self.c_accel = c_accel # Initial Accerlation
        self.c_d = c_d # Initial Lateral Position 
        self.c_d_d = c_d_d # Initial Lateral Speed 
        self.c_d_dd = c_d_dd # Initial Lateral Accleration
        self.s0 = s0 # Initial Course Position
        self.yaw0 = yaw0
        self.FIRST_RUN = True
        return None

    def set_speed_limit(self, speed_limit:float):
        """
        Author: Yide Tao
        Description: Function for change the speed limit
        Last Modified: 20th Jul 2024 
        """
        # speed_limit: km/hr
        # Convert the speed limit from km/h to m/s
        self.MAX_SPEED = speed_limit
        return None
    
    def set_road_width(self, road_width:float):
        """
        Author: Yide Tao
        Description: Function for change the Road Width
        Last Modified: 20th Jul 2024 
        """
        self.MAX_ROAD_WIDTH = road_width
        return None

    def plan(self, csp, ob, lead_exist=False, lead_d = 0, lead_v = 0, stop_exist = False, stop_d = 0):
        """
        Author: Yide Tao
        Description: Function for Initiating One iteration of the Planning
        Last Modified: 20th Jul 2024 
        """

        # Detect for change in stopping condition
        '''
        if self.previous_stop_exist == False and stop_exist == True:
            self.TARGET_STOP = stop_d + self.s0 # Calculate the Target Stopping Position
            print('Current Position:' + str(self.s0) +', Target Stop: ' + str(self.TARGET_STOP))
        '''
        self.MIN_STOP_DISTANCE = self.c_speed**2/(abs(self.MAX_NACCEL)*0.7*2)
        self.MAX_STOP_DISTANCE = self.MAX_SPEED**2/(abs(self.MAX_NACCEL)*0.7*2)

        # Check for Lead Vehicle
        if lead_exist:
            self.TARGET_LEAD = lead_d + self.s0 
            self.TARGET_LOC = (self.TARGET_LEAD-self.vehicle.vehicle_length/2) - self.DT*lead_v - self.c_speed*3
            print('Distance Between Vehicles: '+str(lead_d))
            # Lead Vehicle too close Initiating ADS
            if lead_d < self.MAX_STOP_DISTANCE + self.vehicle.vehicle_width/2 + 5:
                print('Initiating ADS!!!')
                stop_exist = True
                stop_d = lead_d + self.vehicle.vehicle_width/2 + self.MAX_SPEED 
            # Lead Vehicle has sufficient space for Normal Stopping.
            elif abs(lead_v) < self.MAX_SPEED/5 and lead_d > self.MAX_STOP_DISTANCE + self.vehicle.vehicle_width/2:
                print('Lead Vehicle STOP Detected, Initiating Normal Stopping!')
                stop_exist = True
                stop_d = lead_d + self.vehicle.vehicle_width/2 
            # Lead Vehicle Too Far Away
            elif lead_d > self.MAX_SPEED*6 and self.c_speed > self.MAX_SPEED/5: 
                print('Lead Vehicle Too Far Away, initiating Cruise Control')
                lead_exist = False
        # Check if the stop command is provided:
        if stop_exist:
            print('Initiating Stopping')
            self.TARGET_STOP = stop_d + self.s0
            self.STOP_RATIO = self.MIN_STOP_DISTANCE/stop_d
            #print('Stop Ratio:'+str(self.STOP_RATIO))
            #print('Speed:'+str(self.c_speed))
            # Ensure vehicle is actually stopping
            if self.c_speed < self.MAX_SPEED/self.MAX_T:
                    self.vehicle_stopped = True
                    #print('Vehicle Stopped!!!')
        else:
            print('Initializing Cruise Control.')
            self.TARGET_SPEED = self.MAX_SPEED
        # Check if a lead Vehicle Exist:
        
        path = self.__frenet_optimal_planning__(csp, self.s0, self.c_speed, 
                                                self.c_accel, self.c_d, self.c_d_d, self.c_d_dd, ob, 
                                                lead_exist, lead_v,
                                                stop_exist)
        # If a path exist plan for a path 
        if self.FIRST_RUN:
            path.yaw[0] = self.yaw0
            path.d_d[1] = -0.005
            path.d_dd[1] = 0
            self.FIRST_RUN = False
        try:
            self.s0 = path.s[1]
            self.yaw0 = path.yaw[1]
            self.c_d = path.d[1]
            self.c_d_d = path.d_d[1]
            self.c_d_dd = path.d_dd[1]
            self.c_speed = path.s_d[1]
            self.c_accel = path.s_dd[1]
            #print('Vehicle yaw: ' + str(path.yaw))
            self.vehicle.change_position(path.x[1], path.y[1], path.yaw[1])
            #self.previous_stop_exist = stop_exist
        except:
        # Stop the Vehicle if collision or near collision has been detected
            print('Collision or Near Collision Detected!!!')
            if path is not None:
                path.x.append(self.vehicle.x)
                path.y.append(self.vehicle.y)
                path.yaw.append(self.yaw0)
                path.s.append(self.s0)
                path.d.append(self.c_d)
                path.d_d.append(0)
                path.d_dd.append(0)
                path.s_d.append(0)
                path.s_dd.append(0)
                
                # Append the next position with zero velocity and acceleration
                next_x = self.vehicle.x + self.c_speed * self.DT * math.cos(self.vehicle.yaw)
                next_y = self.vehicle.y + self.c_speed * self.DT * math.sin(self.vehicle.yaw)
                path.x.append(next_x)
                path.y.append(next_y)
                path.yaw.append(self.yaw0)
                path.s.append(self.s0 + self.c_speed * self.DT)
                path.d.append(self.c_d)
                path.d_d.append(0)
                path.d_dd.append(0)
                path.s_d.append(0)
                path.s_dd.append(0)
            else:
                # Create a new path if no path exists
                path = FrenetPath()
                path.x = [self.vehicle.x]
                path.y = [self.vehicle.y]
                path.yaw = [self.vehicle.yaw]
                path.s = [self.s0]
                path.d = [self.c_d]
                path.d_d = [0]
                path.d_dd = [0]
                path.s_d = [0]
                path.s_dd = [0]
                
                # Append the next position with zero velocity and acceleration
                next_x = self.vehicle.x + self.c_speed * self.DT * math.cos(self.vehicle.yaw)
                next_y = self.vehicle.y + self.c_speed * self.DT * math.sin(self.vehicle.yaw)
                path.x.append(next_x)
                path.y.append(next_y)
                path.yaw.append(self.vehicle.yaw)
                path.s.append(self.s0 + self.c_speed * self.DT)
                path.d.append(self.c_d)
                path.d_d.append(0)
                path.d_dd.append(0)
                path.s_d.append(0)
                path.s_dd.append(0)
                
            self.c_speed = 0
            self.c_accel = 0

        return path

    def __frenet_optimal_planning__(self, csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob, 
                                    lead_exist, lead_v,# Parameters for vehicle following
                                    stop_exist): # Parameters for vehicle stopping
        """
        Author: Python Robotics
        Description: Function for Frenet Generation
        Last Modified: 20th Jul 2024 
        """
        fplist = self.__calc_frenet_paths__(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0,
                                            lead_exist, lead_v, # Parameters for vehicle following
                                            stop_exist) 
        fplist = self.__calc_global_paths__(fplist, csp)
        fplist = self.__check_paths__(fplist, ob)

        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for fp in fplist:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp

        return best_path
  
    def __calc_frenet_paths__(
            self, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, 
            lead_exist, lead_v,
            stop_exist): 
                            
            """
            Author: Python Robotics
            Description: Function for simulating the Frenet Path based on the Vehicle
            Last Modified: 20th Jul 2024 
            """
            frenet_paths = []
            road_width = (self.MAX_ROAD_WIDTH/2)-self.vehicle.vehicle_width/2
            if lead_exist:
                road_width = self.vehicle.vehicle_width/2
                print('road_width: '+str(road_width))
            for di in np.arange(-road_width, road_width, 
                                self.D_ROAD_W):
                for Ti in np.arange(self.MIN_T, self.MAX_T, self.DT):
                    
                    lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                    fp = FrenetPath()
                    fp.t = [t for t in np.arange(0.0, Ti, self.DT)]
                    fp.d = [lat_qp.calc_point(t) for t in fp.t]
                    fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                    fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                    fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                    for tv in np.arange(self.TARGET_SPEED - self.D_T_S * self.N_S_SAMPLE,
                                        self.TARGET_SPEED + self.D_T_S * self.N_S_SAMPLE, self.D_T_S):
                        
                        
                        
                        lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)
                        tfp = FrenetPath()
                        tfp.t = fp.t
                        tfp.d = fp.d
                        tfp.d_d = fp.d_d
                        tfp.d_dd = fp.d_dd
                        tfp.d_ddd = fp.d_ddd

                        tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                        tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                        tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                        tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                        Jp = sum(np.power(tfp.d_ddd, 2))
                        Js = sum(np.power(tfp.s_ddd, 2))
                        
                        # Stop Condition
                        if stop_exist:
                            ds = (self.TARGET_STOP-tfp.s[1]-self.vehicle.vehicle_length/2)**2 + self.STOP_RATIO*(tfp.s_d[-1])**2
                            tfp.cv = self.K_J * Js + self.K_T * Ti + self.K_D* ds 
                            if self.vehicle_stopped:
                                self.STOP_RATIO = 100
                                ds = self.STOP_RATIO*(tfp.s_d[1])**2
                                tfp.s_d = [0 for _ in fp.t]
                                tfp.d_d = [0 for _ in fp.t]
                                
                            
                        # Vehicle Following Condition        
                        elif lead_exist:
                            self.vehicle_stopped = False
                            ds =  (tfp.s[1] - self.TARGET_LOC)**2
                            dv = (lead_v -  tfp.s_d[1])**2
                            tfp.cv = self.K_J * Js + self.K_T * Ti + 2*self.K_D*ds + self.K_D*dv
                        
                        # Velocity Keeping condition
                        else:
                            self.vehicle_stopped = False
                            ds = (self.TARGET_SPEED - tfp.s_d[-1])**2 # Constant speed
                            tfp.cv = self.K_J * Js + self.K_T * Ti + self.K_D * ds

                        tfp.cd = self.K_J * Jp + self.K_T * Ti + self.K_D * tfp.d[-1] ** 2
                        tfp.cf = self.K_LAT * tfp.cd + self.K_LON * tfp.cv

                        frenet_paths.append(tfp)
            return frenet_paths
    

    def __calc_global_paths__(self, fplist, csp):
        """
        Author: Python Robotics
        Description: Function for Generating the Global Path
        Last Modified: 20th Jul 2024 
        """
        for fp in fplist:
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = csp.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                if self.vehicle_stopped:
                    fp.x.append(self.vehicle.x)
                    fp.y.append(self.vehicle.y)
                else:
                    fp.x.append(fx)
                    fp.y.append(fy)
                

            # calc yaw and ds #
            if len(fp.x) > 1:  # Ensure there are enough points to calculate yaw
                for i in range(len(fp.x) - 1):
                    dx = fp.x[i + 1] - fp.x[i]
                    dy = fp.y[i + 1] - fp.y[i]
                    if dx == 0 and dy == 0 or self.vehicle_stopped:
                        fp.yaw.append(self.vehicle.yaw)
                    else:
                        fp.yaw.append(math.atan2(dy, dx))
                    fp.ds.append(math.hypot(dx, dy))
            

                fp.yaw.append(fp.yaw[-1])
                fp.ds.append(fp.ds[-1])
            
            # If there are not enough points, append default values
            else:
                
                fp.yaw.append(self.yaw0)
                fp.ds.append(0.0)

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / (fp.ds[i] + 0.01))

        return fplist

   
    def __check_collision__(self, fp, ob):
        ob_positions = ob[:, :2]
        ob_radii_squared = (self.vehicle.vehicle_cir_r + ob[:, 2]) ** 2
        #print(ob_radii_squared)
        for j in range(self.vehicle.vehicle_cir_n):
            ix = np.array([self.vehicle.calculate_body_xy(fp.x[k], fp.y[k], fp.yaw[k], j)[0] for k in range(len(fp.x))])
            iy = np.array([self.vehicle.calculate_body_xy(fp.x[k], fp.y[k], fp.yaw[k], j)[1] for k in range(len(fp.x))])
            distances_squared = (ob_positions[:, 0, np.newaxis] - ix) ** 2 + (ob_positions[:, 1, np.newaxis] - iy) ** 2

            if np.any(distances_squared < ob_radii_squared[:, np.newaxis]):
                return False

        return True

    
    def __check_paths__(self, fplist, ob):
        ok_ind = []
        for i, fp in enumerate(fplist):
            yaw_rate_ok = all(abs((fp.yaw[j + 1] - fp.yaw[j]) / 0.1) <= self.MAX_YAW_RATE for j in range(len(fp.yaw) - 1))
            if (any(v > self.MAX_SPEED for v in fp.s_d) or
                any(a > self.MAX_ACCEL or a < self.MAX_NACCEL for a in fp.s_dd) or
                any(abs(c) > self.MAX_CURVATURE for c in fp.c) or
                not yaw_rate_ok or
                not self.__check_collision__(fp, ob)):
                continue
            ok_ind.append(i)
        #print(self.__check_collision__(fp, ob))

        # If our tentacle collides with the vehicle we should 

        return [fplist[i] for i in ok_ind]
    
class AVDynamicPlanning:

    def __init__(self, route_file: str):

        self.initial = False
        # way points
        path_raw = pd.read_csv(route_file)
        path_np = path_raw.to_numpy()
        path_np[0,0]= 124.7523096154207
        path_np[0,1]= 15.917477574911015
        path_np = path_np[::100]
        self.wx = path_np[:,0]
        self.wy = path_np[:,1]
        # driving Vehicle:
        vehicle_drive = VehicleController(1.2, 5, 124.7523096154207, 15.917477574911015, 0.326795)
        # obstacles
        newVehicle = VehicleController(1.2, 5, -100, -100, np.pi/4)
        self.obstacles = {"id": newVehicle}
        self.ob_av_info = np.vstack([newVehicle.vehicle_obs_arr])
        
        # Generate Target Course. Only thing we care about is the course itself
        _, _, _, _, self.csp = generate_target_course(self.wx, self.wy)
        self.Planner = FrenetPathPlanner(vehicle_drive)
        # initial state
        c_speed = 0.0  # current speed [m/s]
        c_accel = 0.0  # current acceleration [m/ss]
        c_d = 0.0  # current lateral position [m]
        c_d_d = 0.0  # current lateral speed [m/s]
        c_d_dd = 0.0  # current lateral acceleration [m/s]
        s = 0
        self.Planner.set_initial_variables(
            c_speed,
            c_accel,
            c_d,
            c_d_d,
            c_d_dd,
            s)

    def step(self, av_state, av_context_info):
        # Iterate through each vehicle and change its x and y position
        for id, bvehicle in av_context_info.items():
            if id in self.obstacles:
                self.obstacles[id].change_position(bvehicle["x"], bvehicle["y"], bvehicle["orientation"])
            else:
                newVehicle = VehicleController(bvehicle["width"], bvehicle["length"], bvehicle["x"], bvehicle["y"], bvehicle["orientation"])
                self.obstacles[id] = newVehicle
                self.ob_av_info = np.vstack(self.ob_av_info, newVehicle.vehicle_obs_arr)
        path = self.Planner.plan(self.csp, self.ob_av_info)
        #print('x: '+str(path.x[1])+',y: '+str(path.y[1])+',yaw: '+str(path.yaw[1]))
        return path.x[0], path.y[0], path.yaw[0], self.Planner.c_speed
    

