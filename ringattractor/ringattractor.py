"""ROS2 IMPORTS"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from turtlesim.msg import Pose
from vicon_receiver.msg import Position # Ensure you haave the corresponding package for this custom message type
from vicon_receiver.msg import PositionList # Ensure you haave the corresponding package for this custom message type
from ament_index_python.packages import get_package_share_directory

"""Robot actuation imports"""
import RPi.GPIO as GPIO
import Adafruit_PCA9685
import signal

"""Standard imports"""
import time
import math
import json
import random
import threading

"""Scientific imports"""
import numpy as np
from scipy.stats import vonmises
from scipy.integrate import solve_ivp, odeint

"""Logging imports"""
import os
import csv

from numba import njit, prange

# Initialize motor control
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1 = 23
IN2 = 24
IN3 = 27
IN4 = 22
ENA = 0
ENB = 1

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# Shared variable for target heading
target_heading = 0.0
# Flag to indicate if a turn should be performed
perform_turn = False
heading_lock = threading.Lock()
perform_turn_lock = threading.Lock()

# Estimated horizontal field of view (in degrees)
CAMERA_FOV_X = 62  # Raspberry Pi cam or USB webcam typical FOV
cap = None  # Initialize camera capture

#======================define global pose variable=========================

pos_message_g = {}
pos_lock = threading.Lock()


#====================== Helper Functions ======================
# === Helper: find angular distance between two angles ===
def delta_angle(a1, a2):
    """Return the smallest signed difference between two angles in radians."""
    return ((a1 - a2 + np.pi) % (2 * np.pi)) - np.pi

def get_noise(sigma, n):
    return np.random.normal(0, sigma, size=n)

# === Helper: wrap angle to [-pi, pi] ===
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# === Helper: Compute center of mass of neural activation ===
def compute_center_of_mass(z, theta_i):
    sin_sum = np.sum(z * np.sin(theta_i))
    cos_sum = np.sum(z * np.cos(theta_i))
    return np.arctan2(sin_sum, cos_sum)

# === Helper: Computes yaw from quaternion ===
def quaternion_to_yaw(w, x_rot, y_rot, z_rot):
    # Compute yaw (heading) angle in radians
    yaw = math.atan2(2.0 * (w * z_rot + x_rot * y_rot), 1.0 - 2.0 * (y_rot**2 + z_rot**2))
    return yaw

def angle_to_guard_egocentric(agent_pos, agent_heading, guard_pos):
    ax, ay = agent_pos
    gx, gy = guard_pos
    angle_global = np.arctan2(gy - ay, gx - ax)
    return wrap_angle(angle_global)

#======================define parameters=============================
package_name = 'ringattractor'

param_file = os.path.join(
    get_package_share_directory(package_name),
    'parameters.json'
)

with open(param_file, 'r') as f:
    parameters = json.load(f)

class Options():
    def __init__(self):
        self.id = parameters["id"]
        self.guard_ids = np.array(parameters["guard_ids"])
        self.target_ids = np.array(parameters["target_ids"])
        self.num_neurons = int(parameters["num_neurons"])
        self.num_of_targets = int(parameters["num_of_targets"])
        self.num_of_guards = int(parameters["num_of_guards"])
        self.target_qualities = np.array(parameters["target_qualities"])
        self.guard_qualities = np.array(parameters["guard_qualities"])
        self.kappa = float(parameters["kappa"])
        self.v = float(parameters["v"])
        self.u = float(parameters["u"])
        self.beta = float(parameters["beta"])
        self.spatial_decay = float(parameters["spatial_decay"])
        self.sigma = float(parameters["sigma"])
        self.linear_speed = float(parameters["linear_speed"])
        self.angular_speed = float(parameters["angular_speed"])
        self.time_to_compute_dynamics = float(parameters['time_to_compute_dynamics'])

opt = Options()

#====================== Ring Attractor Model ======================

class RingAttractorModel():
    def __init__(self):
        # Number of neurons in the ring attractor
        self.num_neurons = opt.num_neurons
        # Number of targets (e.g., blobs) to track
        self.num_targets = opt.num_of_targets
        # Number of guards (e.g., obstacles)
        self.num_guards = opt.num_of_guards
        # Quality of targets (e.g., blob detection confidence)
        self.target_qualities = opt.target_qualities
        # Quality of guards (e.g., negative influence)
        self.guard_qualities = opt.guard_qualities
        # Concentration (inverse of variance) for von Mises
        self.kappa = opt.kappa
        # # Shape parameter v for the interaction kernel
        self.v = opt.v
        # Coupling strength u for the neural field dynamics
        self.u = opt.u
        # Beta value for the neural field dynamics
        self.beta = opt.beta
        # Spatial decay rate for guard influence
        self.spatial_decay = opt.spatial_decay  
        # Sigma value, for noise
        self.sigma = opt.sigma
        # Neuron preferred angles evenly spaced on the circle
        self.theta_i = np.linspace(-np.pi, np.pi, self.num_neurons, endpoint=False)
        # Sensory map (sensory input vector)
        self.b = np.zeros(self.num_neurons)
        # Interaction kernel
        self.M = np.zeros((self.num_neurons, self.num_neurons))
        self.M = self.compute_interaction_kernel()
        # Neural field state
        self.neural_field = np.zeros(self.num_neurons)

        self.running = True
        

        # Initialize neural activation log
        self.neural_log = []  # Each entry: [timestamp, val1, val2, ..., valN]

        # Initialilize sensory map logging
        self.sensory_log = []  # Each entry: [timestamp, val1, val2, ..., valN]

        self.position_log = []  # Each entry: [timestamp, x, y, heading]

        # Optional: Publisher for bump CoM
        #self.field_pub = self.create_publisher(Float64MultiArray, 'neural_field', 10)
        self.output_received = threading.Event()

    
    def compute_sensory_map(self, target_positions, target_qualities, theta_i, kappa, 
              guard_angles=None, guard_qualities=None, r=None, d=None):
        """
        Computes the sensory input vector b (vectorised).
        """
        # --- Contribution from targets ---
        delta_matrix = delta_angle(self.theta_i[:, np.newaxis], target_positions[np.newaxis, :])  # shape: (num_neurons, num_targets)
        #print('delta_matrix: ', delta_matrix.shape, flush=True)
        vm_targets = np.exp(kappa * (np.cos(delta_matrix) - 1))     # (num_neurons, num_targets)
        #print('vm_targets: ', vm_targets.shape, flush=True)

        # Weighted sum over targets
        self.b = vm_targets @ target_qualities   # (num_neurons,)
        #print('vm_targets @ target_qualities: ', self.b.shape, flush=True)
        # --- Inhibition from guards ---
        if guard_angles is not None and guard_qualities is not None and r is not None and d is not None:
            delta_guards = delta_angle(self.theta_i[:, np.newaxis], guard_angles[np.newaxis, :])  # shape: (num_neurons, num_guards)  # wrap & abs
            #print('delta_guards: ', delta_guards.shape, flush=True)
            vm_guards = np.exp(kappa * (np.cos(delta_guards) - 1))             # (num_neurons, num_guards)
            #print('vm_guards: ', vm_guards.shape, flush=True)    
            # Each guard contributes -gamma * exp(-r*d)
            #print('r, d: ', r, d, flush=True)
            s_guards = guard_qualities * np.exp(-r * d)                # (num_guards,)
            #print('s_guards: ', s_guards, flush=True)
            # Weighted sum over guards
            weighted_sum = vm_guards @ s_guards
            #print('vm_guards @ s_guards: ', weighted_sum.shape, flush=True)
            self.b += weighted_sum   # (num_neurons,)
            #print('b after guards: ', self.b.shape, flush=True)

            
        # --- Normalization ---
        self.b /= np.sqrt(self.num_neurons)
        #print('Final b: ', self.b.shape, flush=True)
        #return b
    
    def compute_interaction_kernel(self):
        """
        Computes the interaction kernel M, i,e. Compute m_ij for all pairs (i, j).

        Returns:
            M (np.array): Interaction kernel, shape (num_neurons, num_neurons)
        """
        # theta_i: shape (num_neurons,)
        theta_i_col = self.theta_i[:, np.newaxis]  # shape (num_neurons, 1)
        theta_i_row = self.theta_i[np.newaxis, :]  # shape (1, num_neurons)

        # Compute pairwise delta_ij matrix: shape (num_neurons, num_neurons)
        delta_ij = np.abs(delta_angle(theta_i_col, theta_i_row))

        # Now compute M using the vectorized formula
        return ((1 / self.num_neurons) * np.cos(np.pi * (delta_ij / np.pi) ** self.v))

    @staticmethod
    @njit(fastmath=True)
    def randn_like(y, sigma, inv_sqrt_n):
        out = np.empty_like(y)
        for i in prange(y.size):
            out[i] = np.random.normal(0.0, sigma) * inv_sqrt_n
        return out

    @staticmethod
    @njit(fastmath=True)
    def dynamics(t, y, u, b, M, beta, n, sigma, randn_like_func):
        """ Computes the dynamics of the ring attractor model."""
        # Noise vector (same shape as y)
        #noise = (sigma / n) * np.random.randn(*y.shape)
        #noise = np.random.normal(0, sigma, size=y.shape) / np.sqrt(n)  # Scale noise by sqrt(n)
        noise = randn_like_func(y, sigma, 1.0 / np.sqrt(n))
        dydt = -y + np.tanh(u * M @ y + b - beta) - np.tanh(-beta) + noise 
        #print("y.shape:", y.shape, "dydt.shape:", dydt.shape)
        # Dynamics for z
        return dydt
        
    @njit
    def euler_integrate(y0, t_eval, u, b, M, beta, n, sigma, randn_like_func):
        dt = t_eval[1] - t_eval[0]
        y = np.zeros((len(t_eval), len(y0)))
        y[0] = y0
        for i in range(1, len(t_eval)):
            # Scale noise by sqrt(dt) for proper SDE approximation
            noise = randn_like_func(y[i-1], sigma * np.sqrt(dt), 1.0 / np.sqrt(n))
            dydt = -y[i-1] + np.tanh(u * M @ y[i-1] + b - beta) - np.tanh(-beta) + noise
            y[i] = y[i-1] + dt * dydt
        return y
        

    # Integrate timesteps to simulate the neural field dynamics
    def compute_dynamics(self, total_time=200, dt=0.1):
        t_eval = np.arange(0, total_time, dt)
        y0 = np.random.randn(self.num_neurons) * 0.1
        
        """result = solve_ivp(
            fun=lambda t, y: RingAttractorModel.dynamics(t, y, self.u, self.b, self.M, self.beta, self.num_neurons, self.sigma, RingAttractorModel.randn_like),
            t_span=(0, total_time),
            y0=y0,
            t_eval=t_eval,
            method='LSODA',       # use faster method
            rtol=1e-2,             # relax tolerance for speed
            atol=1e-4,  
            vectorized=False # Enable vectorization for efficiency
        )
        
        self.neural_field = result.y.T  # Final state
        times = result.t"""

        """result = odeint(
            func=lambda y, t: RingAttractorModel.dynamics(t, y, self.u, self.b, self.M, self.beta, self.num_neurons, self.sigma, RingAttractorModel.randn_like),
            y0=y0,
            t=t_eval,
            rtol=1e-1,  # Looser relative tolerance
            atol=1e-3   # Looser absolute tolerance
        )
        self.neural_field = result  # Already shaped as (len(t_eval), num_neurons)
        times = t_eval  # Time points are just the input t_eval"""
        self.neural_field = RingAttractorModel.euler_integrate(y0, t_eval, self.u, self.b, self.M, self.beta, self.num_neurons, self.sigma, RingAttractorModel.randn_like)
        times = t_eval
        
    
        # Compute CoM of bump activity at each time
        bump_positions = np.array([compute_center_of_mass(z_t, self.theta_i) for z_t in self.neural_field])
    
        final_norm = np.linalg.norm(self.neural_field[-1])
        return times, bump_positions, final_norm

    def save_ring_log(self):
        log_path = os.path.expanduser("~/neural_activation_log_stationary.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["target"] + ["guard"] + [f"bump_positions"] + [f"neuron_{i}" for i in range(self.num_neurons)]
            writer.writerow(header)
            writer.writerows(self.neural_log)

    def save_sensory_log(self):
        log_path = os.path.expanduser("~/sensory_map_log_stationary.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["target"] + [f"neuron_{i}" for i in range(self.num_neurons)]
            writer.writerow(header)
            writer.writerows(self.sensory_log)

    def save_position_log(self):
        log_path = os.path.expanduser("~/position_log_stationary.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["x"] + ["y"] + ["heading"] + ["x_target"] + ["y_target"] + ["heading_target"]
            writer.writerow(header)
            writer.writerows(self.position_log)

    def run(self):
        """
        Run the model for a specified number of timesteps.
        This method integrates the neural field dynamics and logs the state.
        """
        global pos_message_g  # Use the global position message
        global target_heading
        global perform_turn
        pos_message = {}

        while self.running:
            # Copy the global position message to local variable
            with pos_lock:
                pos_message = pos_message_g.copy()
            
            guard_angles = None
            guard_agent_dist = None
            target_angles = None
            # --------------------- Compute target angles ---------------------
            if self.num_targets != len(opt.target_ids):
                print(f"Error: num_targets {self.num_targets} does not match length of target_ids {len(opt.target_ids)}", flush=True)
                return
            else:
                for i, target_id in enumerate(opt.target_ids):
                    if target_id in pos_message and 'self' in pos_message:
                        angle_to_target = angle_to_guard_egocentric(
                            (pos_message['self'][0], pos_message['self'][1]),
                            pos_message['self'][2],
                            (pos_message[target_id][0], pos_message[target_id][1])
                        )
                        if target_angles is None:
                            target_angles = np.array([angle_to_target])
                        else:
                            target_angles = np.append(target_angles, angle_to_target)
                    else:
                        if target_id not in pos_message:
                            print(f"Warning: {target_id} missing from pos_message: {pos_message.keys()}", flush=True)
                            angle_to_target = 0.0  # Default to 0 if no target
                        if 'self' not in pos_message:
                            print(f"Warning: 'self' missing from pos_message: {pos_message.keys()}", flush=True)
                            angle_to_target = 0.0  # Default to 0 if no self
                        if target_angles is None:
                            target_angles = np.array([angle_to_target])
                        else:
                            target_angles = np.append(target_angles, angle_to_target)
            # -----------------------------------------------------------------------------

            # --------------------- Compute guard angles and distance ---------------------
            if self.num_guards != len(opt.guard_ids):
                print(f"Error: num_guards {self.num_guards} does not match length of guard_ids {len(opt.guard_ids)}", flush=True)
                return
            else:
                for i, guard_id in enumerate(opt.guard_ids):
                    if guard_id in pos_message and 'self' in pos_message:
                        # ------- Compute guard angles and distance to guard if available -------
                        angle_to_guard = angle_to_guard_egocentric(
                            (pos_message['self'][0], pos_message['self'][1]),
                            pos_message['self'][2],
                            (pos_message[guard_id][0], pos_message[guard_id][1])
                        )
                        if guard_angles is None:
                            guard_angles = np.array([angle_to_guard])
                        else:
                            guard_angles = np.append(guard_angles, angle_to_guard)
                        # -----------------------------------------------------------------------

                        # ------- Compute distance to guard -------
                        guard_agent_dist = math.sqrt( (pos_message[guard_id][0] - pos_message['self'][0]) ** 2 \
                                            + (pos_message[guard_id][1] - pos_message['self'][1]) ** 2 )
                        guard_agent_dist /= 1000.0  # convert to meters
                        # -----------------------------------------------------------------------
                    else:
                        if guard_id not in pos_message:
                            print(f"Warning: {guard_id} missing from pos_message: {pos_message.keys()}", flush=True)
                            angle_to_guard = 0.0  # Default to 0 if no guard
                        if 'self' not in pos_message:
                            print(f"Warning: 'self' missing from pos_message: {pos_message.keys()}", flush=True)
                            angle_to_guard = 0.0  # Default to 0 if no self
                        if guard_angles is None:
                            guard_angles = np.array([angle_to_guard])
                        else:
                            guard_angles = np.append(guard_angles, angle_to_guard)  
                # -----------------------------------------------------------------------------         

            if target_angles is None:
                target_angles = np.array([0.0])  # Default to 0 if no target, should not happen once here already
            # Compute sensory input vector b
            self.compute_sensory_map(
                            target_positions=target_angles,
                            target_qualities=self.target_qualities,
                            theta_i=self.theta_i,
                            kappa=self.kappa,
                            guard_angles=guard_angles,
                            guard_qualities=self.guard_qualities,  # Guard quality
                            r=self.spatial_decay,  # Spatial decay rate
                            d=guard_agent_dist  # Distance to guard
                        )
            
            #print("Running Ring Attractor Model...")
            start_time = time.time()
            times, bump_positions, final_norm = self.compute_dynamics(total_time=opt.time_to_compute_dynamics)
            # Log current time and neural field state
            #now = self.get_clock().now().nanoseconds / 1e9  # seconds
        
            end_time = time.time()
            

            # Store heading over time
            new_heading = 0.0  # one per time step

            # Loop over time steps
            norm_z = np.linalg.norm(self.neural_field[-1])

            new_heading = compute_center_of_mass(self.neural_field[-1], self.theta_i)

            # Log the neural state
            self.neural_log.append([target_angles, guard_angles, norm_z] + self.neural_field[-1].tolist())
            self.save_ring_log()

            # Log the sensory map
            self.sensory_log.append([target_angles] + self.b.tolist())
            self.save_sensory_log()

            # Log the position
            for i, target_id in enumerate(opt.target_ids):
                if target_id in pos_message and 'self' in pos_message:
                    #
                    self.position_log.append([pos_message['self'][0]] + [pos_message['self'][1]] + [pos_message['self'][2]] \
                                        + [pos_message[target_id][0]] + [pos_message[target_id][1]] + [pos_message[target_id][2]])  
                else:
                    self.position_log.append([pos_message['self'][0]] + [pos_message['self'][1]] + [pos_message['self'][2]])


            """if opt.target_id in pos_message and 'self' in pos_message:
                self.position_log.append([pos_message['self'][0]] + [pos_message['self'][1]] + [pos_message['self'][2]] \
                                        + [pos_message[opt.target_id][0]] + [pos_message[opt.target_id][1]] + [pos_message[opt.target_id][2]])
            else:
                self.position_log.append([pos_message['self'][0]] + [pos_message['self'][1]] + [pos_message['self'][2]])"""

            self.save_position_log()

            with heading_lock:
                target_heading = new_heading
            
            with perform_turn_lock:
                perform_turn = True
            # Signal that at least one pose has been received
            self.output_received.set()
            print(f"Model run completed in {end_time - start_time:.2f} seconds. Current robot heading in degrees: {math.degrees(pos_message['self'][2]):.2f} New target heading in degrees: {math.degrees(new_heading):.2f} rad, Final norm: {final_norm:.2f}", flush=True)

            

    def stop(self):
        """Stop the ring attractor."""
        self.running = False

#====================== Vicon Subscriber Node ======================
class ViconSubscriber(Node):
    """Node to subscribe to Vicon position updates."""
    def __init__(self):
        super().__init__('vicon_subscriber')
        self.id = opt.id
        self.subscription = self.create_subscription(
            PositionList,
            "vicon/default/data",
            self.listener_callback,
            10)

        self.pose_received = threading.Event()  # Event to signal first pose received


    def listener_callback(self, msg):
        with pos_lock:
            global pos_message_g 
            pos_message_g= {}
        for i in range(msg.n):
            if msg.positions[i].subject_name == self.id:
                # Update the current position with the received data
                with pos_lock:
                    pos_message_g['self'] = (float(msg.positions[i].x_trans), float(msg.positions[i].y_trans), 
                                            float( msg.positions[i].z_rot_euler))
            else:
                with pos_lock:
                    pos_message_g[msg.positions[i].subject_name] = (float(msg.positions[i].x_trans), float(msg.positions[i].y_trans), 
                                                                float(msg.positions[i].z_rot_euler))
        self.pose_received.set()  # Signal that at least one pose has been received
                # For debugging, print the received position
                #self.get_logger().info('subject "%s" with segment %s:' %(msg.positions[i].subject_name, msg.positions[i].segment_name))
                #self.get_logger().info('I heard translation in x, y, z: "%f", "%f", "%f"' % (msg.positions[i].x_trans, msg.positions[i].y_trans, msg.positions[i].z_trans))
                #self.get_logger().info('I heard rotation in x, y, z, w: "%f", "%f", "%f", "%f": ' % (msg.positions[i].x_rot, msg.positions[i].y_rot, msg.positions[i].z_rot, msg.positions[i].w))



class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, signum, frame):
    self.kill_now = True


#====================== Robot Actuation Functions ======================
""" Functions to control the robot's motors using GPIO and PCA9685.
These functions include setting custom speeds, stopping the car, calculating speeds based on linear and angular velocities,
and turning to a specific heading."""
def custom_speed(speed_left, speed_right):
    # Set motor directions and speeds
    if speed_left < 0:
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN1, GPIO.HIGH)
        speed_left = -speed_left
    else:
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN1, GPIO.LOW)

    if speed_right < 0:
        GPIO.output(IN4, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        speed_right = -speed_right
    else:
        GPIO.output(IN4, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)

    pwm.set_pwm(ENA, 0, int(speed_left))
    pwm.set_pwm(ENB, 0, int(speed_right))

def stopcar():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm.set_pwm(ENA, 0, 0)
    pwm.set_pwm(ENB, 0, 0)

""" Helper function to calculate motor speeds based on linear and angular velocities. """
def get_pwm(V_R, V_L):

    d_0 = 1200
    d_max = 4095

    V_max = 0.5 # maximum tangential velocity (m/s) 

    V_R = abs(V_R)
    V_L = abs(V_L)

    V_sat_R = min(V_R, V_max)
    V_sat_L = min(V_L, V_max)

    alpha_R = 1
    alpha_L = 1
    if V_R > 0:
        alpha_R = V_sat_R / V_R
    if V_L > 0:
        alpha_L = V_sat_L / V_L

    alpha = min(alpha_L, alpha_R)

    V_scaled_R = V_sat_R * alpha
    V_scaled_L = V_sat_L * alpha

    v_R = V_scaled_R / V_max
    v_L = V_scaled_L / V_max

    pwm_R = 0
    pwm_L = 0

    if v_R > 0:
        pwm_R = d_0 + (d_max - d_0) * v_R
    if v_L > 0:
        pwm_L = d_0 + (d_max - d_0) * v_L

    return pwm_L, pwm_R



# def calculate_speed(v_lin, v_ang):
#     left_motor = 4.899 * v_lin - 384.44 * v_ang
#     right_motor = 4.892 * v_lin + 427.5061 * v_ang
#     print(f'Calculated motor speeds: left {left_motor}, right {right_motor}')

#     return left_motor, right_motor

def calculate_speed(v_lin, v_ang):  # v_lin (m/s), v_ang (rad/s)

    d = 0.143    # distance btw wheels (m)
    r = 0.035   # wheel radius (m)

    

    V_R = v_lin + d/2 * v_ang   # (m/s)
    V_L = v_lin - d/2 * v_ang   # (m/s)

    sign_R = 1
    sign_L = 1

    if V_R < 0:
        sign_R = -1
    if V_L < 0:
        sign_L = -1

    pwm_L, pwm_R = get_pwm(V_R, V_L)

    pwm_L *= sign_L
    pwm_R *= sign_R

    return pwm_L, pwm_R
    



def move_robot(v_lin, v_ang):
    
    v_left, v_right = calculate_speed(v_lin, v_ang)
    custom_speed(v_left, v_right)

def turn_to_heading(target_heading):
    # Parameters
    v_lin = 0.0  # No forward motion, only rotation
    v_ang = 3.0  # Fixed angular velocity (rad/s)
    max_motor_speed = 4095  # Max PWM value for PCA9685 (12-bit)

    # Calculate time to turn
    angle_to_turn = abs(target_heading)  # Magnitude of rotation needed
    turn_time = angle_to_turn / abs(v_ang)  # Time = angle / angular speed

    # Determine turn direction
    if target_heading < 0:
        v_ang = -v_ang  # Negative for clockwise turn

    # Calculate motor speeds
    v_left, v_right = calculate_speed(v_lin, v_ang)

    # Limit motor speeds to valid PWM range
    v_left = max(min(v_left, max_motor_speed), -max_motor_speed)
    v_right = max(min(v_right, max_motor_speed), -max_motor_speed)

    #print(f"Turning {target_heading:.2f} rad, v_ang: {v_ang:.2f} rad/s, time: {turn_time:.2f} s")
    #print(f"Left motor: {v_left:.2f}, Right motor: {v_right:.2f}")

    # Actuate motors
    start_time = time.time()
    custom_speed(v_left, v_right)

    # Run for calculated duration
    while time.time() - start_time < turn_time:
        time.sleep(0.01)  # Small delay to prevent blocking

    # Stop the robot
    stopcar()
    #print("Turn complete")

def move_forward():
    # Parameters for straight motion
    v_lin = 200.0  # Constant linear velocity (m/s)
    v_ang = 0.0  # No rotation
    max_motor_speed = 4095  # Max PWM value
    move_duration = 2.0  # Move forward for 2 seconds

    # Calculate motor speeds
    v_left, v_right = calculate_speed(v_lin, v_ang)

    # Limit motor speeds to valid PWM range
    v_left = max(min(v_left, max_motor_speed), -max_motor_speed)
    v_right = max(min(v_right, max_motor_speed), -max_motor_speed)

    #print(f"Moving forward with v_lin: {v_lin:.2f} m/s")
    #print(f"Left motor: {v_left:.2f}, Right motor: {v_right:.2f}")

    # Actuate motors
    start_time = time.time()
    custom_speed(v_left, v_right)

    # Run for fixed duration
    while time.time() - start_time < move_duration:
        time.sleep(0.01)  # Small delay to prevent blocking

    # Stop the robot
    stopcar()
    #print("Forward motion complete")
#====================== PID Controller Class ======================
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-1.0, 1.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = 0
        self.output_limits = output_limits

    def compute(self, measured_value, dt):
        error = self.setpoint - measured_value
        self.integral += error * dt
        # Anti-windup: Clamp integral
        self.integral = np.clip(self.integral, -1.0 / self.Ki if self.Ki else 0, 1.0 / self.Ki if self.Ki else 0)
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        # Clamp output
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        self.prev_error = error
        return output

def robot_motion():
    """Thread function for robot motion."""
    killer = GracefulKiller()

    global target_heading
    global pos_message_g

    angular_pid = PID(
        Kp=0.5, 
        Ki=0.1, 
        Kd=0.2, 
        setpoint=0, 
        output_limits=(-1.0, 1.0)
        )
    
    dt = 0.1  # Control loop time step

    try:
        while not killer.kill_now:
            if 'self' in pos_message_g:
                """with pos_lock:
                    current_heading = pos_message_g['self'][2]
                
                
                # handle angle wrapping (PID doesn’t know about 2π wrap-around)
                dif = target_heading - current_heading
                error = wrap_angle(dif)
                # instead of feeding current_heading directly,
                # feed the error so PID "thinks" measured=error, setpoint=0
                #angular_v_pid = controller(-error)

                #print('Error (rad): ', abs(error), flush=True)

                v = 0.1 * np.cos(error - current_heading)
                w = 0.5 * np.sin(error - current_heading)

                angular_v_motor = 0.1 * error #angular_v_pid * 30
                                
                # Optional: edge case handling if needed
                if abs(error) > math.radians(15):#math.pi/12:  # too far off
                    linear_v = 0
                else:
                    linear_v = 0.1

                # Send commands
                move_robot(linear_v, angular_v_motor)"""

                # In your control loop (assuming dt is time since last update, e.g., 0.1s)
                with pos_lock:
                    current_heading = pos_message_g['self'][2]

                # Handle angle wrapping
                dif = target_heading - current_heading
                error = wrap_angle(dif)

                # Compute angular velocity using PID (setpoint=0, measured=-error to correct towards 0)
                angular_v_motor = angular_pid.compute(-error, dt)  # dt from loop timing

                # Smooth linear velocity: Scale with cos(error) for gradual reduction
                linear_v = 0.1 * np.cos(error)  # Ranges from 0.1 (error=0) to 0 (error=±90deg)
                linear_v = max(linear_v, 0.01)  # Optional min speed to avoid full stop

                # Optional: Further reduce linear_v for very large errors
                if abs(error) > math.radians(90):
                    linear_v = 0

                # Send commands
                move_robot(linear_v, angular_v_motor)

            else:
                set_speed(0, 0)

    except KeyboardInterrupt:
        print("Stopping robot motion")
    finally:
        stopcar()
        GPIO.cleanup()

def main(args=None):
    rclpy.init()
    vicon_node = ViconSubscriber()

    # Use a multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(vicon_node)

    # Start executor in a background thread
    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()
    print("ViconSubscriber spinning...", flush=True)

    # Wait until we have received at least one pose
    print("Waiting for first Vicon pose...", flush=True)
    vicon_node.pose_received.wait()   # blocks until first pose callback
    print("Pose received, starting control threads!", flush=True)

    # Now start your other threads
    ring_attractor = RingAttractorModel()
    ring_thread = threading.Thread(target=ring_attractor.run, daemon=True)
    #motion_thread = threading.Thread(target=robot_motion, daemon=True)

    ring_thread.start()
    
    # Wait until we have received at least one pose
    print("Waiting for first ring attractor output...", flush=True)
    ring_attractor.output_received.wait()   # blocks until first pose callback
    print("Ringattractor output received, starting control threads!", flush=True)
    
    robot_motion()  # Run in main thread for better signal handling
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down")
        ring_attractor.stop()
    finally:
        stopcar()
        GPIO.cleanup()
        executor.shutdown()
        vicon_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
