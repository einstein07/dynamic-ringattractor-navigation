"""ROS2 IMPORTS"""
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from vicon_receiver.msg import Position # Ensure you haave the corresponding package for this custom message type
from vicon_receiver.msg import PositionList # Ensure you haave the corresponding package for this custom message type
from ament_index_python.packages import get_package_share_directory

"""Robot actuation imports"""
import RPi.GPIO as GPIO
import Adafruit_PCA9685

"""Standard imports"""
import time
import math
import json
import random
import threading

"""Scientific imports"""
import numpy as np
from scipy.stats import vonmises
from scipy.integrate import solve_ivp

"""Logging imports"""
import os
import csv
import cv2

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
    return wrap_angle(angle_global - agent_heading)

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
        self.guard_id = parameters["guard_id"]
        self.target_id = parameters["target_id"]
        self.corner1 = (float(parameters["corner1"][0]), float(parameters["corner1"][1]))  # limit moving area x, y
        self.corner2 = (float(parameters["corner2"][0]), float(parameters["corner2"][1]))
        #self.rot_index = int(parameters["rot_index"])  # message index of rotation angle
        self.update_time = float(parameters["update_time"])  # time for updating current position in seconds
        self.minimal_dist = float(parameters["minimal_dist"])  # distance defining two car too close
        self.away_scale = float(parameters["away_scale"])  # amount of going away when two cars are too close
        self.angular_speed = float(parameters["angular_speed"])
        self.rotate_scale = float(parameters['rotate_scale'])

opt = Options()

#====================== Ring Attractor Model ======================

class RingAttractorModel():
    def __init__(self, num_neurons=100, num_targets=1, target_quality=1.0, guard_quality=-15.0, kappa=20.0, v=0.5, u=1.0, beta=1.0, spatial_decay=2.0, sigma=0.01, update_interval=1.0):
        # Number of neurons in the ring attractor
        self.num_neurons = num_neurons
        # Number of targets (e.g., blobs) to track
        self.num_targets = num_targets
        # Quality of targets (e.g., blob detection confidence)
        self.target_quality = np.array([target_quality])
        # Quality of guards (e.g., negative influence)
        self.guard_quality = np.array([guard_quality])
        # Concentration (inverse of variance) for von Mises
        self.kappa = kappa
        # # Shape parameter v for the interaction kernel
        self.v = v
        # Coupling strength u for the neural field dynamics
        self.u = u
        # Beta value for the neural field dynamics
        self.beta = beta
        # Spatial decay rate for guard influence
        self.spatial_decay = spatial_decay  
        # Sigma value, for noise
        self.sigma = sigma
        # Neuron preferred angles evenly spaced on the circle
        self.theta_i = np.linspace(-np.pi, np.pi, self.num_neurons, endpoint=False)
        # Sensory map (sensory input vector)
        self.b = np.zeros(self.num_neurons)
        # Interaction kernel
        self.M = np.zeros((self.num_neurons, self.num_neurons))
        self.M = self.compute_interaction_kernel()
        # Neural field state
        self.neural_field = np.zeros(self.num_neurons)

        # Update interval for target heading
        self.update_interval = update_interval  # Time between heading updates (seconds)
        self.running = True
        

        # Initialize neural activation log
        self.neural_log = []  # Each entry: [timestamp, val1, val2, ..., valN]

        # Optional: Publisher for bump CoM
        #self.field_pub = self.create_publisher(Float64MultiArray, 'neural_field', 10)

    
    def compute_sensory_map(self, target_positions, target_qualities, theta_i, kappa, 
              guard_angles=None, guard_qualities=None, r=None, d=None):
        """
        Computes the sensory input vector b (vectorised).
        """
        n = len(theta_i)

        # --- Contribution from targets ---
        # Shape: (n, k)
        delta_targets = theta_i[:, None] - target_positions[None, :]
        delta_targets = (delta_targets + np.pi) % (2*np.pi) - np.pi  # wrap
        vm_targets = np.exp(kappa * (np.cos(delta_targets) - 1))     # (n, k)

        # Weighted sum over targets
        b = vm_targets @ target_qualities   # (n,)

        # --- Inhibition from guards ---
        if guard_angles is not None and guard_qualities is not None and r is not None and d is not None:
            # Shape: (n, m)
            delta_guards = theta_i[:, None] - guard_angles[None, :]
            delta_guards = np.abs((delta_guards + np.pi) % (2*np.pi) - np.pi)  # wrap & abs
            vm_guards = np.exp(kappa * (np.cos(delta_guards) - 1))             # (n, m)

            # Each guard contributes -gamma * exp(-r*d)
            s_guards = -np.array(guard_qualities) * np.exp(-r * d)                # (m,)

            b += vm_guards @ s_guards   # (n,)

        # --- Normalization ---
        b /= np.sqrt(n)
        return b
    
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

        

    def dynamics(self, t, y, u, b, M, beta, n, sigma):
        """ Computes the dynamics of the ring attractor model."""
        # Noise vector (same shape as y)
        #noise = (sigma / n) * np.random.randn(*y.shape)
        noise = np.random.normal(0, sigma, size=y.shape) / np.sqrt(n)  # Scale noise by sqrt(n)
        dydt = -y + np.tanh(u * M @ y + b - beta) - np.tanh(-beta) + noise 
        #print("y.shape:", y.shape, "dydt.shape:", dydt.shape)
        # Dynamics for z
        return dydt
        
        

    # Integrate timesteps to simulate the neural field dynamics
    def compute_dynamics(self, total_time=200, dt=0.1):
        t_eval = np.arange(0, total_time, dt)
        y0 = np.random.randn(self.num_neurons) * 0.1
        
        result = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y, self.u, self.b, self.M, self.beta, self.num_neurons, self.sigma),
            t_span=(0, total_time),
            y0=y0,
            t_eval=t_eval,
            method='DOP853',       # use faster method
            rtol=1e-2,             # relax tolerance for speed
            atol=1e-4,  
            vectorized=False # Enable vectorization for efficiency
        )
        
        self.neural_field = result.y.T  # Final state
        times = result.t
    
        # Compute CoM of bump activity at each time
        bump_positions = np.array([compute_center_of_mass(z_t, self.theta_i) for z_t in self.neural_field])
    
        final_norm = np.linalg.norm(self.neural_field[-1])
        return times, bump_positions, final_norm

    def save_log(self):
        log_path = os.path.expanduser("~/neural_activation_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["time"] + [f"bump_positions"] + [f"neuron_{i}" for i in range(self.num_neurons)]
            writer.writerow(header)
            writer.writerows(self.neural_log)

    def run(self):
        """
        Run the model for a specified number of timesteps.
        This method integrates the neural field dynamics and logs the state.
        """
        global pos_message_g  # Use the global position message
        global target_heading
        global perform_turn
        pos_message = {}
        # Copy the global position message to local variable
        with pos_lock:
            pos_message = pos_message_g.copy()
        while self.running:
            
            guard_angles = None
            guard_agent_dist = None
            target_angles = None

            if opt.guard_id in pos_message and 'self' in pos_message:
                """ Compute guard angles and distance to guard if available """

                guard_agent_dist = math.sqrt( (pos_message[opt.guard_id][0] - pos_message['self'][0]) ** 2 \
                                    + (pos_message[opt.guard_id][1] - pos_message['self'][1]) ** 2 )
                
                angle_to_guard = angle_to_guard_egocentric(
                    (pos_message['self'][0], pos_message['self'][1]),
                    pos_message['self'][2],
                    (pos_message[opt.guard_id][0], pos_message[opt.guard_id][1])
                )

                guard_angles = np.array([angle_to_guard])  # shape (1,)

                """ Compute target angles if available """
                angle_to_target = angle_to_guard_egocentric(
                        (pos_message['self'][0], pos_message['self'][1]),
                        pos_message['self'][2],
                        (pos_message[opt.target_id][0], pos_message[opt.target_id][1])
                )
            
                target_angles = np.array([angle_to_target])  # shape (1,)
                
            else:
                if opt.target_id not in pos_message:
                    print(f"Warning: {opt.guard_id} missing from pos_message: {pos_message.keys()}")
                    target_angles = np.array([0.0])  # Default to 0 if no target
                if 'self' not in pos_message:
                    print(f"Warning: 'self' missing from pos_message: {pos_message.keys()}")

            
                

            # Compute sensory input vector b
            self.compute_sensory_map(
                target_positions=target_angles,
                target_qualities=self.target_quality,
                theta_i=self.theta_i,
                kappa=self.kappa,
                guard_angles=guard_angles,
                guard_qualities=self.guard_quality,  # Guard quality
                r=self.spatial_decay,  # Spatial decay rate
                d=guard_agent_dist  # Distance to guard
            )
            print("Running Ring Attractor Model...")
            start_time = time.time()
            times, bump_positions, final_norm = self.compute_dynamics(total_time=50)
            # Log current time and neural field state
            #now = self.get_clock().now().nanoseconds / 1e9  # seconds
        
            end_time = time.time()
            print(f"Model run completed in {end_time - start_time:.2f} seconds.")

            # Store heading over time
            new_heading = 0.0  # one per time step

            # Loop over time steps
            for z_t in self.neural_field:
                norm_z = np.linalg.norm(z_t)

                if norm_z > 0.9:
                    new_heading = compute_center_of_mass(z_t, self.theta_i)

            # Log the neural state
            for t, bump, field in zip(times, bump_positions, self.neural_field):
                self.neural_log.append([t, bump] + field.tolist())
            self.save_log()

            with heading_lock:
                target_heading = new_heading
            
            with perform_turn_lock:
                perform_turn = True


            print(f"Ring attractor updated heading: {new_heading:.2f} rad")
            # Maybe no need to sleep?
            #time.sleep(self.update_interval)

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

    def listener_callback(self, msg):
        with pos_lock:
            global pos_message_g 
            pos_message_g= {}
            for i in range(msg.n):
                if msg.positions[i].subject_name == self.id:
                    # Update the current position with the received data
                    self.current_position = msg.positions[i]
                    pos_message_g['self'] = (float(msg.positions[i].x_trans), float(msg.positions[i].y_trans), 
                                             float( msg.positions[i].z_rot_euler))
                else:
                    pos_message_g[msg.positions[i].subject_name] = (float(msg.positions[i].x_trans), float(msg.positions[i].y_trans), 
                                                                    float(msg.positions[i].z_rot_euler))
                
                #self.get_logger().info('subject "%s" with segment %s:' %(msg.positions[i].subject_name, msg.positions[i].segment_name))
                #self.get_logger().info('I heard translation in x, y, z: "%f", "%f", "%f"' % (msg.positions[i].x_trans, msg.positions[i].y_trans, msg.positions[i].z_trans))
                #self.get_logger().info('I heard rotation in x, y, z, w: "%f", "%f", "%f", "%f": ' % (msg.positions[i].x_rot, msg.positions[i].y_rot, msg.positions[i].z_rot, msg.positions[i].w))

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
def calculate_speed(v_lin, v_ang):
    left_motor = (4.895 * v_lin - 400 * v_ang) * 0.944
    right_motor = 4.895 * v_lin + 400 * v_ang
    return left_motor, right_motor

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

def robot_motion():
    """Thread function for robot motion."""
    global perform_turn  # 👈 tell Python to use the global variable

    try:
        while True:
            with perform_turn_lock:
                is_performing_turn = perform_turn

            if  not is_performing_turn:
                # Move forward
                move_forward()
            
            else:
                
                # Get latest heading from ring attractor
                with heading_lock:
                    current_heading = target_heading
                print(f"Turning to ring attractor heading: {current_heading:.2f} rad")
                turn_to_heading(current_heading)

                with perform_turn_lock:
                    perform_turn = False

            
            # Small pause between cycles
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Stopping robot motion")
    finally:
        stopcar()
        GPIO.cleanup()

def main():
    # Initialize ring attractor
    ring_attractor = RingAttractorModel(num_targets=1)
    
    rclpy.init()
    vicon_node = ViconSubscriber()

    # start other workers
    ring_thread = threading.Thread(target=ring_attractor.run, daemon=True)
    motion_thread = threading.Thread(target=robot_motion, daemon=True)
    ring_thread.start()
    motion_thread.start()

    try:
        # spin blocks main thread until shutdown
        rclpy.spin(vicon_node)
    except KeyboardInterrupt:
        print("Shutting down")
        ring_attractor.stop()
    finally:
        vicon_node.destroy_node()
        rclpy.shutdown()
        stopcar()
        GPIO.cleanup()


if __name__ == '__main__':
    main()