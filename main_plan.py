import math
import matplotlib.pyplot as plt
import numpy as np

from a_star import AStarPlanner

show_animation = True

from NLinkArm import NLinkArm
from kinematics_main import inverse_kinematics, forward_kinematics, jacobian_inverse, distance_to_goal, ang_diff

####forward kinematics stuff
Kp = 2
dt = 0.1
N_LINKS = 10
N_ITERATIONS = 10000

# States
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2

def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstable positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()
        
    start_kin_x = rx[0]
    start_kin_y = ry[1]
    goal_x_kin, goal_y_kin = gx, gy
    
    #do forward kinematics
    
    link_lengths = [1] * N_LINKS
    joint_angles = np.array([0] * N_LINKS)
    goal_pos = goal_x_kin, goal_y_kin
    
    arm = NLinkArm(link_lengths, joint_angles, goal_pos, show_animation)
    state = WAIT_FOR_NEW_GOAL
    solution_found = False
    
    while(1):
        if state is WAIT_FOR_NEW_GOAL:
            
            old_goal = np.array(goal_pos)
            goal_pos = np.array(arm.goal)
            end_effector = arm.end_effector
            errors, distance = distance_to_goal(end_effector, goal_pos)
        
            if distance > 0.1 and not solution_found:
                joint_goal_angles, solution_found = inverse_kinematics(
                    link_lengths, joint_angles, goal_pos)
                if not solution_found:
                    print("Solution could not be found.")
                    break
                elif solution_found:
                    state = MOVING_TO_GOAL
                        
            elif state is MOVING_TO_GOAL:
                if distance > 0.1 and all(old_goal == goal_pos):
                    joint_angles = joint_angles + Kp * \
                        ang_diff(joint_goal_angles, joint_angles) * dt
                else:
                    state = WAIT_FOR_NEW_GOAL
                    solution_found = False
                    print("Solution could not be found.")
                    break
                    
        arm.update_joints(joint_angles)
    

if __name__ == '__main__':
    main()