######################################
#Imports
######################################
#general imports
import math
import matplotlib.pyplot as plt
import numpy as np

#Imports needed for path planning
from a_star import AStarPlanner

#Imports needed for arm kinematics
from NLinkArm import NLinkArm
from kinematics_main import inverse_kinematics, forward_kinematics, jacobian_inverse, distance_to_goal, ang_diff

######################################
#INPUTS 
######################################
#for A* path planning
show_animation = True
grid_size = 2.0  # [m]
robot_radius = 1.0  # [m]

# start and goal position
robotStartPose = (10.0,10.0)  # (x,y) [m]
robotEndPose = (48.0, 48.0)  # (x,y) [m]
objectPose = (50.0, 50.0)

#needed for arm kinematics
N_LINKS = 3
link_lengths = [1] * N_LINKS
joint_angles = np.array([0] * N_LINKS)  
N_ITERATIONS = 10000
WAIT_FOR_NEW_GOAL = 1
MOVING_TO_GOAL = 2
#Kp = 1 #0.2
#dt = 0.1





######################################
#functions
######################################
def generateObstacleLocationMap():
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
    return ox,oy

##
#Gets x and y pixel positions containing obstacles. obstacles include the bounding wall
##
def generateMapPlotWithRobot(xPositionOfObstacles,yPositionOfObstacles, robotStartPose, robotEndPose, objectPose):
    if show_animation:  # pragma: no cover
        plt.plot(xPositionOfObstacles, yPositionOfObstacles, ".k")
        plt.plot(robotStartPose[0], robotStartPose[1], "og")
        plt.plot(robotEndPose[0], robotEndPose[1], "xb")
        plt.plot(objectPose[0], objectPose[1], "xr")
        plt.grid(True)
        plt.axis("equal")

def main():
    #initial robot parameters
  

    #Gets x and y positions of image coordinates containing obstacles. obstacles include the bounding wall
    xPositionOfObstacles, yPositionOfObstacles = generateObstacleLocationMap()
    
    #plot the map with the obstacles and robot
    generateMapPlotWithRobot(xPositionOfObstacles, yPositionOfObstacles, robotStartPose, robotEndPose, objectPose)

    #Generate path to goal
    sx = robotStartPose[0]
    sy = robotStartPose[1]
    gx = robotEndPose[0]
    gy = robotEndPose[1]
    obj_x = objectPose[0]
    obj_y = objectPose[1]

    a_star = AStarPlanner(xPositionOfObstacles, yPositionOfObstacles, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    #Show calculated path
    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()
    
    #animate robot to move along desired trajectory
    #TO DO: 


    goal_x_kin, goal_y_kin = (obj_x-gx), (obj_y-gy)
    
    #Calculate arm pose to grab object
    #reorientArmToGrabObject()
    

    goal_pos = goal_x_kin, goal_y_kin
    
    arm = NLinkArm(link_lengths, joint_angles, goal_pos, show_animation)
    state = WAIT_FOR_NEW_GOAL
    solution_found = False
       
    old_goal = np.array(goal_pos)
    print("old_goal = ", old_goal)
    goal_pos = np.array(arm.goal)
    print("goal_pos = ", goal_pos)
    end_effector = arm.end_effector
    print("end_effector = ", end_effector)
    errors, distance = distance_to_goal(end_effector, goal_pos)
    print("errors = ", errors)
    print("distance = ", distance)
    print("state = ", state)
    print("solution_found = ", solution_found)
    
    if distance > 0.1 and not solution_found:
        joint_goal_angles, solution_found = inverse_kinematics(
            link_lengths, joint_angles, goal_pos)
        if not solution_found:
            print("Solution could not be found.")
        elif solution_found:
                state = MOVING_TO_GOAL
    arm.update_joints(joint_goal_angles)

#    if state is WAIT_FOR_NEW_GOAL:
#    
#        if distance > 0.1 and not solution_found:
#            joint_goal_angles, solution_found = inverse_kinematics(
#                link_lengths, joint_angles, goal_pos)
#            if not solution_found:
#                print("Solution could not be found.")
#    
#            elif solution_found:
#                state = MOVING_TO_GOAL
#    #joint_angles = joint_angles + Kp * \
#                #ang_diff(joint_goal_angles, joint_angles) * dt
#    arm.update_joints(joint_goal_angles)
            
#    if state is MOVING_TO_GOAL:
#        if distance > 0.1 and all(old_goal == goal_pos):
#            joint_angles = joint_angles + Kp * \
#                ang_diff(joint_goal_angles, joint_angles) * dt
#        else:
#            state = WAIT_FOR_NEW_GOAL
#            solution_found = False
#            print("Solution could not be found.")
#                    
#        arm.update_joints(joint_angles)


if __name__ == '__main__':
    main()
    while 1:
        a = 1
