######################################
#Imports
######################################
#general imports
import time
import math
import matplotlib.pyplot as plt
import numpy as np

#Imports needed for path planning
from a_star import AStarPlanner
pi= math.pi

#imports for obstacle avoidance
from arm_obstacle_navigation import NLinkArm, get_occupancy_grid, astar_torus

######################################
#INPUTS 
######################################
testcase = 0
if testcase == 0:
        obstacleMapId = 0
        robotStartPose = (14.0, 14.0, 0)#(48.0,48.0, 0)  # (x,y) [m]
        robotEndPose = (48.0,48.0, 0)#(14.0, 14.0, 0)  # (x,y) [m]
        objectPose = (52.0, 53.0, pi/2)
        useJointSpaceMotionControl = True 
if testcase == 1:
        obstacleMapId = 1
        robotStartPose = (48.0,48.0, 0)#(10.0,10.0,0)  # (x,y) [m]
        robotEndPose = (14.0, 14.0, 0)#(48.0, 48.0, 0)  # (x,y) [m]
        objectPose = (8.0, 10.0, pi)#(55.0, 51.0, 0)
        useJointSpaceMotionControl = False

#for A* path planning

show_animation = True
grid_size = 2.0  # [m]
robot_radius = 3.0  # [m]

# start and goal position
#robotStartPose = (10.0,10.0,0)  # (x,y) [m]
#robotEndPose = (48.0, 48.0, 0)  # (x,y) [m]
#objectPose = (50.0, 50.0, pi/2)
#robotStartPose = (48.0,48.0, 0)  # (x,y) [m]
#robotEndPose = (14.0, 14.0, 0)  # (x,y) [m]
#objectPose = (10.0, 10.0, -pi/2)

#needed for arm kinematics
N_LINKS = 2
link_lengths = [robot_radius] * N_LINKS
joint_angles = np.array([-np.pi, np.pi]) #np.array([0] * N_LINKS)  
N_ITERATIONS = 10000

M = 100
obstacles = [[2.35, 3.75, 0.6], [4.55, 2.5, 0.5], [0, -1, 0.25]]



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

def generateObstacleLocationMap2():
    # set obstable positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 60):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 60):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        oy.append(20.0)
        ox.append(i)
    for i in range(0, 40):
        oy.append(40.0)
        ox.append(60.0 - i)

    for i in range(-10+16, 60-16):
        ox.append(i)
        oy.append(30.0)

    for i in range(16, 20):
        ox.append(30.0)
        oy.append(i)

    for i in range(8, 14):
        ox.append(30.0)
        oy.append(i)

    for i in range(-10, 0):
        ox.append(30.0)
        oy.append(i)

    return ox,oy
##
#Gets x and y pixel positions containing obstacles. obstacles include the bounding wall
##

def plotRobotTrajectory(traj_x, traj_y, xPositionOfObstacles, yPositionOfObstacles):
    
    rev_traj_x = traj_x[::-1]
    rev_traj_y = traj_y[::-1]
    
    for i in range(len(traj_x)):
        plt.cla()
        generateMapPlotWithRobot(xPositionOfObstacles, yPositionOfObstacles, robotStartPose, robotEndPose, objectPose)
        plt.plot(traj_x, traj_y, "-r")
        
        points = [[rev_traj_x[i], rev_traj_y[i]] for _ in range(N_LINKS + 1)]
        
        for i in range(1, N_LINKS + 1):
            points[i][0] = points[i - 1][0] + link_lengths[i - 1] * \
                np.cos(np.sum(joint_angles[:i]))
            points[i][1] = points[i - 1][1] + link_lengths[i - 1] * \
                np.sin(np.sum(joint_angles[:i]))
        
        for i in range(N_LINKS + 1):
            if i is not N_LINKS:
                x = plt.plot([points[i][0], points[i + 1][0]],
                            [points[i][1], points[i + 1][1]], 'g-', linewidth=2)
            plt.plot(points[i][0], points[i][1], 'ko', markersize=1)
            
        for obstacle in obstacles:
            circle = plt.Circle(
                (robotEndPose[0] + obstacle[0], robotEndPose[1] + obstacle[1]), radius=0.5 * obstacle[2], fc='k')
            plt.gca().add_patch(circle)
            
        plt.draw()
        plt.pause(0.0001)
        
def plotManipulatorTrajectory(xPositionOfObstacles, yPositionOfObstacles, traj_angles):
    
    # rev_traj_x = traj_x[::-1]
    # rev_traj_y = traj_y[::-1]
    
    for i in range(len(traj_angles)):
        
        curr_joint_angles = traj_angles[i]
        
        plt.cla()
        generateMapPlotWithRobot(xPositionOfObstacles, yPositionOfObstacles, robotStartPose, robotEndPose, objectPose)
        # plt.plot(traj_x, traj_y, "-r")
        
        points = [[robotEndPose[0], robotEndPose[1]] for _ in range(N_LINKS + 1)]
        
        for i in range(1, N_LINKS + 1):
            points[i][0] = points[i - 1][0] + link_lengths[i - 1] * \
                np.cos(np.sum(curr_joint_angles[:i]))
            points[i][1] = points[i - 1][1] + link_lengths[i - 1] * \
                np.sin(np.sum(curr_joint_angles[:i]))
        
        for i in range(N_LINKS + 1):
            if i is not N_LINKS:
                x = plt.plot([points[i][0], points[i + 1][0]],
                            [points[i][1], points[i + 1][1]], 'g-', linewidth=2)
            plt.plot(points[i][0], points[i][1], 'ko', markersize=1)
            
        plt.draw()
        plt.pause(0.0001)

def generateMapPlotWithRobot(xPositionOfObstacles,yPositionOfObstacles, robotStartPose, robotEndPose, objectPose):
    if show_animation:  # pragma: no cover
        plt.plot(xPositionOfObstacles, yPositionOfObstacles, ".k")
        plt.plot(robotStartPose[0], robotStartPose[1], "og")
        plt.plot(robotEndPose[0], robotEndPose[1], "xb")
        plt.plot(objectPose[0], objectPose[1], "xr")
        
        for obstacle in obstacles:
            circle = plt.Circle(
                (robotEndPose[0] + obstacle[0], robotEndPose[1] + obstacle[1]), radius=0.5 * obstacle[2], fc='k')
            plt.gca().add_patch(circle)
        
        plt.grid(True)
        plt.axis("equal")

def main():
    print("testcase = ", testcase)
    #Gets x and y positions of image coordinates containing obstacles. obstacles include the bounding wall
    if obstacleMapId == 0:
    	xPositionOfObstacles, yPositionOfObstacles = generateObstacleLocationMap()
    elif obstacleMapId == 1:
        xPositionOfObstacles, yPositionOfObstacles = generateObstacleLocationMap2()
    
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

    #animate robot to move along desired trajectory
    plotRobotTrajectory(rx, ry, xPositionOfObstacles, yPositionOfObstacles)

    #do complicated manipulator stuff with obstacles
    
    arm = NLinkArm(link_lengths, joint_angles)
    start = (0, 0)
    goal = (61, 57)
    grid = get_occupancy_grid(arm, obstacles)
    route = astar_torus(grid, start, goal)
    
    listOfJointAnglesInTrajectory = []
    
    for node in route:
        theta1 = 2 * pi * node[0] / M - pi
        theta2 = 2 * pi * node[1] / M - pi
        listOfJointAnglesInTrajectory.append((theta1, theta2))
        
    print(listOfJointAnglesInTrajectory)
    
    plotManipulatorTrajectory(xPositionOfObstacles, yPositionOfObstacles, listOfJointAnglesInTrajectory)
    
    



if __name__ == '__main__':
    main()
    while 1:
        a = 1
