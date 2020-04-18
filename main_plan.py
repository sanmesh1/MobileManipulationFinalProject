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
robot_radius = 2.0  # [m]

# start and goal position
robotStartPose = (10.0,10.0)  # (x,y) [m]
robotEndPose = (48.0, 48.0)  # (x,y) [m]
objectPose = (50.0, 50.0)

#needed for arm kinematics
N_LINKS = 3
link_lengths = [2] * N_LINKS
joint_angles = np.array([0, -np.pi, -np.pi]) #np.array([0] * N_LINKS)  
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
                            [points[i][1], points[i + 1][1]], 'r-', linewidth=2)
            plt.plot(points[i][0], points[i][1], 'ko', markersize=1)
            
        plt.draw()
        plt.pause(0.0001)
        
def plotManipulatorTrajectory(traj_x, traj_y, xPositionOfObstacles, yPositionOfObstacles, traj_angles):
    
    rev_traj_x = traj_x[::-1]
    rev_traj_y = traj_y[::-1]
    
    for i in range(len(traj_angles)):
        
        curr_joint_angles = traj_angles[i]
        
        plt.cla()
        generateMapPlotWithRobot(xPositionOfObstacles, yPositionOfObstacles, robotStartPose, robotEndPose, objectPose)
        plt.plot(traj_x, traj_y, "-r")
        
        points = [[robotEndPose[0], robotEndPose[1]] for _ in range(N_LINKS + 1)]
        
        for i in range(1, N_LINKS + 1):
            points[i][0] = points[i - 1][0] + link_lengths[i - 1] * \
                np.cos(np.sum(curr_joint_angles[:i]))
            points[i][1] = points[i - 1][1] + link_lengths[i - 1] * \
                np.sin(np.sum(curr_joint_angles[:i]))
        
        for i in range(N_LINKS + 1):
            if i is not N_LINKS:
                x = plt.plot([points[i][0], points[i + 1][0]],
                            [points[i][1], points[i + 1][1]], 'r-', linewidth=2)
            plt.plot(points[i][0], points[i][1], 'ko', markersize=1)
            
        plt.draw()
        plt.pause(0.0001)

def generateMapPlotWithRobot(xPositionOfObstacles,yPositionOfObstacles, robotStartPose, robotEndPose, objectPose):
    if show_animation:  # pragma: no cover
        plt.plot(xPositionOfObstacles, yPositionOfObstacles, ".k")
        plt.plot(robotStartPose[0], robotStartPose[1], "og")
        plt.plot(robotEndPose[0], robotEndPose[1], "xb")
        plt.plot(objectPose[0], objectPose[1], "xr")
        
        plt.grid(True)
        plt.axis("equal")

def jacobian(q):
# Calculate manipulator Jacobian.
#Takes numpy array of joint angles, in radians.
    l1 = link_lengths[0]
    l2 = link_lengths[1]
    l3 = link_lengths[2]

    zero_zero = -l1*math.sin(q[0]) -l2*math.sin(q[0]+q[1]) - l3*math.sin(q[0]+q[1]+q[2]) 
    zero_one = -l2*math.sin(q[0]+q[1]) -l3*math.sin(q[0]+q[1]+q[2])
    zero_two = - l3*math.sin(q[0]+q[1]+q[2])  
    one_zero = l1*math.cos(q[0]) +l2*math.cos(q[0]+q[1]) + l3*math.cos(q[0]+q[1]+q[2])
    one_one= l2*math.cos(q[0]+q[1]) + l3*math.cos(q[0]+q[1]+q[2])
    one_two =   l3*math.cos(q[0]+q[1]+q[2])
    return np.array([[zero_zero, zero_one, zero_two],
                 [one_zero, one_one,one_two],
                 [1,1,1] 
                 ])#np.eye(3)

def inverseKinematicCartesianAngleStep(targetPose, currentPose, q, Kp):
    pi = math.pi
    #arm pose
    armX = currentPose[0]
    armY = currentPose[1]
    armTheta = currentPose[2]
    #goal pose
    goal_x_kin = targetPose[0]
    goal_y_kin = targetPose[1]
    goal_theta_kin = targetPose[2]

    while (armTheta>pi):
        armTheta -= 2*pi
    while (armTheta<-pi):
        armTheta += 2*pi

    # 2.Calculate position error
    ex = goal_x_kin - armX
    ey = goal_y_kin - armY
    et = goal_theta_kin - armTheta

    while (et>pi):
        et -= 2*pi
    while (et<-pi):
        et += 2*pi

    #calculate error
    err = np.array([ex, ey, et])
    #calculate inverse jacobian
    invJ = np.linalg.pinv(jacobian(q))

    # 3. Do PID Control on position
    q = np.matmul(invJ, err)*Kp + q

    return q, err

def reorientArmToGrabObject(robotEndPose, objectPose, arm, Kp, marginFromGoal):
    #temp

    pi = math.pi
    #object pose
    obj_x = objectPose[0]
    obj_y = objectPose[1]
    obj_theta = math.pi/2
    #robot base end pose
    robot_x = robotEndPose[0]
    robot_y = robotEndPose[1]
    robot_theta = np.sum(arm.joint_angles)
    #arm goal pose relative to robot base
    goal_x_kin, goal_y_kin = (obj_x-robot_x), (obj_y-robot_y)
    arm.goal = goal_x_kin, goal_y_kin
    goal_pos = arm.goal
    goal_theta_kin = obj_theta - robot_theta

    err_norm = 100
    listOfJointAnglesInTrajectory = []
    while (err_norm  > marginFromGoal):
        #calculate current arm pose
        armX, armY = forward_kinematics(link_lengths, arm.joint_angles)
        armTheta = np.sum(arm.joint_angles)
        q = arm.joint_angles
        targetPose = (goal_x_kin,goal_y_kin, goal_theta_kin)
        currentPose = (armX, armY, armTheta)

        #do one step of cartesian velocity control
        q, err = inverseKinematicCartesianAngleStep(targetPose, currentPose, q, Kp)

        #if error is less than margin, stop
        err_norm = np.linalg.norm(err)
        if err_norm  < marginFromGoal:
            return arm.joint_angles, True, listOfJointAnglesInTrajectory

        listOfJointAnglesInTrajectory.append(q)
        arm.joint_angles = q
        arm.update_joints(q)
        

    return arm.joint_angles, False, listOfJointAnglesInTrajectory

def main():

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

    #animate robot to move along desired trajectory
    plotRobotTrajectory(rx, ry, xPositionOfObstacles, yPositionOfObstacles)

    #initial robot parameters
    # arm = NLinkArm(link_lengths, joint_angles, np.array([0,0]), show_animation)
    arm = NLinkArm(link_lengths, joint_angles, np.array([0,0]), False)
    
    
    #Calculate arm pose to grab object
    Kp = 0.1
    marginFromGoal = 1e-2
    joint_goal_angles, solution_found, listOfJointAnglesInTrajectory = reorientArmToGrabObject(robotEndPose, objectPose, arm, Kp, marginFromGoal)
    
    plotManipulatorTrajectory(rx, ry, xPositionOfObstacles, yPositionOfObstacles, listOfJointAnglesInTrajectory)



if __name__ == '__main__':
    main()
    while 1:
        a = 1
