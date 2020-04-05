"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""
#####################################################################################
#####################################################################################
# pylint: disable=invalid-name, E1101

from __future__ import print_function

import math
import unittest
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import mpl_toolkits.mplot3d.axes3d as p3

# Required to do animations in colab
from matplotlib import animation
from IPython.display import HTML

import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2

import scipy
from scipy import linalg, matrix

from sympy import Matrix

# Set plot parameters
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8.0, 8.0)
plt.rc('animation', html='jshtml') # needed for animations!
arrowOptions = dict(head_width=.02,head_length=.02, width=0.01)
#####################################################################################
# Some utility functions for Pose2
def vector3(x, y, z):
    """Create 3D double numpy array."""
    return np.array([x, y, z], dtype=np.float)


def compose(*poses):
    """Compose all Pose2 transforms given as arguments from left to right."""
    return reduce((lambda x, y: x.compose(y)), poses)


def vee(M):
    """Pose2 vee operator."""
    return vector3(M[0, 2], M[1, 2], M[1, 0])


def delta(g0, g1):
    """Difference between x,y,,theta components of SE(2) poses."""
    return vector3(g1.x() - g0.x(), g1.y() - g0.y(), g1.theta() - g0.theta())


def trajectory(g0, g1, N=20):
    """ Create an interpolated trajectory in SE(2), treating x,y, and theta separately.
        g0 and g1 are the initial and final pose, respectively.
        N is the number of *intervals*
        Returns N+1 poses
    """
    e = delta(g0, g1)
    return [Pose2(g0.x()+e[0]*t, g0.y()+e[1]*t, g0.theta()+e[2]*t) for t in np.linspace(0, 1, N)]
#####################################################################################
from sympy.functions import transpose
class FourLinkMM(object):
    """Three-link arm class."""

    def __init__(self):
        self.x_b = 0
        self.y_b = 0
        self.theta_b =0
        
        self.L1 = 3.5
        self.L2 = 3.5
        self.L3 = 2.5
        self.L4 = 0.5


    
    def jacobian(self, q):
        """ Calculate manipulator Jacobian.
            Takes numpy array of joint angles, in radians.
        """        
        
        alpha = self.theta_b + q[0]
        beta = self.theta_b + q[0] + q[1]
        gamma = self.theta_b + q[0] + q[1] + q[2]
        delta = self.theta_b + q[0] + q[1] + q[2] + q[3]

        dxee_wrt_dthetaB = -self.L1*math.cos(alpha) -self.L2*math.cos(beta) -self.L3*math.cos(gamma) -self.L4*math.cos(delta)
        dxee_wrt_dtheta1 = -self.L1*math.cos(alpha) -self.L2*math.cos(beta) -self.L3*math.cos(gamma) -self.L4*math.cos(delta)
        dxee_wrt_dtheta2 = -self.L2*math.cos(beta) -self.L3*math.cos(gamma) -self.L4*math.cos(delta)
        dxee_wrt_dtheta3 = -self.L3*math.cos(gamma) -self.L4*math.cos(delta)
        dxee_wrt_dtheta4 = -self.L4*math.cos(delta)

        dyee_wrt_dthetaB = -self.L1*math.sin(alpha) -self.L2*math.sin(beta) -self.L3*math.sin(gamma) -self.L4*math.sin(delta)
        dyee_wrt_dtheta1 = -self.L1*math.sin(alpha) -self.L2*math.sin(beta) -self.L3*math.sin(gamma) -self.L4*math.sin(delta)
        dyee_wrt_dtheta2 = -self.L2*math.sin(beta) -self.L3*math.sin(gamma) -self.L4*math.sin(delta)
        dyee_wrt_dtheta3 = -self.L3*math.sin(gamma) -self.L4*math.sin(delta)
        dyee_wrt_dtheta4 = -self.L4*math.sin(delta)

        Jacobian = [[1, 0, dxee_wrt_dthetaB, dxee_wrt_dtheta1, dxee_wrt_dtheta2, dxee_wrt_dtheta3, dxee_wrt_dtheta4], \
            [0, 1, dyee_wrt_dthetaB, dyee_wrt_dtheta1, dyee_wrt_dtheta2, dyee_wrt_dtheta3, dyee_wrt_dtheta4], \
            [0, 0,1,1,1,1,1]]
        return Jacobian

    def Velocity_in_NullSpace(self, J, u):
      """ Given a velocity of the base (u) and the Jacobian (J). Compute the velocity of the manipulator, thus the end-effector stays in place.
      """
      
      q_d = Matrix(7,1,[u[0]*math.cos(self.theta_b),u[1]*math.sin(self.theta_b),u[2], 0, 0, 0, 0])
      #print("u")
      #print(u)
      # q_d = J.pinv()*(transpose(u)*transpose(J[:, 0:3]))
      a = u[0]*J[:,0]+u[1]*J[:,1]+u[2]*J[:,2]
      Jq = J[:,3:7]
      q_d = Jq.pinv()*(-a)
      #print("q_d")
      #print(q_d)
      # q_d = Matrix(7,1,[u[0]*math.cos(self.theta_b),u[1]*math.sin(self.theta_b),u[2], 0, 0, 0, 0])

      # count = 0
      # while count <1000:
      #   q_d = J.pinv()*u + (Matrix.eye(7)-J.pinv()*J)*q_d
      #   count +=1
      #q_d = Matrix([q_d[3],q_d[4],q_d[5],q_d[6]])
      return q_d

    def  null_space_projector(self, J):
      """ Compute the null space projector
      """
      I = Matrix.eye(7)

      return I-J.pinv()*J
    
    def partialReduceN(self, N):
      #row echelon
      #repeat steps 1 to 4 but move down one pivot row
      for pivotRow in  range(0,3):
        #find pivot (first nonzero entry) in first column
        # a = N[:,0]
        # print(a)
        pivot = pivotRow
        #switch this row to the top
        
        #divide the row by this pivot value
        N[pivotRow,:]= N[pivotRow,:]/N[pivot,pivotRow]

        #subtract below rows by multiples of top row
        for r in range(pivotRow+1,3):
          N[r,:] =N[r,:] - N[r,pivotRow]*N[pivotRow,:]

      #reduced echelon form
      #add multiples of pivot row
      for pivotRow in  range(2, -1, -1):
        for r in  range(pivotRow-1, -1, -1):
          N[r,:] =N[r,:] - N[r,pivotRow]*N[pivotRow,:]

      #print(N)
      
      #get 0's in the 3 by 4 top left region on N
      for i in  range(3, 7):
        N[:,i] = N[:,i] - N[0,i]*N[:,0]
        N[:,i] = N[:,i] - N[1,i]*N[:,1]
        N[:,i] = N[:,i] - N[2,i]*N[:,2]

      return N
    def null_space_m(self, N):
      """ Modify the null space projector, so it give us the required linear and angular velocity
      """
      
      #print("N.rref()", N.rref())
      N = self.partialReduceN(N)
      #print("N", N)
      return N

#####################################################################################
import math

import matplotlib.pyplot as plt

show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, reso, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(nstart)] = nstart

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(ngoal,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.minx),
                         self.calc_grid_position(current.y, self.miny), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closed_set)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [self.calc_grid_position(ngoal.x, self.minx)], [
            self.calc_grid_position(ngoal.y, self.miny)]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(self.calc_grid_position(n.x, self.minx))
            ry.append(self.calc_grid_position(n.y, self.miny))
            pind = n.pind

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, minp):
        """
        calc grid position

        :param index:
        :param minp:
        :return:
        """
        pos = index * self.reso + minp
        return pos

    def calc_xyindex(self, position, min_pos):
        return round((position - min_pos) / self.reso)

    def calc_grid_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.minx)
        py = self.calc_grid_position(node.y, self.miny)

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        # collision check
        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx) / self.reso)
        self.ywidth = round((self.maxy - self.miny) / self.reso)
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        # obstacle map generation
        self.obmap = [[False for i in range(self.ywidth)]
                      for i in range(self.xwidth)]
        for ix in range(self.xwidth):
            x = self.calc_grid_position(ix, self.minx)
            for iy in range(self.ywidth):
                y = self.calc_grid_position(iy, self.miny)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    fig, ax = plt.subplots()
    ##########################################
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 5.0  # [m]

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

    ##########################################

    MM = FourLinkMM()
    MM_aux = FourLinkMM()
    MM_aux.x_b = MM.x_b = sx
    MM_aux.y_b = MM.y_b = sy


    N=206
    omega = 2*math.pi/N
    
    q = np.radians(np.array([-90, 90, 90,90], dtype=np.float))
    len_b = 2
    d = np.sqrt(2*len_b*len_b/4)
    
    #rect = rect = mpatches.Rectangle([MM.x_b-d*np.cos(MM.theta_b+np.radians(45)),MM.y_b-d*np.sin(MM.theta_b+np.radians(45))], len_b, len_b, angle = MM.theta_b*180/np.pi)
    #ax.add_artist(rect)
    #ax.add_artist(mpatches.FancyArrow(MM.x_b,MM.y_b, 1.5*np.cos(MM.theta_b), 1.5*np.sin(MM.theta_b), color='r',head_width = 0.3))

    rect = mpatches.Rectangle([MM_aux.x_b-d*np.cos(MM_aux.theta_b+np.radians(45)),MM_aux.y_b-d*np.sin(MM_aux.theta_b+np.radians(45))], len_b, len_b, angle = MM_aux.theta_b*180/np.pi, color = 'r', alpha=0.4)
    #ax.clear()
    #ax.set_xlim((-size, size))
    #ax.set_ylim((-size, size))
    ax.add_artist(rect)
    rect = mpatches.Rectangle([MM.x_b-d*np.cos(MM.theta_b+np.radians(45)),MM.y_b-d*np.sin(MM.theta_b+np.radians(45))], len_b, len_b, angle = MM.theta_b*180/np.pi)
    ax.add_artist(rect)

    sXl1 = Pose2(0, 0, math.radians(90)+MM.theta_b)
    l1Zl1 = Pose2(0, 0, q[0])
    l1Xl2 = Pose2(MM.L1, 0, 0)
    sTl2 = compose(sXl1, l1Zl1, l1Xl2)
    t1 = sTl2.translation()
    ax.add_artist(mpatches.Rectangle([MM.x_b,MM.y_b], 3.5, 0.1, angle =(MM.theta_b+q[0])*180/np.pi+90, color='r'))
    ax.add_artist(mpatches.FancyArrow(MM.x_b,MM.y_b, 1.5*np.cos(MM.theta_b), 1.5*np.sin(MM.theta_b), color='r',head_width = 0.3))

    l2Zl2 = Pose2(0, 0, q[1])
    l2Xl3 = Pose2(MM.L2, 0, 0)
    sTl3 = compose(sTl2, l2Zl2, l2Xl3)
    t2 = sTl3.translation()
    ax.add_artist(mpatches.Rectangle([t1.x()+MM.x_b,t1.y()+MM.y_b], 3.5, 0.1, angle =(MM.theta_b+q[0]+q[1])*180/np.pi+90, color='g'))

    l3Zl3 = Pose2(0, 0, q[2])
    l3X4 = Pose2(MM.L3, 0, 0)
    sTl4 = compose(sTl3, l3Zl3, l3X4)
    t3 = sTl4.translation()
    ax.add_artist(mpatches.Rectangle([t2.x()+MM.x_b,t2.y()+MM.y_b], 2.5, 0.1, angle =(MM.theta_b+q[0]+q[1]+q[2])*180/np.pi+90, color='b'))

    l4Zl4 = Pose2(0, 0, q[3])
    l4Xt = Pose2(MM.L4, 0, 0)
    sTt = compose(sTl4, l4Zl4, l4Xt)
    t4 = sTt.translation()
    ax.add_artist(mpatches.Rectangle([t3.x()+MM.x_b,t3.y()+MM.y_b], 0.5, 0.1, angle =(MM.theta_b+q[0]+q[1]+q[2]+q[3])*180/np.pi+90, color='k'))
    ##########################################

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()
