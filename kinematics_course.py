from __future__ import print_function

import math
import unittest
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import mpl_toolkits.mplot3d.axes3d as p3

from matplotlib import animation
from IPython.display import HTML

import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2

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

# The 3-link manipulator class

class ThreeLinkArm(object):
	"""Three-link arm class."""

	def __init__(self):
		self.L1 = 3.5
		self.L2 = 3.5
		self.L3 = 2.5

	def fk(self, q):
		""" Forward kinematics.
			Takes numpy array of joint angles, in radians.
		"""

		T_0 = Pose2(0, 0, math.radians(90) + q[0])
		T1 = Pose2(self.L1, 0, q[1])
		T2 = Pose2(self.L2, 0, q[2])
		T3 = Pose2(self.L3, 0, 0)
		out = compose(T_0, T1, T2, T3)

		return out

	def jacobian(self, q):
		""" Calculate manipulator Jacobian.
			Takes numpy array of joint angles, in radians.
		"""

		theta1 = q[0]
		theta2 = q[1]
		theta3 = q[2]

		alpha = theta1 + theta2 
		beta = theta1 + theta2 + theta3

		J = [[-self.L1 * math.cos(theta1) - self.L2 * math.cos(alpha) - self.L3 * math.cos(beta),
			  -self.L2 * math.cos(alpha) - self.L3 * math.cos(beta), - self.L3 * math.cos(beta)], 
			 [-self.L1 * math.sin(theta1) - self.L2 * math.sin(alpha) - self.L3 * math.sin(beta),
			  - self.L2 * math.sin(alpha) - self.L3 * math.sin(beta), - self.L3 * math.sin(beta)], 
			 [1, 1, 1]]

		J = np.array(J)

		return J
	
def main(): 
	# First set up the figure, the axis, and the plot element we want to animate
	fig, ax = plt.subplots()
	plt.close()
	N=50
	size=10.5
	ax.set_xlim((-size, size))
	ax.set_ylim((-size, size))
	omega = 2*math.pi/N

	arm = ThreeLinkArm()
	q = np.radians(vector3(30, -30, 45))
	sTt_initial = arm.fk(q)
	sTt_goal = Pose2(2.4, 4.3, math.radians(0))
	poses = trajectory(sTt_initial, sTt_goal, N)

	def init():
		rect = mpatches.Rectangle([0,0], 1, 1, angle =0)
		return (rect,)

	# animation function. This is called sequentially  
	def animate(i):
		global pose
		global arm
		global q

		# Computes the forward kinematics to get the pose of the end-effector for the given angular position of the joints (q)
		sTt = arm.fk(q)
		# Evaluate the error between the current position of the end-effector and the desired position at moment i
		error = delta(sTt, poses[i])
		# Get the jacobian of the arm at the given pose
		J = arm.jacobian(q)
		# Move the arm joints in the respective direction
		q += np.dot(np.linalg.inv(J), error)

		# ------------------------- ANIMATION ----------------------------------------------------
		rect = rect = mpatches.Rectangle([-0.5,-0.5], 1, 1, angle =0)
		ax.clear()
		ax.set_xlim((-size, size))
		ax.set_ylim((-size, size))
		ax.add_artist(rect)
		
		sXl1 = Pose2(0, 0, math.radians(90))
		l1Zl1 = Pose2(0, 0, q[0])
		l1Xl2 = Pose2(arm.L1, 0, 0)
		sTl2 = compose(sXl1, l1Zl1, l1Xl2)
		t1 = sTl2.translation()
		ax.add_artist(mpatches.Rectangle([0,0], 3.5, 0.1, angle =q[0]*180/np.pi+90, color='r'))

		l2Zl2 = Pose2(0, 0, q[1])
		l2Xl3 = Pose2(arm.L2, 0, 0)
		sTl3 = compose(sTl2, l2Zl2, l2Xl3)
		t2 = sTl3.translation()
		ax.add_artist(mpatches.Rectangle([t1.x(),t1.y()], 3.5, 0.1, angle =(q[0]+q[1])*180/np.pi+90, color='g'))

		l3Zl3 = Pose2(0, 0, q[2])
		l3Xt = Pose2(arm.L3, 0, 0)
		sTt = compose(sTl3, l3Zl3, l3Xt)
		t3 = sTt.translation()
		ax.add_artist(mpatches.Rectangle([t2.x(),t2.y()], 2.5, 0.1, angle =(q[0]+q[1]+q[2])*180/np.pi+90, color='b'))

	animation.FuncAnimation(fig, animate, init_func=init, 
							frames=N, interval=100, blit=False)


if __name__ == '__main__':
	main()




