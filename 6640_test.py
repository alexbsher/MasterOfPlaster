#!/usr/bin/env python

# Standard Python Imports
import os
import numpy as np
import time
import imp
import sys
from argparse import ArgumentParser

# OpenRAVE
from openravepy import *
#RaveInitialize(True, DebugLevel.Debug)

# Import Dfab Python, Watch out for hard coded directory
dfab_pack = imp.load_package('dfab', '../dfab/python/dfab/')
from dfab.mocap import extract_trajectory, datafiles
from dfab.geometry.quaternion import to_threexform
from dfab.rapid.joint_sequence import single_trajectory_program

from scipy.optimize import fmin
from math import sqrt, cos, sin, radians

curr_path = os.getcwd()
relative_ordata = '/models'
ordata_path_thispack = curr_path + relative_ordata

#this sets up the OPENRAVE_DATA environment variable to include the files we're using
openrave_data_path = os.getenv('OPENRAVE_DATA', '')
openrave_data_paths = openrave_data_path.split(':')
if ordata_path_thispack not in openrave_data_paths:
  if openrave_data_path == '':
      os.putenv('OPENRAVE_DATA', ordata_path_thispack)
  else:
      os.putenv('OPENRAVE_DATA', '%s:%s'%(ordata_path_thispack, openrave_data_path))


class RoboHandler:
  def __init__(self, mode):
    self.env = Environment()
    self.env.SetViewer('qtcoin')
    self.env.GetViewer().SetName('Tutorial Viewer')
    self.env.Load('6640_test.env.xml')
    self.robot = self.env.GetRobots()[0]
    self.wall = self.env.GetKinBody('plaster_wall')

    # Init IK Solutions
    self.manip = self.robot.GetActiveManipulator()
    ikmodel = databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=IkParameterizationType.Transform6D)
    if not ikmodel.load():
      ikmodel.autogenerate()

    # Storage of real/sim mode
    self.mode = mode
    
  def getMocapData(self, filename, body='6 Point Trowel', subsample=12):
    '''
    Looks up a csv filename for mocap data and if successful, returns times, x, 
    q, ypr in world frame.
    @ Params -> filename : CSV File with Mocap Data
    @ Returns -> times   : vector of times corresponding to readings
                 x       : (x, y, z) of centroid of body relative to world frame
                 q       : quaternion of body relative to world frame
                 ypr     : Yaw, Pitch, and Roll of body relative to world frame 
    '''
    data = extract_trajectory.load_csv_data(filename)
    return extract_trajectory.extract_trajectory(data, body=body)

  def getMocapTraj(self, filename):
    '''
    Looks up a csv filename for mocap trajectory and if successful, returns times, 
    and 4x4 transforms in world frame.
    @ Params -> filename : CSV File with Mocap Data
    @ Returns -> paths : list of 4x4 transforms found in traj file
                 times : vector of times corresponding to readings
                 
    '''
    paths, times = datafiles.read_frame_trajectory_file(filename)
    transforms = [np.identity(4) for i in paths]
    for i in xrange(0, len(transforms)):
      self.write_frame_record_to_transform(paths[i], transforms[i])

    return (transforms, times)

  def write_frame_record_to_transform( self, frame, transform ):
    """Write a 2-D frame list in file format into a numpy matrix."""
    # write the frame components into the homogeneous transform
    transform[0:3, 3] = frame[0] # origin
    transform[0:3, 0] = frame[1] # X axis
    transform[0:3, 1] = frame[2] # Y axis
    transform[0:3, 2] = frame[3] # Z axis

    # scale the millimeters from the file to meters for the graphics
    transform[0:3, 3] *= 0.001
    # Swap X and Y, as they're flipped between data and sim bot
    #temp = transform[1, 3] 
    #transform[1, 3] = transform[0, 3] 
    #transform[0, 3] = temp

    return

  
  def moveIK(self, Tgoal, move=True): 
    '''
    Attempts to move the robots end effector to a given transform denoted by
    Tgoal.  Returns None if no solution is found or Returns the solution if
    there is one.
    '''
    # Get a solution from IK Solver
    sol = self.manip.FindIKSolution(Tgoal, IkFilterOptions.CheckEnvCollisions)
    # If solution is none, print info
    if sol == None:
      print "No Solution Found!"
      print Tgoal
    # If we're actually supposed to move, do it
    elif move:
      self.robot.SetDOFValues(sol, self.manip.GetArmIndices())
    return sol

  def moveTrajectory(self, traj):
    '''
    Takes a trajectory, which is [for now] just a 2xn list of points to put the 
    end effector, where this list is a list of 4x4 transforms in the world frame 
    and i=1:n where n is the number of points.
    '''
    for pos, q in traj:
      # If Sim, then OR works in m, not mm
      if self.mode == 'sim':
        pos = pos /1000
      rot_t = to_threexform(q)
      Tgoal = np.dot(np.eye(4), np.eye(4)) # Start with Orientation
      Tgoal[0:3, 3] = pos                  # Add Position
      self.moveIK(Tgoal)
      time.sleep(.1)

  def moveTransforms(self, transforms, toolframe=False):
    '''
    Takes a list of transforms and moves the robot through the trajectory.
    '''
    # Iterate over all transforms
    for t in transforms:
      # If coordinates passed in tool frame, change to robot frame
      if toolframe:
        t = self.toolToRobotTransform(t)
      # Move the robot
      self.moveIK(t)
      time.sleep(.1)

  def toolToRobotTransform(self, transform):
    '''
    Transforms from the tool frame to the robot frame
    '''
    # Rotate about z by 90 degrees
    t_z_90 = numpy.array([[0, -1, 0, 0], 
                           [1,  0, 0, 0], 
                           [0,  0, 1, 0], 
                           [0,  0, 0, 1]])

    # Rotate about x by -90 degrees
    t_y_m90 = numpy.array([[0, 0, 1, 0], 
                           [0, 1, 0, 0], 
                           [-1, 0, 0, 0], 
                           [0, 0, 0, 1]])

    # Frame change is rotate about z by 90, then x by -90
    frame_change = np.dot(t_z_90, t_y_m90)
    # Return H_wt dot H_tr
    transform[2, 3] += .351  # Ground to Robot Height Offset 
    transform[0, 3] += .35   # X Addition
    transform[1, 3] -= 0.15 # Y Offset
    return np.dot(transform, frame_change)

  def writeModFile(self, times, transforms, filename='seq.mod', toolframe=False, ik='fmin'):
    '''
    Writes a mod file for the trajectory described by t (the list of times) and
    transforms (the list of 4x4 homogeneous transforms the robots end effector)
    goes through.  The toolframe parameter indicates whether the Transformations
    are in the tools frame (toolframe=True) or Robots frame (toolframe=False).
    '''
    # Get pruned times and joints from function
    times, joints = self.generateJointTrajectoryFromIK(times, transforms, toolframe=toolframe, ik=ik)

    # Format data
    if ik is 'ikfast':
      data = [[time, [0] + j_vals.tolist()] for (time, j_vals) in zip(times, joints)]
    elif ik is 'fmin':
      data = [[time, j_vals.tolist()] for (time, j_vals) in zip(times, joints)]
    # Write File
    mod_file = single_trajectory_program(data, a_unit='radian', l_unit='meter')

    f = open(filename, 'w')
    f.write(mod_file)
    f.close()

  def generateJointTrajectoryFromIK(self, times, transforms, toolframe=False, ik='fmin'):
    '''
    Generates a Joint Trajectory from a list of 4x4 homogeneous transforms the robots'
    end effector is supposed to go through and the list of times it is supposed
    to be there.  The toolframe parameter indicates whether the Transformations
    are in the tools frame (toolframe=True) or Robots frame (toolframe=False).
    '''

    if ik is 'fmin':
      ikfun = self.fminIK
    else:
      ikfun = self.moveIK

    # Initialize returned lists
    joint_traj = []
    traj_times = []
    # Iterate over trajectory
    for time, t in zip(times, transforms):
      if toolframe:
        # Transform tool frame to robot frame
        t = self.toolToRobotTransform(t)
      # Get Solution
      sol = ikfun(t)
      # If solution exists, append it to the list
      if sol != None:
        joint_traj.append(sol)
        traj_times.append(time)

    return traj_times, joint_traj

  def segmentTransforms(self, trans_times, transforms):
    '''
    Takes a path in the form of 4x4 homogeneous transforms and corresponding times, 
    and segments the path to the motions where the tool path is close to the plastering
    wall.  Further filters these motions down to vertical motions, and those with more than
    80 unique timestamps.  Retruns a tuple of times and paths.
    @ Params : trans_times -> Timestamps corresponsding to transforms[i]
               transforms  -> 4x4 homogeneous transforms for tool path to follow
    @ Returns : trans2_times -> List of List of Timestamps corresponding to filtered transforms
                trans2       -> List of List of 4x4 Homogeneuous Transforms for tool to follow
                max_id       -> Location of longest path in trans2
    '''

    count  = 0
    trans1 = []
    trans1_times = []
    trans2 = []
    trans2_times = []
    for i in range(0,len(transforms)):
      if trans[i][1][3] > 2.3:
          count = count + 1
          trans1.append(trans[i])
          trans1_times.append(times[i])
    for i in range(0,len(trans1)-1):
      k = i
      count2 = 0
      x = []
      t = []
      while trans1[k+1][2][3] > trans1[k][2][3] and k < len(trans1)- 2:
        count2 = count2 + 1
        x.append(trans1[k])
        t.append(trans1_times[k])
        k = k + 1 
      if count2 > 80:
        trans2.append(x)
        trans2_times.append(t)

    lengths = []

    for i in range(0,len(trans2)-1):
      lengths.append([len(trans2[i]),i])

    l = np.array(lengths)
    l_sorted = sorted(l, key=lambda j: j[0],reverse=True) 
    max_id = l_sorted[0][1]

    vertical_move = trans2[max_id]
    vertical_move_times = trans2_times[max_id]

    return (trans2_times, trans2, max_id)

  def getVerticalTransforms(self, zs=.14, ze=2.6, y=2.35, x=.65, theta=-25):
    '''
    Function takes a start and end height to move the robot arm through and returns a list of 4x4
    homogeneous transforms (separated by 1 cm in the z direction) to move through IK.
    '''
    transforms = []
    z = zs
    t_i = numpy.array([[ 0.  ,  0.  ,  1.  ,  x   ],
                       [ 1.  ,  0.  ,  0.  ,  y   ],
                       [ 0.  ,  1.  ,  0.  ,  zs  ],
                       [ 0.  ,  0.  ,  0.  ,  1.  ]])
    r_x_25 = numpy.array([[cos(radians(theta)) , sin(radians(theta)) , 0 , 0],
                          [-sin(radians(theta) , cos(radians(theta)) , 0 , 0],
                          [0                   , 0                   , 1 , 0],
                          [0                   , 0                   , 0 , 1]])
    t_r = numpy.dot(t_i, r_x_25)

    while z < ze:
      t = t_r.copy()
      t[2][3] = z
      transforms.append(t)
      z = z + .01

    return transforms

  def getHorizontalTransforms(self, xs=.14, xe=2.6, y=2.35, z=1.5):
    '''
    Function takes a start and end height to move the robot arm through and returns a list of 4x4
    homogeneous transforms (separated by 1 cm in the z direction) to move through IK.
    '''
    transforms = []
    x = xs
    t_i = numpy.array([[ 0.  ,  0.  ,  1.  ,  xs  ],
                       [ 1.  ,  0.  ,  0.  ,  y   ],
                       [ 0.  ,  1.  ,  0.  ,  z   ],
                       [ 0.  ,  0.  ,  0.  ,  1.  ]])

    while x < xe:
      t = t_i.copy()
      t[0][3] = x
      transforms.append(t)
      x = x + .01

    return transforms

  def fminCost(self, q_guess, q_prev, t_goal):
    '''
    Cost Function for determining Cost of a move given a new desired IK solution
    @ Params : q_new  -> Joint values of new location
               q_prev -> Vector of joint values before move
    @ Returns : cost -> Euclidean Distance between start and end configuration
    '''
    alpha = 10
    beta = 5
    gamma = 0.5
    ret = self.robot.GetDOFValues()
    with self.env:
      self.robot.SetDOFValues(q_guess)
      # if(self.env.CheckCollision(self.robot, self.wall)):
      #   return 10000000000
      t_guess = self.manip.GetTransform()
      diff_car = t_goal - t_guess
      space_vec = abs(diff_car[0:3, 3])
      aa_guess = axisAngleFromRotationMatrix(t_guess[:3, :3])
      aa_goal =  axisAngleFromRotationMatrix(t_goal[:3, :3])
      rot_vec = abs(aa_guess - aa_goal)
      diff_js = abs(q_guess - q_prev)
      self.robot.SetDOFValues(ret)

    return alpha*sum(space_vec**2) + beta*sum(rot_vec**2) + gamma*sum(diff_js[1:]**2)



  def fminIK(self, t_goal):

    q_old = self.robot.GetDOFValues()
    sol = fmin(self.fminCost, q_old.tolist(), [q_old.tolist(), t_goal], disp=False)
    print(self.fminCost(sol, q_old, t_goal))
    return sol

  def fminTrajectory(self, transforms):
    traj = []
    c_points = []
    c_flag = False
    for t in transforms:
      s = self.fminIK(t)
      traj.append(s)
      self.robot.SetDOFValues(s)
      if self.env.CheckCollision(self.robot):
        c_flag = True
        c_points.append(t.copy())
      time.sleep(.1)

    if c_flag:
      print("There was a Collision! Happened @")
      print(c_points)
    return traj

  def fminTrackIKArm(self, times, transforms):
    final_traj = []
    final_times = []
    traj = self.fminTrajectory(transforms)
    for i in xrange(len(times)):
      self.robot.SetDOFValues([traj[i][0]], [0])
      arm_sol = self.moveIK(transforms[i])
      if arm_sol != None:
        traj_now = [traj[i][0]]
        traj_now.extend(arm_sol.tolist())
        final_traj.append(traj_now)
        final_times.append(times[i])

    return final_times, final_traj

  def testTrajectory(self, transforms, joint_trajectory):

    for i in xrange(len(joint_trajectory)):
      self.robot.SetDOFValues(joint_trajectory[i])
      t_now = self.manip.GetTransform()
      diff = transforms[i] - t_now
      print diff


if __name__ == '__main__':

  parser = ArgumentParser( description = """Python Script for Planning or Simulating IRB 6640 with an OpenRAVE Environment.""")
  parser.add_argument('-m', '--mode', default='sim', help='Mode For Script to Run in.  Options are sim and real (default is sim)')
  parser.add_argument('-t', '--trajectory', help='Name of Trajectory File to read as input.')
  parser.add_argument('-c', '--csv', help='Name of Mocap CSV File captured with 6 point trowel to load as data')
  parser.add_argument('-b', '--body', default='6 Point Trowel', help='Name of Body in Mocap CSV File captured to look for (default is 6 Point Trowel')

  args = parser.parse_args()

  robo = RoboHandler(args.mode)

  robo.robot.SetVisible(True)

  if args.csv != None:
    data = robo.getMocapData(args.csv, body=args.body)

  if args.trajectory != None:
    (trans, times) = robo.getMocapTraj(args.trajectory)
    (seg_move_times, seg_moves, max_id) = robo.segmentTransforms(times, trans)
    raw_input('Press enter to execute the trajectory in vertical direction:')
    robo.moveTransforms(seg_moves[max_id], toolframe=True)

  # Set Camera
  t = np.array([ [0, -1.0, 0, 2.25], [-1.0, 0, 0, 0], [0, 0, -1.0, 4.0], [0, 0, 0, 1.0] ])  
  robo.env.GetViewer().SetCamera(t)

  # Drop Into Shell
  import IPython
  IPython.embed()
  
  
