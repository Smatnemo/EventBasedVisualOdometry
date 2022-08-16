import cv2
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os
import progressbar


class Dataset_Class():
    
    def __init__(self, sequence, lidar=True, progress_bar=True, low_memory=True):
        
        self.lidar = lidar
        self.low_memory = low_memory
        # Load standard framees
        self.seq_dir = '/Smat/Nemo/dataset/kitti_odometry_dataset/original/sequences/{}/'.format(sequence)
        self.poses_dir = '/Smat/Nemo/dataset/kitti_odometry_dataset/original/poses/{}.txt'.format(sequence)
        
        self.left_image_files = os.listdir(self.seq_dir + 'image_0/')
        self.left_image_files.sort()
        self.right_image_files = os.listdir(self.seq_dir + 'image_1/')
        self.right_image_files.sort()
        self.velodyne_files = os.listdir(self.seq_dir + 'velodyne/')
        self.velodyne_files.sort()
        self.num_frames = len(self.left_image_files)
        self.lidar_path = self.seq_dir + 'velodyne/'
        
        # Load event frames
        # Load event frames for 10fps
        self.event_frame_dir_10 = '/Smat/Nemo/dataset/kitti_odometry_dataset/reconstruction_10/sequences/{}/'.format(sequence)
        self.left_event_frame_image_files_10 = os.listdir(self.event_frame_dir_10 + 'image_0/')
        self.left_event_frame_image_files_10.sort()
        self.right_event_frame_image_files_10 = os.listdir(self.event_frame_dir_10 + 'image_1/')
        self.right_event_frame_image_files_10.sort()
        
        # Load event frames for 30fps
        self.event_frame_dir_30 = '/Smat/Nemo/dataset/kitti_odometry_dataset/reconstruction_30/sequences/{}/'.format(sequence)
        self.left_event_frame_image_files_30 = os.listdir(self.event_frame_dir_30 + 'image_0/')
        self.left_event_frame_image_files_30.sort()
        self.right_event_frame_image_files_30 = os.listdir(self.event_frame_dir_30 + 'image_1/')
        self.right_event_frame_image_files_30.sort()
        
        
        poses = pd.read_csv(self.poses_dir, delimiter = ' ', header=None)
        
        self.gt = np.zeros((self.num_frames, 3, 4))
        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape(3,4)
            
        calib = pd.read_csv(self.seq_dir + 'calib.txt', delimiter = ' ', header = None, index_col = 0)
        
        self.P0 =np.array(calib.loc['P0:']).reshape((3, 4))
        self.P1 =np.array(calib.loc['P1:']).reshape((3, 4))
        self.P2 =np.array(calib.loc['P2:']).reshape((3, 4))
        self.P3 =np.array(calib.loc['P3:']).reshape((3, 4))
        self.Tr =np.array(calib.loc['Tr:']).reshape((3, 4))
        
        if low_memory:
            self.reset_frames()
            self.first_left_image = cv2.imread(self.seq_dir + 'image_0/'
                                              + self.left_image_files[0], 0)
            self.first_right_image = cv2.imread(self. seq_dir + 'image_1/'
                                               + self.right_image_files[0], 0)
            self.second_left_image = cv2.imread(self.seq_dir + 'image_0/'
                                               + self.left_image_files[1], 0)
            # Load event frames 10fps
            self.first_left_event_image_10 = cv2.imread(self.event_frame_dir_10 + 'image_0/'
                                               + self.left_event_frame_image_files_10[0], 0)
            self.first_right_event_image_10 = cv2.imread(self.event_frame_dir_10 + 'image_1/'
                                               + self.right_event_frame_image_files_10[0], 0)
            self.second_left_event_image_10 = cv2.imread(self.event_frame_dir_10 + 'image_0/'
                                              + self.left_event_frame_image_files_10[1], 0)
            
            # Load event frames 30fps
            self.first_left_event_image_30 = cv2.imread(self.event_frame_dir_30 + 'image_0/'
                                               + self.left_event_frame_image_files_30[0], 0)
            self.first_right_event_image_30 = cv2.imread(self.event_frame_dir_30 + 'image_1/'
                                               + self.right_event_frame_image_files_30[0], 0)
            self.second_left_event_image_30 = cv2.imread(self.event_frame_dir_30 + 'image_0/'
                                               + self.left_event_frame_image_files_30[1], 0)
            
            
        
            if lidar:
                self.first_pointcloud = np.fromfile(self.lidar_path+self.velodyne_files[0], 
                                                   dtype=np.float32, count=-1).reshape((-1, 4))
            self.imheight = self.first_left_image.shape[0]
            self.imwidth = self.first_left_image.shape[1]
        else:
            self.left_images = []
            self.right_images = []
            if progress_bar:
                bar = progressbar.ProgressBar(max_value=self.num_frames)
            for i, name_left in enumerate(self.left_image_files):
                name_right = self.right_image_files[i]
                self.left_images.append(cv2.imread(self.seq_dir + 'image_0/' + name_left))
                self.right_images.append(cv2.imread(self.seq_dir + 'image_1/' + name_right))
                if lidar:
                    pointcloud = np.fromfile(self.lidar_path + velodyne_file, dtype=np.float32).reshape((-1, 4))
                    self.pointclouds.append(pointcloud)
                if progress_bar:
                    bar.update(i+i)
            self.imheight = self.left_images[0].shape[0]
            self.imwidth = self.right_images[0].shape[1]
            self.first_left_image = self.left_images[0]
            self.first_right_image = self.right_images[0]
            self.second_left_image = self.left_images[1]
            if self.lidar:
                self.first_pointcloud = self.pointclouds[0]
                
    
    def reset_frames(self):
        self.left_images = (cv2.imread(self.seq_dir + 'image_0/' + name_left, 0) for name_left in self.left_image_files)
        self.right_images = (cv2.imread(self.seq_dir + 'image_1/' + name_right, 0) for name_right in self.right_image_files)
        if self.lidar:
            self.pointclouds = (np.fromfile(self.lidar_path + velodyne_file, dtype=np.float32, count=-1).reshape((-1, 4)) for velodyne_file in self.velodyne_files)
        
        pass

    