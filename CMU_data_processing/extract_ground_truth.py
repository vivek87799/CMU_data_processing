from os import listdir
from os.path import isfile, join

import argparse
import sys
import csv
import json
import pickle

import numpy as np

from gt_to_json import SkeletonPoseToJson


#########################################
############helper functions#############
#########################################
def read_json_file(file_path):
    with open(file_path, 'r') as js:
        json_data = json.load(js)
    return json_data
#########################################
#########################################

# Get the path for GT
def get_path_to_skeleton_data(haggle_file_path_GT, frame_number):
    frame_number = str(frame_number).zfill(8)
    print(frame_number)
    skeleton_path = join(haggle_file_path_GT, "body3DScene_"+frame_number+".json")
    print(skeleton_path)
    return skeleton_path

# Read all the files
# Get the skeleton in the format used for CMU-14 used for visualizing
# Store the data in the format used for visualizing

# Processing loop
def gen_ground_truth(haggle_file_path_GT, start_frame, end_frame):
    poses_all = []
    for frame_number in range(start_frame, end_frame):
        skeletons_path = get_path_to_skeleton_data(haggle_file_path_GT, frame_number)
        skeletons_data = read_json_file(skeletons_path)
        # skeleton_data["bodies"] is a list of subjects
        # Iterate over all the subjects 
        skeletons_pose_3D = SkeletonPoseToJson()
        for subject in skeletons_data["bodies"]:
            
            print(subject["id"])
            # Get the 3d joint coordinates
            keypoints_3d = np.asarray(subject['joints19'])
            # TODO mask and take only the required 15 joints
            print("before==", keypoints_3d.shape, keypoints_3d)
            keypoints_3d = keypoints_3d.reshape(4, -1, order='F')
            print("after==", keypoints_3d.shape, keypoints_3d)
            # Remove the scores in 4th row and the select the 15 keypoints that match between NTU(kinetic keypoints) and CMU(openpose keypoints) 
            keypoints_3d = keypoints_3d[:3, :15]
            print("15 keypoints", keypoints_3d.shape)
            skeletons_pose_3D.add_pose(subject["id"], np.transpose(keypoints_3d))
        with open("ground_truth.json", "a") as log:
            log.write(skeletons_pose_3D.toJson())
        # json.dump(skeletons_pose_3D, log)
        # poses_all.append(skeletons_pose_3D.toJson())
    # Write the json file
    
    


if __name__ == '__main__':
    gen_ground_truth("hdPose3d_stage1_coco19", 1800, 1802)