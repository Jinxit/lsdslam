import os
import subprocess

url = "http://vision.in.tum.de/rgbd/dataset/%s/rgbd_dataset_%s_%s.tgz"

names = ["freiburg3_sitting_static",
         "freiburg3_walking_static",
         "freiburg1_xyz",
         "freiburg1_rpy",
         "freiburg2_xyz",
         "freiburg2_rpy",
         "freiburg1_360",
         "freiburg1_floor",
         "freiburg1_desk",
         "freiburg1_desk2",
         "freiburg1_room",
         "freiburg2_360_hemisphere",
         "freiburg2_desk",
         "freiburg2_large_no_loop",
         "freiburg2_large_with_loop",
         "freiburg3_long_office_household",
         "freiburg2_pioneer_slam",
         "freiburg3_nostructure_notexture_far",
         "freiburg3_nostructure_notexture_near_withloop",
         "freiburg3_nostructure_texture_far",
         "freiburg3_nostructure_texture_near_withloop",
         "freiburg3_structure_notexture_far",
         "freiburg3_structure_notexture_near",
         "freiburg3_structure_texture_far",
         "freiburg3_structure_texture_near",
         "freiburg2_desk_with_person",
         "freiburg3_sitting_static",
         "freiburg3_sitting_halfsphere",
         "freiburg3_walking_halfsphere"]

def get_url(name):
    folder = name[:9]
    sequence = name[10:]
    return url % (folder, folder, sequence)

os.system("svn checkout https://svncvpr.in.tum.de/cvpr-ros-pkg/" +
          "trunk/rgbd_benchmark/rgbd_benchmark_tools")

try:
    os.makedirs("data/TUM")
except:
    pass

os.chdir("data/TUM")

for name in names:
    os.system("wget " + get_url(name) + " -O " + name + ".tgz")
    full_name = "rgbd_dataset_" + name
    os.system("tar -xvf " + name + ".tgz")
    os.system("rm -rf " + name + ".tgz")
    os.system("mv " + full_name + " " + name)
    os.system("python ../../rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py " +
              name + "/rgb.txt " + name + "/depth.txt > " + name + "/synced.txt")

