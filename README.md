###########################################################################################
#                                                                                         #
#  Multi-Resolution Surfel Map (MRSMap) Library                                           #
#  (c) 2010- Joerg Stueckler, Computer Science Institute VI, University of Bonn Germany   #
#                                                                                         #
#  Version: 0.3, 01.07.2014                                                              #
#                                                                                         #
###########################################################################################
 

This source package contains our implementation of Multi-Resolution Surfel Maps for
image registration [1], object modelling and tracking [1], and SLAM [2].

Note that the API is not stable yet and may be subject to change in future releases.
We are very thankful for any comments on the code, the methods, etc. 
Please contact Joerg Stueckler if you have comments or encounter any problems (email: joerg.stueckler.bw@gmail.com)

[1] Jörg Stückler and Sven Behnke. Model Learning and Real-Time Tracking using Multi-Resolution Surfel Maps.
    In Proceedings of the AAAI Conference on Artificial Intelligence, Toronto, Canada, July 2012. 

[2] Jörg Stückler and Sven Behnke. Integrating Depth and Color Cues for Dense Multi-Resolution Scene Mapping Using RGB-D Cameras.
    In Proceedings of the IEEE International Conference on Multisensor Fusion and Information Integration (MFI), Hamburg, Germany, September 2012.


Please cite reference [1] if you use the code, e.g., for image registration, object modelling or tracking.
For indoor SLAM purposes, [2] is the appropriate reference.

bibtex:
[1]
@inproceedings{stueckler2012_mrs_objects,
  author = {J\"org St\"uckler and Sven Behnke},
  title = {Model Learning and Real-Time Tracking using Multi-Resolution Surfel Maps},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-12)},
  year = {2012}  
}

[2]
@inproceedings{stueckler2012_mrs_slam,
  author = {J\"org St\"uckler and Sven Behnke},
  title = {Integrating Depth and Color Cues for Dense Multi-Resolution Scene Mapping Using RGB-D Cameras},
  booktitle = {Proceedings of the IEEE International Conference on Multisensor Fusion and Information Integration (MFI)},
  year = {2012}
}


Contact Information:

Joerg Stueckler, joerg.stueckler.bw@gmail.com



#######################################################################################################
#                                                                                                     #
#  INSTALLATION                                                                                       #
#                                                                                                     #
#######################################################################################################


The source code has been tested with Ubuntu 14.04 LTS Trusty Tahr and PCL 1.7.
Feel free to adapt the code to other Operating Systems, Linux distributions, or library dependencies.
You are very much invited to contribute your changes to the source code!

MRSMap depends on the following third party libraries:
- CMake >= 2.8
- PCL >= 1.7
- GSL
- TBB
- OpenCV >= 2.4.8
- CSparse
- OctreeLib (from github: https://github.com/ais-hilgert/octreelib)
- g2o (from github: https://github.com/RainerKuemmerle/g2o)
- Boost >= 1.54
- ClooG

Detailed build instructions can be found on https://github.com/ais-hilgert/mrsmap/wiki


Compile instructions:
mkdir build
cd build
cmake ..
make
sudo make install


There is a FindMRSMap.cmake script in the cmake/Modules folder to conveniently include the library into your project.



##########################################################################################################################
#                                                                                                                        #
#  Usage and Examples                                                                                                    #
#                                                                                                                        #
##########################################################################################################################


You can find some applications of the lib in the src/apps folder:

* evaluate_visualodometry: evaluation of successive image registration on the RGB-D benchmark dataset by Juergen Sturm et al.
  the dataset can be found at http://vision.in.tum.de/data/datasets/rgbd-dataset/download
  simply call: ./evaluate_visualodometry -i <data_folder> -s k 
  where data_folder is the root of the RGB-D dataset and contains a file associations.txt that associates depth (first column, e.g., associate.py depth.txt rgb.txt > associations.txt)
  and rgb images by timestamps. you should use the associate.py tool of the RGB-D benchmark dataset. evaluates for k frame skips.
  note that you can use some keys to change the visualization, for example you can switch through the resolutions using the "d" (next higher res) and "D" (next lower res) keys.

* evaluate_slam: evaluates SLAM performance on the RGB-D benchmark dataset. simply call ./evaluate_slam -i <data_folder> 
  further parameters are available, please check the command line help (-h).
  also requires a file associations.txt like the evaluage_visualodometry tool.
  note that slowdown over time comes from the visualization, not the SLAM method itself. pcl visualizer cant handle updating the pointclouds well and needs to redraw all maps when
  a new key frame is added. you can use keys to change the visualization as in evaluate_visualodometry.

* evaluate_pose_tracking: evaluates tracking performance on RGB-D datasets. call ./evaluate_pose_tracking -h to get further details.











   
