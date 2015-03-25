/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 12.12.2011
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of University of Bonn, Computer Science Institute 
 *     VI nor the names of its contributors may be used to endorse or 
 *     promote products derived from this software without specific 
 *     prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef SLAM_H_
#define SLAM_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


#include "g2o/core/sparse_optimizer.h"

#include "g2o/types/slam3d/types_slam3d.h"

#include "g2o/core/block_solver.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"


typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> >  SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
typedef g2o::LinearSolverCholmod<SlamBlockSolver::PoseMatrixType> SlamLinearCholmodSolver;
//typedef std::tr1::unordered_map<int, g2o::HyperGraph::Vertex*>     VertexIDMap;
typedef std::set<g2o::HyperGraph::Edge*> EdgeSet;

#include <mrsmap/map/multiresolution_surfel_map.h>

#include <set>

#include <pcl/PolygonMesh.h>




namespace mrsmap {


	class KeyFrame {
	public:
		KeyFrame() { sumLogLikelihood_ = 0.0; numEdges_ = 0.0; }
		~KeyFrame() {}

		boost::shared_ptr< MultiResolutionSurfelMap > map_;
		unsigned int nodeId_;

		pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_;
		cv::Mat img_rgb;

		double sumLogLikelihood_;
		double numEdges_;

		std::set< unsigned int > checkedKeyFrames_;

	};

	class IntermediateFrame {
	public:
		IntermediateFrame() {}
		~IntermediateFrame() {}

		unsigned int nodeId_;

	};


	class SLAM {
	public:

		class Params {
		public:
			Params();
			~Params() {}

			bool usePointFeatures_;
			bool debugPointFeatures_;
			unsigned int downsamplingMRSMapImage_;
			bool regularizePose_;
			double map_dist_dependency_;
			bool connectRandom_;
			bool optimize_;
			double loglikelihood_threshold_;
			bool optimize_full_trajectory_;

			double pose_close_angle_, pose_close_dist_;
			double pose_far_angle_, pose_far_dist_;

		};

		SLAM();
		virtual ~SLAM();

		unsigned int addKeyFrame( unsigned int v_prev_id, boost::shared_ptr< KeyFrame >& keyFrame, const Eigen::Matrix4d& transform );
		unsigned int addIntermediateFrame( unsigned int kf_ref_id, boost::shared_ptr< IntermediateFrame >& intermediateFrame, const Eigen::Matrix4d& transform );

		bool addEdge( unsigned int v1_id, unsigned int v2_id, float register_start_resolution, float register_stop_resolution, bool checkMatchingLikelihood = true );
		bool addEdge( unsigned int v1_id, unsigned int v2_id, const Eigen::Matrix4d& transformGuess, float register_start_resolution, float register_stop_resolution, bool checkMatchingLikelihood = true );
		bool addEdge( unsigned int v1_id, unsigned int v2_id, const Eigen::Matrix4d& transform, const Eigen::Matrix< double, 6, 6 >& covariance );

		bool poseIsClose( const Eigen::Matrix4d& transform );
		bool poseIsFar( const Eigen::Matrix4d& transform );

		bool addImage( const cv::Mat& img_rgb, const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& pointCloudIn, float startResolution, float stopResolution, float minResolution, bool storeCloud = false );

		bool connectClosePoses( float register_start_resolution, float register_stop_resolution, bool random = false );

		bool refineEdge( g2o::EdgeSE3* edge, float register_start_resolution, float register_stop_resolution );


		void refine( unsigned int refineIterations, unsigned int optimizeIterations, float register_start_resolution, float register_stop_resolution );
		void refineInConvexHull( unsigned int refineIterations, unsigned int optimizeIterations, float register_start_resolution, float register_stop_resolution, float minResolution, const Eigen::Matrix4d& referenceTransform, float minHeight, float maxHeight, std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > convexHull );
		void refineWorstEdges( float fraction, float register_start_resolution, float register_stop_resolution );
		void refineWorstEdgeRandom( float register_start_resolution, float register_stop_resolution );

		void dumpError();

		boost::shared_ptr< MultiResolutionSurfelMap > getMap( const Eigen::Matrix4d& referenceTransform, float minResolution );
		boost::shared_ptr< pcl::PointCloud< pcl::PointXYZRGB > > getMapCloud( const Eigen::Matrix4d& referenceTransform );
		boost::shared_ptr< MultiResolutionSurfelMap > getMapInConvexHull( const Eigen::Matrix4d& referenceTransform, float minResolution, float minHeight, float maxHeight, std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > convexHull );
		boost::shared_ptr< pcl::PolygonMesh > getMapMesh( unsigned int res, unsigned int repetitions );

		// used for all kinds of maps
		boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator > imageAllocator_[2];

		// only for the current frame
		boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > > treeNodeAllocator_[2];

		unsigned int currentAllocIdx_;

		g2o::SparseOptimizer* optimizer_;

		std::vector< boost::shared_ptr< KeyFrame > > keyFrames_;
		std::vector< boost::shared_ptr< IntermediateFrame > > intermediateFrames_;
		std::map< unsigned int, boost::shared_ptr< KeyFrame > > keyFrameNodeMap_;
		unsigned int referenceKeyFrameId_;

		// wrt reference key frame pose
		Eigen::Matrix4d lastTransform_, lastFrameTransform_; // the latter is the pose used to build the lastFrameMap (in the map reference frame!)

		boost::shared_ptr< MultiResolutionSurfelMap > lastFrameMap_;

		Eigen::Matrix4d deltaIncTransform_;

		Params params_;

		double tracking_time_, generate_keyview_time_, connect_time_, optimize_time_;

	};

};


#endif
