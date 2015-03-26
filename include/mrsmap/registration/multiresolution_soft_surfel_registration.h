/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 02.07.2014
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


// if you build on soft associations as implemented here, cite the paper
// David Droeschel, Jörg Stückler, and Sven Behnke:
// Local Multi-Resolution Representation for 6D Motion Estimation and Mapping with a Continuously Rotating 3D Laser Scanner
// In Proceedings of IEEE International Conference on Robotics and Automation (ICRA), Hong Kong, May 2014.
// until a better own publication appears

#ifndef MULTIRESOLUTION_SOFT_SURFEL_REGISTRATION_H_
#define MULTIRESOLUTION_SOFT_SURFEL_REGISTRATION_H_

#include <gsl/gsl_multimin.h>

#include "mrsmap/map/multiresolution_surfel_map.h"

#include "mrsmap/utilities/geometry.h"

#include "octreelib/algorithm/downsample.h"

#include <list>
#include <mrsmap/mrsmap_api.h>


namespace mrsmap {

	class MRSMAP_API MultiResolutionSoftSurfelRegistration {
	public:

		class Params {
		public:
			Params();
			~Params() {}

			void init();
			std::string toString();

			bool use_prior_pose_;
			Eigen::Matrix< double, 6, 1 > prior_pose_mean_;
			Eigen::Matrix< double, 6, 6 > prior_pose_invcov_;

			bool add_smooth_pos_covariance_;
			float smooth_surface_cov_factor_;
			double surfel_match_angle_threshold_;
			unsigned int registration_min_num_surfels_;
			double max_feature_dist2_;
			bool use_features_;
			bool match_likelihood_use_color_, registration_use_color_;
			bool match_likelihood_use_normals_;
			double luminance_damp_diff_, luminance_reg_threshold_;
			double color_damp_diff_, color_reg_threshold_;
			double occlusion_z_similarity_factor_;
			unsigned int image_border_range_;
			double interpolation_cov_factor_;
			int model_visibility_max_depth_;

			double prior_prob_;

			bool parallel_;

			float startResolution_;
			float stopResolution_;


			double max_processing_time_;

			double model_num_points_, scene_num_points_;

		};


		MultiResolutionSoftSurfelRegistration();
		MultiResolutionSoftSurfelRegistration( const Params& params );
		~MultiResolutionSoftSurfelRegistration() {}



		class SurfelAssociations {
		public:
			SurfelAssociations()
			: n_src_(NULL), src_(NULL), src_idx_(0) {}
			SurfelAssociations( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_src, MultiResolutionSurfelMap::Surfel* src, unsigned int src_idx )
			: n_src_(n_src), src_(src), src_idx_(src_idx) {}
			~SurfelAssociations() {}

			spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_src_;
			MultiResolutionSurfelMap::Surfel* src_;
			unsigned int src_idx_;

			double sigma, sigma2;

			class Surfel2SurfelAssociation {
			public:
				Surfel2SurfelAssociation()
				: n_dst_(NULL), dst_(NULL), dst_idx_(0), match(0), loglikelihood_(0.0) {}

				Surfel2SurfelAssociation( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_dst, MultiResolutionSurfelMap::Surfel* dst, unsigned int dst_idx )
				: n_dst_(n_dst), dst_(dst), dst_idx_(dst_idx), match(1), loglikelihood_(0.0) {}

				spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_dst_;
				MultiResolutionSurfelMap::Surfel* dst_;
				unsigned int dst_idx_;

				double error;

				Eigen::Vector3d z, h;
				Eigen::Matrix< double, 3, 6 > dh_dx;
				Eigen::Matrix3d W;

				double weight;
				int match;
				double loglikelihood_;

			public:
				EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
			};

			typedef std::vector< Surfel2SurfelAssociation, Eigen::aligned_allocator< Surfel2SurfelAssociation > > Surfel2SurfelAssociationsList;

			Surfel2SurfelAssociationsList associations_;


		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		};

		typedef std::vector< SurfelAssociations, Eigen::aligned_allocator< SurfelAssociations > > SurfelAssociationsList;




		void associateMapsBreadthFirstParallel( SurfelAssociationsList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, algorithm::OcTreeSamplingVectorMap< float, MultiResolutionSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution, double searchDistFactor, double maxSearchDist );
		void associateNodeListParallel( SurfelAssociationsList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >& nodes, int processDepth, Eigen::Matrix4d& transform, double searchDistFactor, double maxSearchDist );


		bool estimateTransformationLevenbergMarquardt( Eigen::Matrix4d& transform, int maxIterations );

		bool estimateTransformation( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesTargetPoints, int iterations = 100 );


		bool estimatePoseCovarianceLM( Eigen::Matrix< double, 6, 6 >& cov, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, SurfelAssociationsList* surfelAssociations = NULL, bool knownAssociations = false );


		void setPriorPoseEnabled( bool enabled ) { params_.use_prior_pose_ = enabled; }
		void setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances );
		void setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 6 >& prior_pose_cov );


		Params params_;

		MultiResolutionSurfelMap* source_;
		MultiResolutionSurfelMap* target_;
		float lastWSign_;

		pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondences_source_points_;
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondences_target_points_;

		pcl::StopWatch processTimeWatch;


	protected:

		bool registrationErrorFunctionLM( const Eigen::Matrix<double, 6, 1>& x, double wsign, double& f, MultiResolutionSoftSurfelRegistration::SurfelAssociationsList& surfelAssociations );
		bool registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( const Eigen::Matrix< double, 6, 1 >& x, double wsign, double& f, Eigen::Matrix< double, 6, 1 >& df, Eigen::Matrix< double, 6, 6 >& d2f, MultiResolutionSoftSurfelRegistration::SurfelAssociationsList& surfelAssociations );


	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};

};


#endif /* MULTIRESOLUTION_SOFT_SURFEL_REGISTRATION_H_ */


