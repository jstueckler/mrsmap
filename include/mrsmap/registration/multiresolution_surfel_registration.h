/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 16.05.2011
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


#ifndef MULTIRESOLUTION_SURFEL_REGISTRATION_H_
#define MULTIRESOLUTION_SURFEL_REGISTRATION_H_

#include <gsl/gsl_multimin.h>

#include "mrsmap/map/multiresolution_surfel_map.h"

#include "mrsmap/utilities/geometry.h"

#include "octreelib/algorithm/downsample.h"

#include <list>


// takes in two map for which it estimates the rigid transformation with a coarse-to-fine strategy.
namespace mrsmap {

	class MultiResolutionSurfelRegistration {
	public:

		class Params {
		public:
			Params();
			~Params() {}

			void init();
			std::string toString();

			bool registerSurfels_, registerFeatures_;

			bool softSurfelAssociation_;

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

			bool parallel_;

			bool recover_associations_;


			float startResolution_;
			float stopResolution_;

			int pointFeatureMatchingNumNeighbors_;
			int pointFeatureMatchingThreshold_;
			float pointFeatureMatchingCoarseImagePosMahalDist_, pointFeatureMatchingFineImagePosMahalDist_;
			float pointFeatureWeight_;
			int pointFeatureMinNumMatches_;
			double calibration_f_, calibration_c1_, calibration_c2_;
			Eigen::Matrix3d K_, KInv_;
			bool debugFeatures_;

			double max_processing_time_;


		};


		MultiResolutionSurfelRegistration();
		MultiResolutionSurfelRegistration( const Params& params );
		~MultiResolutionSurfelRegistration() {}


	    class PoseAndLikelihood {
	    public:
	        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	        PoseAndLikelihood(  const int index, const Geometry::PoseAndVelocity pose ) {
	            pose_ = pose;
	            index_ = index;
	            likelihood_ = 0.f;
	            totalMatchedSurfels_ = 0;
	            discarded_ = false;
	            transformTgt2Src_ = pose.pose_.asMatrix4d();
	            transformSrc2Tgt_ = transformTgt2Src_.inverse();
	            rotationTgt2Src_ = transformTgt2Src_.topLeftCorner(3,3);
	            rotationTgt2SrcT_ = rotationTgt2Src_.transpose();
	        }

	        void updateTransform( const Eigen::Matrix4d & transform ) {
	            Eigen::Matrix4d inv = transform.inverse();
	            updateTransform( transform, inv );
	        }

	        void updateTransform( const Eigen::Matrix4d & transformTgt2Src, Eigen::Matrix4d & transformSrc2Tgt ) {
	            pose_.pose_ = Geometry::Pose( transformSrc2Tgt );
	            transformSrc2Tgt_ = transformSrc2Tgt;
	            transformTgt2Src_ = transformTgt2Src;
	            rotationTgt2Src_ = transformTgt2Src.topLeftCorner(3,3);
	            rotationTgt2SrcT_ = rotationTgt2Src_.transpose();
	        }

	        bool operator<(const PoseAndLikelihood& rhs) const
	        {
	            return this->likelihood_ < rhs.likelihood_;
	        }

	        Geometry::PoseAndVelocity pose_;
	        int index_;
	        Eigen::Matrix4d transformSrc2Tgt_;
	        Eigen::Matrix4d transformTgt2Src_;
	        Eigen::Matrix3d rotationTgt2Src_;
	        Eigen::Matrix3d rotationTgt2SrcT_;
	        float likelihood_;
	        bool discarded_;
	        unsigned int totalMatchedSurfels_;
	    };


		class SurfelAssociation {
		public:
			SurfelAssociation()
			: n_src_(NULL), src_(NULL), src_idx_(0), n_dst_(NULL), dst_(NULL), dst_idx_(0), match(0), loglikelihood_(0.0) {}
			SurfelAssociation( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_src, MultiResolutionSurfelMap::Surfel* src, unsigned int src_idx, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_dst, MultiResolutionSurfelMap::Surfel* dst, unsigned int dst_idx )
			: n_src_(n_src), src_(src), src_idx_(src_idx), n_dst_(n_dst), dst_(dst), dst_idx_(dst_idx), match(1), loglikelihood_(0.0) {}
			~SurfelAssociation() {}

			void revert() {

				spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_tmp_ = n_src_;
				MultiResolutionSurfelMap::Surfel* tmp_ = src_;
				unsigned int tmp_idx_ = src_idx_;

				n_src_ = n_dst_;
				src_ = dst_;
				src_idx_ = dst_idx_;

				n_dst_ = n_tmp_;
				dst_ = tmp_;
				dst_idx_ = tmp_idx_;

			}

			spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_src_;
			MultiResolutionSurfelMap::Surfel* src_;
			unsigned int src_idx_;
			spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_dst_;
			MultiResolutionSurfelMap::Surfel* dst_;
			unsigned int dst_idx_;

			Eigen::Matrix< double, 6, 1 > df_dx;
			Eigen::Matrix< double, 6, 6 > d2f, JSzJ;
			double error;
			double weight;
			int match;
			double loglikelihood_;


			// for Levenberg-Marquardt
			// (z - h)^T W (z - h)
			Eigen::Vector3d z, h;
			Eigen::Matrix< double, 3, 6 > dh_dx;
			Eigen::Matrix3d W;


		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		};

		typedef std::vector< SurfelAssociation, Eigen::aligned_allocator< SurfelAssociation > > SurfelAssociationList;


		class FeatureAssociation {
		public:
			FeatureAssociation()
			: src_idx_(0), dst_idx_(0), match(0), weight(1.0) {}
			FeatureAssociation( unsigned int src_idx, unsigned int dst_idx )
			: src_idx_(src_idx), dst_idx_(dst_idx), match(1), weight(1.0) {}
			~FeatureAssociation() {}

			unsigned int src_idx_;
			unsigned int dst_idx_;

			double error;
			int match;
			double weight;

			// for direct derivatives of error function
			Eigen::Matrix< double, 6, 1 > df_dx;
			Eigen::Matrix< double, 6, 6 > d2f, JSzJ;

			// AreNo
			Eigen::Vector3d landmark_pos, tmp_landmark_pos;	// estimation for 3D position in source-frame
			Eigen::Matrix<double, 6, 6> Hpp;
			Eigen::Matrix<double, 3, 6> Hpl;
			Eigen::Matrix<double, 3, 3> Hll;
			Eigen::Matrix<double, 6, 1> bp;
			Eigen::Vector3d				 bl;

		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		};

		typedef std::vector< FeatureAssociation, Eigen::aligned_allocator< FeatureAssociation > > FeatureAssociationList;


		class NodeLogLikelihood {
		public:
			NodeLogLikelihood()
			: n_(NULL), loglikelihood_(0.0) {}
			NodeLogLikelihood( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n )
			: n_(n), loglikelihood_(0.0) {}
			~NodeLogLikelihood() {}

			spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_;

			double loglikelihood_;

			SurfelAssociation surfelassocs_[MultiResolutionSurfelMap::NodeValue::num_surfels_];

		};

		typedef std::vector< NodeLogLikelihood > NodeLogLikelihoodList;


		void associateMapsBreadthFirstParallel( SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, algorithm::OcTreeSamplingVectorMap< float, MultiResolutionSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution, double searchDistFactor, double maxSearchDist, bool useFeatures );


		void associateNodeListParallel( SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >& nodes, int processDepth, Eigen::Matrix4d& transform, double searchDistFactor, double maxSearchDist, bool useFeatures );

		SurfelAssociationList revertSurfelAssociations( const SurfelAssociationList& surfelAssociations );

		void associatePointFeatures();

		double preparePointFeatureDerivatives( const Eigen::Matrix<double, 6, 1>& x, double qw, double mahaldist );


		std::pair< int, int > calculateNegLogLikelihood( double& logLikelihood, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node_src, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node_tgt, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate );
		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* calculateNegLogLikelihoodFeatureScoreN( double& logLikelihood, double& featureScore, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, const MultiResolutionSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate = false );
		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* calculateNegLogLikelihoodN( double& logLikelihood, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, const MultiResolutionSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate = false );
		bool calculateNegLogLikelihood( double& likelihood, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, const MultiResolutionSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate = false );

		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* getOccluder( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, const MultiResolutionSurfelMap& target, const Eigen::Matrix4d& transform );
		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* getOccluder2( const Eigen::Vector4f& p, const MultiResolutionSurfelMap& target, double z_similarity_factor );

		// transform from src to tgt
		double calculateInPlaneLogLikelihood( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_src, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_tgt, const Eigen::Matrix4d& transform, double normal_z_cov );


		double matchLogLikelihood( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );
		double matchLogLikelihoodKnownAssociations( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, NodeLogLikelihoodList& associations, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );

//		double matchLogLikelihoodKnownAssociations( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, SurfelAssociationList& surfelAssociations, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );
//		NodeLogLikelihoodList precalculateNomatchLogLikelihoodKnownAssociations( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, SurfelAssociationList& surfelAssociations, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );
//		double nomatchLogLikelihoodKnownAssociations( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, SurfelAssociationList& surfelAssociations, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );
//		void nomatchLogLikelihoodKnownAssociationsResetAssocs( const NodeLogLikelihoodList& nodes );
//		double nomatchLogLikelihoodKnownAssociationsPreCalc( const NodeLogLikelihoodList& nodes );

		double selfMatchLogLikelihood( MultiResolutionSurfelMap& target );

		bool estimateTransformationNewton( Eigen::Matrix4d& transform, int coarseToFineIterations, int fineIterations );
		bool estimateTransformationLevenbergMarquardt( Eigen::Matrix4d& transform, int maxIterations, SurfelAssociationList* surfelAssociations = NULL, bool knownAssociations = false, bool interpolate = false );
		bool estimateTransformationGaussNewton( Eigen::Matrix4d& transform, int maxIterations, SurfelAssociationList* surfelAssociations = NULL, bool knownAssociations = false, bool interpolate = false );
		bool estimateTransformationLevenbergMarquardtPF( Eigen::Matrix4d& transform, int maxIterations, double featureAssocMahalDist, double minDelta, bool resetFeatures, double& mu, double& nu );
		bool estimateTransformationGaussNewtonPF( Eigen::Matrix4d& transform, int maxIterations, double featureAssocMahalDist, double minDelta, bool resetFeatures );

		bool estimateTransformation( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesTargetPoints, int gradientIterations = 100, int coarseToFineIterations = 0, int fineIterations = 5 );


		bool estimatePoseCovariance( Eigen::Matrix< double, 6, 6 >& cov, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution );
		bool estimatePoseCovarianceLM( Eigen::Matrix< double, 6, 6 >& cov, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, SurfelAssociationList* surfelAssociations = NULL, bool knownAssociations = false );


		void setPriorPoseEnabled( bool enabled ) { params_.use_prior_pose_ = enabled; }
		void setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances );
		void setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 6 >& prior_pose_cov );



	    void improvedProposalMatchLogLikelihood( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
	            Geometry::PoseAndVelocity & pose, Geometry::PoseAndVelocity & poseOut, double& likelihood, const int numRegistrationStepsCoarse, const int numRegistrationStepsFine, bool add_pose_cov,
	            pcl::PointCloud<pcl::PointXYZRGB>* modelCloud = 0, pcl::PointCloud<pcl::PointXYZRGB>* sceneCloud = 0, const int minDepth = 10, const int maxDepth = 12 );

	    void improvedProposalMatchLogLikelihoodKnownAssociations( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
	            Geometry::PoseAndVelocity & pose, Geometry::PoseAndVelocity & poseOut, double & likelihood,
	            NodeLogLikelihoodList& associations, bool add_pose_cov, pcl::PointCloud<pcl::PointXYZRGB>* modelCloud = 0, pcl::PointCloud<pcl::PointXYZRGB>* sceneCloud = 0, const int minDepth = 10, const int maxDepth = 12 );

	    void getAssociations( SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
	            const Geometry::Pose & pose );

	    bool  registerPose( SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
	            Geometry::PoseWithCovariance & pose, Geometry::PoseWithCovariance & poseOut,
	             pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrSrc,
	             pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrTgt,
	             const int numRegistrationStepsCoarse, const int numRegistrationStepsFine,
	             bool regularizeRegistration = false,
	             const Eigen::Matrix6d& registrationPriorPoseCov = Eigen::Matrix6d::Identity() );

	    bool  registerPoseKnownAssociations( SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
	            Geometry::PoseWithCovariance & pose, Geometry::PoseWithCovariance & poseOut,
	             pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrSrc,
	             pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrTgt,
	             const int numRegistrationStepsCoarse, const int numRegistrationStepsFine,
	             bool regularizeRegistration = false,
	             const Eigen::Matrix6d& registrationPriorPoseCov = Eigen::Matrix6d::Identity() );



		Params params_;

		MultiResolutionSurfelMap* source_;
		MultiResolutionSurfelMap* target_;
//		SurfelAssociationList surfelAssociations_;
		FeatureAssociationList featureAssociations_;
		algorithm::OcTreeSamplingVectorMap< float, MultiResolutionSurfelMap::NodeValue > targetSamplingMap_;
		float lastWSign_;
		bool interpolate_neighbors_;

		pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondences_source_points_;
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondences_target_points_;

		// 2.5D --> 3D
		inline Eigen::Vector3d phi(const Eigen::Vector3d& m) const {
			Eigen::Vector3d tmp = m;
			tmp(0) /= tmp(2);
			tmp(1) /= tmp(2);
			tmp(2) = 1 / tmp(2);
			return params_.KInv_ * tmp;
		}

		// 3D --> 2.5D
		inline Eigen::Vector3d phiInv(const Eigen::Vector3d& lm) const {
			Eigen::Vector3d tmp = lm;
			double depth = lm(2);
			tmp = (params_.K_ * tmp).eval();
			tmp(0) /= depth;
			tmp(1) /= depth;
			tmp(2) /= depth * depth;
			return tmp;
		}

		// h( m , x)
		inline Eigen::Vector3d h(const Eigen::Vector3d& m, const Eigen::Matrix3d& rot, const Eigen::Vector3d& trnsl) const {
			return phiInv(rot * phi(m) + trnsl);
		}

		pcl::StopWatch processTimeWatch;

		NodeLogLikelihoodList lastNodeLogLikelihoodList_;

	protected:

		bool registrationErrorFunctionWithFirstDerivative( const Eigen::Matrix< double, 6, 1 >& x, double& f, Eigen::Matrix< double, 6, 1 >& df_dx, MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations );
		bool registrationErrorFunctionWithFirstAndSecondDerivative( const Eigen::Matrix< double, 6, 1 >& x, bool relativeDerivative, double& f, Eigen::Matrix< double, 6, 1 >& df_dx, Eigen::Matrix< double, 6, 6 >& d2f_dx2, MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations );

		bool registrationErrorFunctionLM( const Eigen::Matrix<double, 6, 1>& x, double& f, MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionSurfelRegistration::FeatureAssociationList& featureAssociations, double mahaldist );

		bool registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( const Eigen::Matrix< double, 6, 1 >& x, double& f, Eigen::Matrix< double, 6, 1 >& df, Eigen::Matrix< double, 6, 6 >& d2f, MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations );


	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};


};


#endif /* MULTIRESOLUTION_SURFEL_REGISTRATION_H_ */


