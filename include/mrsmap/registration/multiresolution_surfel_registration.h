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

	template <typename TMRSMap>
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
			SurfelAssociation( spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* n_src, typename TMRSMap::Surfel* src, unsigned int src_idx, spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* n_dst, typename TMRSMap::Surfel* dst, unsigned int dst_idx )
			: n_src_(n_src), src_(src), src_idx_(src_idx), n_dst_(n_dst), dst_(dst), dst_idx_(dst_idx), match(1), loglikelihood_(0.0) {}
			~SurfelAssociation() {}

			void revert() {

				spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* n_tmp_ = n_src_;
				typename TMRSMap::Surfel* tmp_ = src_;
				unsigned int tmp_idx_ = src_idx_;

				n_src_ = n_dst_;
				src_ = dst_;
				src_idx_ = dst_idx_;

				n_dst_ = n_tmp_;
				dst_ = tmp_;
				dst_idx_ = tmp_idx_;

			}

			spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* n_src_;
			typename TMRSMap::Surfel* src_;
			unsigned int src_idx_;
			spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* n_dst_;
			typename TMRSMap::Surfel* dst_;
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
			NodeLogLikelihood( spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* n )
			: n_(n), loglikelihood_(0.0) {}
			~NodeLogLikelihood() {}

			spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* n_;

			double loglikelihood_;

			SurfelAssociation surfelassocs_[TMRSMap::NodeValue::num_surfels_];

		};

		typedef std::vector< NodeLogLikelihood > NodeLogLikelihoodList;


		void associateMapsBreadthFirstParallel( SurfelAssociationList& surfelAssociations, TMRSMap& source, TMRSMap& target, algorithm::OcTreeSamplingVectorMap< float, typename TMRSMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution, double searchDistFactor, double maxSearchDist, bool useFeatures );


		void associateNodeListParallel( SurfelAssociationList& surfelAssociations, TMRSMap& source, TMRSMap& target, std::vector< spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* >& nodes, int processDepth, Eigen::Matrix4d& transform, double searchDistFactor, double maxSearchDist, bool useFeatures );

		SurfelAssociationList revertSurfelAssociations( const SurfelAssociationList& surfelAssociations );

		void associatePointFeatures();

		double preparePointFeatureDerivatives( const Eigen::Matrix<double, 6, 1>& x, double qw, double mahaldist );

		class AssociateFunctor {
		public:
			AssociateFunctor( tbb::concurrent_vector< SurfelAssociation >* associations, const Params& params, TMRSMap* source, TMRSMap* target, std::vector< spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* >* nodes, const Eigen::Matrix4d& transform, int processDepth, double searchDistFactor, double maxSearchDist, bool useFeatures );

			~AssociateFunctor();

			void operator()( const tbb::blocked_range<size_t>& r ) const;

			void operator()( spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >*& node ) const;

			tbb::concurrent_vector<SurfelAssociation >* associations_;
			Params params_;
			TMRSMap* source_;
			TMRSMap* target_;
			std::vector< spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* >* nodes_;
			Eigen::Matrix4d transform_;
			Eigen::Matrix4f transformf_;
			Eigen::Matrix3d rotation_;
			int process_depth_;
			float process_resolution_, search_dist_, search_dist2_;
			Eigen::Vector4f search_dist_vec_;
			bool use_features_;
			int num_vol_queries_, num_finds_, num_neighbors_;

			EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		};

		class GradientFunctor {
		public:
			GradientFunctor( SurfelAssociationList* assocList, const Params& params, double tx, double ty, double tz, double qx, double qy, double qz, double qw, bool relativeDerivatives, bool deriv2 = false, bool interpolate_neighbors = true, bool derivZ = false );

			~GradientFunctor();


			void operator()( const tbb::blocked_range<size_t>& r ) const;

			void operator()( SurfelAssociation& assoc ) const;


			Params params_;

			double tx, ty, tz, qx, qy, qz, qw;
			Eigen::Matrix4d currentTransform;
			Eigen::Vector3d ddiff_s_tx, ddiff_s_ty, ddiff_s_tz;
			Eigen::Matrix3d dR_qx, dR_qy, dR_qz;
			Eigen::Matrix3d dR_qxT, dR_qyT, dR_qzT;
			Eigen::Vector3d dt_tx, dt_ty, dt_tz;
		//	Eigen::Matrix3d cov_cc_add;
			Eigen::Matrix3d currentRotation;
			Eigen::Matrix3d currentRotationT;
			Eigen::Vector3d currentTranslation;

			// 2nd order derivatives
			Eigen::Matrix3d d2R_qxx, d2R_qxy, d2R_qxz, d2R_qyy, d2R_qyz, d2R_qzz;
			Eigen::Matrix3d d2R_qxxT, d2R_qxyT, d2R_qxzT, d2R_qyyT, d2R_qyzT, d2R_qzzT;

			// 1st and 2nd order derivatives on Z
			Eigen::Vector3d ddiff_dzsx, ddiff_dzsy, ddiff_dzsz;
			Eigen::Vector3d ddiff_dzmx, ddiff_dzmy, ddiff_dzmz;
			Eigen::Vector3d d2diff_qx_zsx, d2diff_qx_zsy, d2diff_qx_zsz;
			Eigen::Vector3d d2diff_qy_zsx, d2diff_qy_zsy, d2diff_qy_zsz;
			Eigen::Vector3d d2diff_qz_zsx, d2diff_qz_zsy, d2diff_qz_zsz;


			bool relativeDerivatives_;
			bool deriv2_, derivZ_;
			bool interpolate_neighbors_;

			SurfelAssociationList* assocList_;

			EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		};

		class GradientFunctorLM {
		public:


			GradientFunctorLM( SurfelAssociationList* assocList, const Params& params, double tx, double ty, double tz, double qx, double qy, double qz, double qw, bool derivs, bool derivZ = false, bool interpolate_neighbors = false );

			~GradientFunctorLM();


			void operator()( const tbb::blocked_range<size_t>& r ) const;

			void operator()( SurfelAssociation& assoc ) const;



			Eigen::Matrix4d currentTransform;

			Eigen::Vector3d currentTranslation;
			Eigen::Vector3d dt_tx, dt_ty, dt_tz;

			Eigen::Matrix3d currentRotation, currentRotationT;
			Eigen::Matrix3d dR_qx, dR_qy, dR_qz;

		    // 1st and 2nd order derivatives on Z
		    Eigen::Vector3d ddiff_dzsx, ddiff_dzsy, ddiff_dzsz;
		    Eigen::Vector3d ddiff_dzmx, ddiff_dzmy, ddiff_dzmz;
		    Eigen::Vector3d d2diff_qx_zsx, d2diff_qx_zsy, d2diff_qx_zsz;
		    Eigen::Vector3d d2diff_qy_zsx, d2diff_qy_zsy, d2diff_qy_zsz;
		    Eigen::Vector3d d2diff_qz_zsx, d2diff_qz_zsy, d2diff_qz_zsz;


			SurfelAssociationList* assocList_;

			Params params_;

			bool derivs_, derivZ_, interpolate_neighbors_;

			EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		};

		class GradientFunctorPointFeature {
		public:

			inline Eigen::Matrix<double, 3, 6> dh_dx(const Eigen::Vector3d& m,const Eigen::Matrix3d& rot, const Eigen::Vector3d& transl) const ;


			inline Eigen::Matrix3d dh_dm(const Eigen::Vector3d& m, const Eigen::Matrix3d& rot, const Eigen::Vector3d& transl) const ;

			GradientFunctorPointFeature(TMRSMap* source,
					TMRSMap* target,
					FeatureAssociationList* assocList,
					const Params& params,
					MultiResolutionSurfelRegistration<TMRSMap>* reg, double tx,
					double ty, double tz, double qx, double qy, double qz, double qw );

			~GradientFunctorPointFeature();


			double tx, ty, tz, qx, qy, qz, qw;
			Eigen::Matrix4d currentTransform;
			Eigen::Matrix3d dR_qx, dR_qy, dR_qz;

			FeatureAssociationList* assocList_;
			Params params_;

			TMRSMap* source_;
			TMRSMap* target_;

			MultiResolutionSurfelRegistration<TMRSMap>* reg_;

		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		};

		class MatchLogLikelihoodFunctor {
		public:

			MatchLogLikelihoodFunctor();
			MatchLogLikelihoodFunctor( NodeLogLikelihoodList* nodes, Params params, TMRSMap* source, TMRSMap* target, const Eigen::Matrix4d& transform, const Eigen::Matrix4d& lastTransform, const Eigen::Matrix< double, 6, 6 >& pcov, double delta_t, bool addPoseCov );

			~MatchLogLikelihoodFunctor();


			void precomputeCovAdd( TMRSMap* target );

			Eigen::Matrix< double, 3, 6 > diff_jac_for_pose( const Eigen::Vector3d& position ) const;

		//	Eigen::Matrix< double, 1, 3 > diff_normals_jac_for_angvel( const Eigen::Vector3d& normal1, const Eigen::Vector3d& normal2 ) const;

			double normalCovFromPoseCov( const Eigen::Vector3d& normal1, const Eigen::Vector3d& normal2, const Eigen::Matrix3d& poseCov ) const;

			void operator()( const tbb::blocked_range<size_t>& r ) const;

			void operator()( NodeLogLikelihood& node ) const;


			NodeLogLikelihoodList* nodes_;
			Params params_;
			TMRSMap* source_;
			TMRSMap* target_;
			Eigen::Matrix4d transform_;
			bool add_pose_cov_;

			double normalStd;
			double normalMinLogLikelihood;
			double normalMinLogLikelihoodSeenThrough;


			Eigen::Matrix4d targetToSourceTransform;
			Eigen::Matrix3d currentRotation;
			Eigen::Matrix3d currentRotationT;
			Eigen::Vector3d currentTranslation;

			Eigen::Matrix4d lastTransform;
			Eigen::Matrix3d lastRotation;

			Eigen::Matrix< double, 6, 6 > pcov_;
			double delta_t_;

			Eigen::Matrix3d cov_add_[MAX_REPRESENTABLE_DEPTH];

		};

		class MatchLogLikelihoodKnownAssociationsFunctor : public MatchLogLikelihoodFunctor {
		public:

			MatchLogLikelihoodKnownAssociationsFunctor( NodeLogLikelihoodList* associations, Params params, TMRSMap* source, TMRSMap* target, const Eigen::Matrix4d& transform, const Eigen::Matrix4d& lastTransform, const Eigen::Matrix< double, 6, 6 >& pcov, double delta_t, bool addPoseCov );

			~MatchLogLikelihoodKnownAssociationsFunctor();


			void operator()( const tbb::blocked_range<size_t>& r ) const;

			void operator()( MultiResolutionSurfelRegistration::NodeLogLikelihood& node ) const;

		};

		// assumes model is in the target, and scene in source has a node image
		class ShootThroughFunctor {
		public:

			ShootThroughFunctor(NodeLogLikelihoodList* nodes, Params params, TMRSMap* source, TMRSMap* target, const Eigen::Matrix4d& transform );

			~ShootThroughFunctor();


			void operator()( const tbb::blocked_range<size_t>& r ) const;

			void operator()( NodeLogLikelihood& node ) const;

			NodeLogLikelihoodList* nodes_;
			Params params_;
			TMRSMap* source_;
			TMRSMap* target_;
			Eigen::Matrix4d transform_, transform_inv_;
			Eigen::Vector3d camera_pos_;

		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		};

		class SelfMatchLogLikelihoodFunctor {
		public:
			SelfMatchLogLikelihoodFunctor( NodeLogLikelihoodList* nodes, Params params, TMRSMap* target );

			~SelfMatchLogLikelihoodFunctor();


			void operator()( const tbb::blocked_range<size_t>& r ) const;

			void operator()(NodeLogLikelihood& node ) const;


			NodeLogLikelihoodList* nodes_;
			Params params_;
			TMRSMap* target_;

			double normalStd;
			double normalMinLogLikelihood;

		};

		std::pair< int, int > calculateNegLogLikelihood( double& logLikelihood, spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* node_src, spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* node_tgt, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate );
		spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* calculateNegLogLikelihoodFeatureScoreN( double& logLikelihood, double& featureScore, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* node, const TMRSMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate = false );
		spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* calculateNegLogLikelihoodN( double& logLikelihood, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* node, const TMRSMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate = false );
		bool calculateNegLogLikelihood( double& likelihood, spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* node, const TMRSMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate = false );

		spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* getOccluder( spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* node, const TMRSMap& target, const Eigen::Matrix4d& transform );
		spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* getOccluder2( const Eigen::Vector4f& p, const TMRSMap& target, double z_similarity_factor );

		// transform from src to tgt
		double calculateInPlaneLogLikelihood( spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* n_src, spatialaggregate::OcTreeNode< float, typename TMRSMap::NodeValue >* n_tgt, const Eigen::Matrix4d& transform, double normal_z_cov );


		double matchLogLikelihood( TMRSMap& source, TMRSMap& target, Eigen::Matrix4d& transform, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );
		double matchLogLikelihoodKnownAssociations( TMRSMap& source, TMRSMap& target, Eigen::Matrix4d& transform, NodeLogLikelihoodList& associations, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );

//		double matchLogLikelihoodKnownAssociations( TMRSMap& source, TMRSMap& target, Eigen::Matrix4d& transform, SurfelAssociationList& surfelAssociations, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );
//		NodeLogLikelihoodList precalculateNomatchLogLikelihoodKnownAssociations( TMRSMap& source, TMRSMap& target, Eigen::Matrix4d& transform, SurfelAssociationList& surfelAssociations, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );
//		double nomatchLogLikelihoodKnownAssociations( TMRSMap& source, TMRSMap& target, Eigen::Matrix4d& transform, SurfelAssociationList& surfelAssociations, const Eigen::Matrix4d& lastTransform = Eigen::Matrix4d::Identity(), const Eigen::Matrix< double, 6, 6 >& pcov = Eigen::Matrix< double, 6, 6 >::Zero(), double delta_t = 1.0, bool addPoseCov = false );
//		void nomatchLogLikelihoodKnownAssociationsResetAssocs( const NodeLogLikelihoodList& nodes );
//		double nomatchLogLikelihoodKnownAssociationsPreCalc( const NodeLogLikelihoodList& nodes );

		double selfMatchLogLikelihood( TMRSMap& target );

		bool estimateTransformationNewton( Eigen::Matrix4d& transform, int coarseToFineIterations, int fineIterations );
		bool estimateTransformationLevenbergMarquardt( Eigen::Matrix4d& transform, int maxIterations, SurfelAssociationList* surfelAssociations = NULL, bool knownAssociations = false, bool interpolate = false );
		bool estimateTransformationGaussNewton( Eigen::Matrix4d& transform, int maxIterations, SurfelAssociationList* surfelAssociations = NULL, bool knownAssociations = false, bool interpolate = false );
		bool estimateTransformationLevenbergMarquardtPF( Eigen::Matrix4d& transform, int maxIterations, double featureAssocMahalDist, double minDelta, bool resetFeatures, double& mu, double& nu );
		bool estimateTransformationGaussNewtonPF( Eigen::Matrix4d& transform, int maxIterations, double featureAssocMahalDist, double minDelta, bool resetFeatures );

		bool estimateTransformation( TMRSMap& source, TMRSMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesTargetPoints, int gradientIterations = 100, int coarseToFineIterations = 0, int fineIterations = 5 );


		bool estimatePoseCovariance( Eigen::Matrix< double, 6, 6 >& cov, TMRSMap& source, TMRSMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution );
		bool estimatePoseCovarianceLM( Eigen::Matrix< double, 6, 6 >& cov, TMRSMap& source, TMRSMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, SurfelAssociationList* surfelAssociations = NULL, bool knownAssociations = false );


		void setPriorPoseEnabled( bool enabled ) { params_.use_prior_pose_ = enabled; }
		void setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances );
		void setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 6 >& prior_pose_cov );



	    void improvedProposalMatchLogLikelihood( TMRSMap& source, TMRSMap& target,
	            Geometry::PoseAndVelocity & pose, Geometry::PoseAndVelocity & poseOut, double& likelihood, const int numRegistrationStepsCoarse, const int numRegistrationStepsFine, bool add_pose_cov,
	            pcl::PointCloud<pcl::PointXYZRGB>* modelCloud = 0, pcl::PointCloud<pcl::PointXYZRGB>* sceneCloud = 0, const int minDepth = 10, const int maxDepth = 12 );

	    void improvedProposalMatchLogLikelihoodKnownAssociations( TMRSMap& source, TMRSMap& target,
	            Geometry::PoseAndVelocity & pose, Geometry::PoseAndVelocity & poseOut, double & likelihood,
	            NodeLogLikelihoodList& associations, bool add_pose_cov, pcl::PointCloud<pcl::PointXYZRGB>* modelCloud = 0, pcl::PointCloud<pcl::PointXYZRGB>* sceneCloud = 0, const int minDepth = 10, const int maxDepth = 12 );

	    void getAssociations( SurfelAssociationList& surfelAssociations, TMRSMap& source, TMRSMap& target,
	            const Geometry::Pose & pose );

	    bool  registerPose( SurfelAssociationList& surfelAssociations, TMRSMap& source, TMRSMap& target,
	            Geometry::PoseWithCovariance & pose, Geometry::PoseWithCovariance & poseOut,
	             pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrSrc,
	             pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrTgt,
	             const int numRegistrationStepsCoarse, const int numRegistrationStepsFine,
	             bool regularizeRegistration = false,
	             const Eigen::Matrix6d& registrationPriorPoseCov = Eigen::Matrix6d::Identity() );

	    bool  registerPoseKnownAssociations( SurfelAssociationList& surfelAssociations, TMRSMap& source, TMRSMap& target,
	            Geometry::PoseWithCovariance & pose, Geometry::PoseWithCovariance & poseOut,
	             pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrSrc,
	             pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrTgt,
	             const int numRegistrationStepsCoarse, const int numRegistrationStepsFine,
	             bool regularizeRegistration = false,
	             const Eigen::Matrix6d& registrationPriorPoseCov = Eigen::Matrix6d::Identity() );



		Params params_;

		TMRSMap* source_;
		TMRSMap* target_;
//		SurfelAssociationList surfelAssociations_;
		FeatureAssociationList featureAssociations_;
		algorithm::OcTreeSamplingVectorMap< float, typename TMRSMap::NodeValue > targetSamplingMap_;
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

	template <typename TMRSMap> bool pointOccluded( const Eigen::Vector4f& p, const TMRSMap& target, double z_similarity_factor );
	template <typename TMRSMap> bool pointSeenThrough( const Eigen::Vector4f& p, const TMRSMap& target, double z_similarity_factor, bool markSeenThrough = false );


};

#include <mrsmap/registration/impl/multiresolution_surfel_registration.hpp>

#endif /* MULTIRESOLUTION_SURFEL_REGISTRATION_H_ */


