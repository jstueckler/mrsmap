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

#include "mrsmap/registration/multiresolution_soft_surfel_registration.h"

#include <mrsmap/utilities/utilities.h>

#include <g2o/types/slam3d/dquat2mat.h>

#include <deque>

#include <fstream>

#include <tbb/tbb.h>

#include <cmath>

#include <mrsmap/utilities/logging.h>

using namespace mrsmap;

// if you build on soft associations as implemented here, cite the paper
// David Droeschel, Jörg Stückler, and Sven Behnke:
// Local Multi-Resolution Representation for 6D Motion Estimation and Mapping with a Continuously Rotating 3D Laser Scanner
// In Proceedings of IEEE International Conference on Robotics and Automation (ICRA), Hong Kong, May 2014.
// until a better own publication appears



MultiResolutionSoftSurfelRegistration::Params::Params() {
	init();
}

void MultiResolutionSoftSurfelRegistration::Params::init() {

	// defaults
	max_processing_time_ = std::numeric_limits<double>::max();


	use_prior_pose_ = false;
	prior_pose_mean_ = Eigen::Matrix< double, 6, 1 >::Zero();
	prior_pose_invcov_ = Eigen::Matrix< double, 6, 6 >::Identity();

	add_smooth_pos_covariance_ = true;
	smooth_surface_cov_factor_ = 0.001f;

	surfel_match_angle_threshold_ = 0.5;
	registration_min_num_surfels_ = 0;
	max_feature_dist2_ = 0.1;
	use_features_ = false;

	match_likelihood_use_color_ = true;
	luminance_damp_diff_ = 0.5;
	color_damp_diff_ = 0.1;
	model_visibility_max_depth_ = 12;

	registration_use_color_ = false;
	luminance_reg_threshold_ = 0.5;
	color_reg_threshold_ = 0.1;

	occlusion_z_similarity_factor_ = 0.02f;
	image_border_range_ = 40;

	prior_prob_ = 0.1;

	parallel_ = true;

	startResolution_ = 0.0125f;
	stopResolution_ = 0.2f;


}

std::string MultiResolutionSoftSurfelRegistration::Params::toString() {

	std::stringstream retVal;

	retVal << "use_prior_pose: " << (use_prior_pose_ ? 1 : 0) << std::endl;
	retVal << "prior_pose_mean: " << prior_pose_mean_.transpose() << std::endl;
	retVal << "prior_pose_invcov: " << prior_pose_invcov_ << std::endl;

	return retVal.str();

}



MultiResolutionSoftSurfelRegistration::MultiResolutionSoftSurfelRegistration() {

}


MultiResolutionSoftSurfelRegistration::MultiResolutionSoftSurfelRegistration( const Params& params ) {

	params_ = params;

}


void MultiResolutionSoftSurfelRegistration::setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances ) {

	params_.use_prior_pose_ = enabled;
	params_.prior_pose_mean_ = prior_pose_mean;
	params_.prior_pose_invcov_ = Eigen::DiagonalMatrix< double, 6 >( prior_pose_variances ).inverse();

}


void MultiResolutionSoftSurfelRegistration::setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 6 >& prior_pose_cov ) {

	params_.use_prior_pose_ = enabled;
	params_.prior_pose_mean_ = prior_pose_mean;
	params_.prior_pose_invcov_ = prior_pose_cov.inverse();

}



void MultiResolutionSoftSurfelRegistration::associateMapsBreadthFirstParallel( MultiResolutionSoftSurfelRegistration::SurfelAssociationsList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, algorithm::OcTreeSamplingVectorMap< float, MultiResolutionSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution, double searchDistFactor, double maxSearchDist ) {


	target.distributeAssociatedFlag();

	int maxDepth = std::min( source.octree_->max_depth_, target.octree_->max_depth_ );

	// start at coarsest resolution
	// if all children associated, skip the node,
	// otherwise
	// - if already associated from previous iteration, search in local neighborhood
	// - if not associated in previous iteration, but parent has been associated, choose among children of parent's match
	// - otherwise, search in local volume for matches

	int countNodes = 0;
	for( int d = maxDepth; d >= 0; d-- ) {

		const float processResolution = source.octree_->volumeSizeForDepth( d );

		if( processResolution < minResolution || processResolution > maxResolution ) {
			continue;
		}

		countNodes += targetSamplingMap[d].size();

	}
	surfelAssociations.reserve( countNodes );

	params_.model_num_points_ = 0;
	params_.scene_num_points_ = 0;

	for( int d = maxDepth; d >= 0; d-- ) {

		const float processResolution = source.octree_->volumeSizeForDepth( d );

		if( processResolution < minResolution || processResolution > maxResolution ) {
			continue;
		}

		source_->buildSamplingMap();

		for( unsigned int i = 0; i < source_->samplingMap_[d].size(); i++ ) {
			for( unsigned int v = 0; v < MultiResolutionSurfelMap::NodeValue::num_surfels_; v++ )
				params_.model_num_points_ += source_->samplingMap_[d][i]->value_.surfels_[v].num_points_;
		}

		for( unsigned int i = 0; i < target_->samplingMap_[d].size(); i++ ) {
			for( unsigned int v = 0; v < MultiResolutionSurfelMap::NodeValue::num_surfels_; v++ )
				params_.scene_num_points_ += target_->samplingMap_[d][i]->value_.surfels_[v].num_points_;
		}

		associateNodeListParallel( surfelAssociations, source, target, targetSamplingMap[d], d, transform, searchDistFactor, maxSearchDist );

	}


}


class SoftAssociateFunctor {
public:
	SoftAssociateFunctor( tbb::concurrent_vector< MultiResolutionSoftSurfelRegistration::SurfelAssociations >* associations, const MultiResolutionSoftSurfelRegistration::Params& params, MultiResolutionSurfelMap* source, MultiResolutionSurfelMap* target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >* nodes, const Eigen::Matrix4d& transform, int processDepth, double searchDistFactor, double maxSearchDist ) {
		associations_ = associations;
		params_ = params;
		source_ = source;
		target_ = target;
		nodes_ = nodes;
		transform_ = transform;
		transformf_ = transform.cast<float>();
		rotation_ = transform.block<3,3>(0,0);

		process_depth_ = processDepth;
		process_resolution_ = source_->octree_->volumeSizeForDepth( processDepth );
		search_dist_ = std::min( searchDistFactor*process_resolution_, maxSearchDist );
		search_dist2_ = search_dist_*search_dist_;
		search_dist_vec_ = Eigen::Vector4f( search_dist_, search_dist_, search_dist_, 0.f );

		num_vol_queries_ = 0;
		num_finds_ = 0;
		num_neighbors_ = 0;

	}

	~SoftAssociateFunctor() {}

	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*nodes_)[i]);
	}


	void operator()( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >*& node ) const {

		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = node;

		if( n->value_.associated_ == -1 )
			return;

		// all children associated?
		int numAssociatedChildren = 0;
		int numChildren = 0;
		if( n->type_ != spatialaggregate::OCTREE_MAX_DEPTH_BRANCHING_NODE ) {
			for( unsigned int i = 0; i < 8; i++ ) {
				if( n->children_[i] ) {
					numChildren++;
					if( n->children_[i]->value_.associated_ == 1 )
						numAssociatedChildren++;
				}
			}

			if( numChildren > 0 && numAssociatedChildren > 0 ) {
				n->value_.associated_ = 1;
				return;
			}
		}

		// check if surfels exist and can be associated by view direction
		// use only one best association per node
		MultiResolutionSoftSurfelRegistration::SurfelAssociations assoc;

		bool hasSurfel = false;

		// check if a surfels exist
		for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

			// if image border points fall into this node, we must check the children_
			if( !n->value_.surfels_[i].applyUpdate_ ) {
				continue;
			}

			if( n->value_.surfels_[i].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
				continue;
			}

			hasSurfel = true;
		}

		if( hasSurfel ) {


			std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* > neighbors;

			neighbors.reserve(50);

			Eigen::Vector4f npos = n->getCenterPosition();
			npos(3) = 1.f;
			Eigen::Vector4f npos_match_src = transformf_ * npos;

			// if direct look-up fails, perform a region query
			// in case there is nothing within the volume, the query will exit early

			Eigen::Vector4f minPosition = npos_match_src - search_dist_vec_;
			Eigen::Vector4f maxPosition = npos_match_src + search_dist_vec_;

			source_->octree_->getAllNodesInVolumeOnDepth( neighbors, minPosition, maxPosition, process_depth_, false );

			if( neighbors.size() == 0 ) {

				n->value_.association_ = NULL;
				n->value_.associated_ = 0;

				return;
			}


			for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

				const MultiResolutionSurfelMap::Surfel& surfel = n->value_.surfels_[i];

				if( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
					continue;
				}

				// transform surfel mean with current transform and find corresponding node in source for current resolution
				// find corresponding surfel in node via the transformed view direction of the surfel

				Eigen::Vector4d pos;
				pos.block<3,1>(0,0) = surfel.mean_.block<3,1>(0,0);
				pos(3,0) = 1.f;

				Eigen::Vector4d pos_match_src = transform_ * pos;
				Eigen::Vector3d dir_match_src = rotation_ * surfel.initial_view_dir_;

				// iterate through neighbors of the directly associated node to eventually find a better match
				for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >::iterator nit = neighbors.begin(); nit != neighbors.end(); ++nit ) {

					spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_src = *nit;

					if( !n_src )
						continue;

					if( n->value_.border_ != n_src->value_.border_ )
						continue;

					MultiResolutionSurfelMap::Surfel* bestMatchSurfel = NULL;
					int bestMatchSurfelIdx = -1;
					for( unsigned int k = 0; k < MultiResolutionSurfelMap::NodeValue::num_surfels_; k++ ) {

						const MultiResolutionSurfelMap::Surfel& srcSurfel = n_src->value_.surfels_[k];

						if( srcSurfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
							continue;

						bestMatchSurfel = &n_src->value_.surfels_[k];
						bestMatchSurfelIdx = k;

						// calculate error metric for matching surfels
						double dist_pos2 = (bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0)).squaredNorm();

						if( dist_pos2 > search_dist2_ )
							continue;

						float featureDist = 0.f;
						if( use_features_) {
							featureDist = surfel.agglomerated_shape_texture_features_.distance( bestMatchSurfel->agglomerated_shape_texture_features_ );
							if( featureDist > params_.max_feature_dist2_ )
								continue;
						}

						if( params_.registration_use_color_ ) {
							Eigen::Matrix< double, 3, 1 > diff;
							diff = bestMatchSurfel->mean_.block<3,1>(3,0) - surfel.mean_.block<3,1>(3,0);
							if( fabs(diff(0)) > params_.luminance_reg_threshold_ )
								continue;
							if( fabs(diff(1)) > params_.color_reg_threshold_ )
								continue;
							if( fabs(diff(2)) > params_.color_reg_threshold_ )
								continue;
						}

						MultiResolutionSoftSurfelRegistration::SurfelAssociations::Surfel2SurfelAssociation softassoc;

						assoc.n_src_ = n;
						assoc.src_ = &n->value_.surfels_[i];
						assoc.src_idx_ = i;
						assoc.sigma = 0.5*process_resolution_;
						assoc.sigma2 = assoc.sigma*assoc.sigma;

						softassoc.n_dst_ = n_src;
						softassoc.dst_ = bestMatchSurfel;
						softassoc.dst_idx_ = bestMatchSurfelIdx;
						softassoc.match = 1;
						softassoc.weight = 1.0;

						assoc.associations_.push_back( softassoc );

					}

				}

			}

		}


		if( assoc.associations_.size() > 0 ) {

			n->value_.association_ = assoc.associations_[0].n_dst_;
			n->value_.associated_ = 1;
			n->value_.assocSurfelIdx_ = assoc.src_idx_;
			n->value_.assocSurfelDstIdx_ = assoc.associations_[0].dst_idx_;

			associations_->push_back( assoc );

		}
		else {

			n->value_.association_ = NULL;
			n->value_.associated_ = 0;

		}


	}


	tbb::concurrent_vector< MultiResolutionSoftSurfelRegistration::SurfelAssociations >* associations_;
	MultiResolutionSoftSurfelRegistration::Params params_;
	MultiResolutionSurfelMap* source_;
	MultiResolutionSurfelMap* target_;
	std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >* nodes_;
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


void MultiResolutionSoftSurfelRegistration::associateNodeListParallel( MultiResolutionSoftSurfelRegistration::SurfelAssociationsList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >& nodes, int processDepth, Eigen::Matrix4d& transform, double searchDistFactor, double maxSearchDist ) {

	tbb::concurrent_vector< MultiResolutionSoftSurfelRegistration::SurfelAssociations > depthAssociations;
	depthAssociations.reserve( nodes.size() );

	// only process nodes that are active (should improve parallel processing)
	std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* > activeNodes;
	activeNodes.reserve( nodes.size() );

	for( unsigned int i = 0; i < nodes.size(); i++ ) {

		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = nodes[i];

		if( n->value_.associated_ == -1 )
			continue;

		activeNodes.push_back( n );

	}



	SoftAssociateFunctor af( &depthAssociations, params_, &source, &target, &activeNodes, transform, processDepth, searchDistFactor, maxSearchDist );

	if( params_.parallel_ ) {
		tbb::parallel_for_each( activeNodes.begin(), activeNodes.end(), af );
	}
	else {
		std::for_each( activeNodes.begin(), activeNodes.end(), af );
	}


	surfelAssociations.insert( surfelAssociations.end(), depthAssociations.begin(), depthAssociations.end() );

}


class SoftGradientFunctorLM {
public:


	SoftGradientFunctorLM( MultiResolutionSoftSurfelRegistration::SurfelAssociationsList* assocList, const MultiResolutionSoftSurfelRegistration::Params& params, double tx, double ty, double tz, double qx, double qy, double qz, double qw, bool derivs ) {

		derivs_ = derivs;

		assocList_ = assocList;

		params_ = params;

		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = tx;
		currentTransform(1,3) = ty;
		currentTransform(2,3) = tz;

		currentRotation = Eigen::Matrix3d( currentTransform.block<3,3>(0,0) );
		currentRotationT = currentRotation.transpose();
		currentTranslation = Eigen::Vector3d( currentTransform.block<3,1>(0,3) );

		if( derivs ) {

			// build up derivatives of rotation and translation for the transformation variables
			dt_tx(0) = 1.f; dt_tx(1) = 0.f; dt_tx(2) = 0.f;
			dt_ty(0) = 0.f; dt_ty(1) = 1.f; dt_ty(2) = 0.f;
			dt_tz(0) = 0.f; dt_tz(1) = 0.f; dt_tz(2) = 1.f;

			dR_qx.setZero();
			dR_qx(1,2) = -2;
			dR_qx(2,1) = 2;

			dR_qy.setZero();
			dR_qy(0,2) = 2;
			dR_qy(2,0) = -2;

			dR_qz.setZero();
			dR_qz(0,1) = -2;
			dR_qz(1,0) = 2;

		}

	}

	~SoftGradientFunctorLM() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*assocList_)[i]);
	}



	void operator()( MultiResolutionSoftSurfelRegistration::SurfelAssociations& assocs ) const {


		if( !assocs.src_->applyUpdate_ ) {
			for( unsigned int i = 0; i < assocs.associations_.size(); i++ )
				assocs.associations_[i].match = 0;
			return;
		}

		const float processResolution = assocs.n_src_->resolution();

		Eigen::Matrix3d cov_ss_add;
		cov_ss_add.setZero();
		if( params_.add_smooth_pos_covariance_ ) {
			cov_ss_add.setIdentity();
			cov_ss_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
		}

		const Eigen::Matrix3d cov_scene = currentRotation * assocs.src_->cov_.block<3,3>(0,0) * currentRotationT + cov_ss_add;
		const Eigen::Vector3d srcMean = assocs.src_->mean_.block<3,1>(0,0);

		for( unsigned int i = 0; i < assocs.associations_.size(); i++ ) {

			MultiResolutionSoftSurfelRegistration::SurfelAssociations::Surfel2SurfelAssociation& assoc = assocs.associations_[i];

			if( assoc.match == 0 || !assoc.dst_->applyUpdate_ ) {
				assoc.match = 0;
				assoc.weight = 0;
				continue;
			}


			const Eigen::Matrix3d cov1_ss = assoc.dst_->cov_.block<3,3>(0,0) + cov_ss_add;
			const Eigen::Vector3d dstMean = assoc.dst_->mean_.block<3,1>(0,0);

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = srcMean;
			pos(3,0) = 1.f;

			const Eigen::Vector4d pos_src = currentTransform * pos;

			const Eigen::Vector3d p_s = pos_src.block<3,1>(0,0);
			const Eigen::Vector3d diff_s = dstMean - p_s;

			const Eigen::Matrix3d cov_ss = cov1_ss + cov_scene;
			const Eigen::Matrix3d invcov_ss = cov_ss.inverse();

			const Eigen::Matrix3d cov_ss_a = cov1_ss + cov_scene + assocs.sigma2 * Eigen::Matrix3d::Identity();
			const Eigen::Matrix3d invcov_ss_a = cov_ss_a.inverse();

			assoc.error = diff_s.dot(invcov_ss * diff_s);

			assoc.z = dstMean;
			assoc.h = p_s;

			// compute weight
			assoc.weight = assoc.dst_->num_points_ * 1.0 / sqrt( 8.0 * M_PI*M_PI*M_PI * cov_ss_a.determinant()) * exp( -0.5 * diff_s.dot(invcov_ss_a * diff_s) );

			if( derivs_ ) {

				assoc.dh_dx.block<3,1>(0,0) = dt_tx;
				assoc.dh_dx.block<3,1>(0,1) = dt_ty;
				assoc.dh_dx.block<3,1>(0,2) = dt_tz;
				assoc.dh_dx.block<3,1>(0,3) = dR_qx * pos_src.block<3,1>(0,0);
				assoc.dh_dx.block<3,1>(0,4) = dR_qy * pos_src.block<3,1>(0,0);
				assoc.dh_dx.block<3,1>(0,5) = dR_qz * pos_src.block<3,1>(0,0);

				assoc.W = invcov_ss;

			}


			assoc.match = 1;

		}


		// normalize weights
		double sumWeight = params_.prior_prob_ / (1.0-params_.prior_prob_) * ((double)params_.model_num_points_ / (double)params_.scene_num_points_) * 1.0 / sqrt( 8.0 * M_PI*M_PI*M_PI * (cov_scene + assocs.sigma2 * Eigen::Matrix3d::Identity()).determinant() );

		for( unsigned int i = 0; i < assocs.associations_.size(); i++ ) {
			if( assocs.associations_[i].match == 0 ) {
				continue;
			}
			sumWeight += assocs.associations_[i].weight;
		}

		if( sumWeight > 0.0 ) {
			double invSumWeight = 1.0 / sumWeight;
			for( unsigned int i = 0; i < assocs.associations_.size(); i++ ) {
				MultiResolutionSoftSurfelRegistration::SurfelAssociations::Surfel2SurfelAssociation& assoc = assocs.associations_[i];
				if( assoc.match == 0 ) {
					continue;
				}
				assoc.weight *= assocs.src_->num_points_ * invSumWeight;

			}
		}

	}



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


	MultiResolutionSoftSurfelRegistration::SurfelAssociationsList* assocList_;

	MultiResolutionSoftSurfelRegistration::Params params_;

	bool derivs_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

bool MultiResolutionSoftSurfelRegistration::registrationErrorFunctionLM( const Eigen::Matrix< double, 6, 1 >& x, double wsign, double& f, MultiResolutionSoftSurfelRegistration::SurfelAssociationsList& surfelAssociations ) {

	double sumSurfelError	= 0.0;
	double sumSurfelWeight	= 0.0;

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = wsign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	SoftGradientFunctorLM gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, false );

	static tbb::affinity_partitioner ap;

	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	int cidx = 0;
	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(surfelAssociations.size());
		correspondences_target_points_->points.resize(surfelAssociations.size());
	}


	double numMatches = 0;
	for( MultiResolutionSoftSurfelRegistration::SurfelAssociationsList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		for( unsigned int i = 0; i < it->associations_.size(); i++ ) {

			MultiResolutionSoftSurfelRegistration::SurfelAssociations::Surfel2SurfelAssociation& assoc = it->associations_[i];

			if( !assoc.match )
				continue;


			float nweight = 1.0;//it->n_src_->value_.assocWeight_ * assoc.n_dst_->value_.assocWeight_;
			float weight = nweight * assoc.weight;

			sumSurfelError += weight * assoc.error;
			sumSurfelWeight += weight;
			numMatches += 1.0;//nweight;



			if( correspondences_source_points_ ) {

				pcl::PointXYZRGB& p1 = correspondences_source_points_->points[cidx];
				pcl::PointXYZRGB& p2 = correspondences_target_points_->points[cidx];

				Eigen::Vector4f pos1 = assoc.n_dst_->getCenterPosition();
				Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

				p1.x = pos1(0);
				p1.y = pos1(1);
				p1.z = pos1(2);

				p1.r = nweight * 255.f;
				p1.g = 0;
				p1.b = (1.f-nweight) * 255.f;

				Eigen::Vector4d pos;
				pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
				pos(3,0) = 1.f;

				const Eigen::Vector4d pos_src = gf.currentTransform * pos;

				p2.x = pos_src[0];
				p2.y = pos_src[1];
				p2.z = pos_src[2];

				p2.r = nweight * 255.f;
				p2.g = 0;
				p2.b = (1.f-nweight) * 255.f;

				cidx++;
			}

		}

	}


	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(cidx);
		correspondences_target_points_->points.resize(cidx);
	}

	if( sumSurfelWeight <= 1e-10 ) {
		sumSurfelError = std::numeric_limits<double>::max();
		return false;
	}
	else if( numMatches < params_.registration_min_num_surfels_ ) {
		sumSurfelError = std::numeric_limits<double>::max();
		std::cout << "not enough surfels for robust matching " << numMatches << "\n";
		return false;
	}



	f = sumSurfelError / sumSurfelWeight * numMatches;


	if( params_.use_prior_pose_ ) {
		f += (params_.prior_pose_mean_ - x).transpose() * params_.prior_pose_invcov_ * (params_.prior_pose_mean_ - x);
	}


	return true;

}



bool MultiResolutionSoftSurfelRegistration::registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( const Eigen::Matrix< double, 6, 1 >& x, double wsign, double& f, Eigen::Matrix< double, 6, 1 >& df, Eigen::Matrix< double, 6, 6 >& d2f, MultiResolutionSoftSurfelRegistration::SurfelAssociationsList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	df.setZero();
	d2f.setZero();

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = wsign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	SoftGradientFunctorLM gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, true );

	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	int cidx = 0;
	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(surfelAssociations.size());
		correspondences_target_points_->points.resize(surfelAssociations.size());
	}


	double numMatches = 0;
	for( MultiResolutionSoftSurfelRegistration::SurfelAssociationsList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		for( unsigned int i = 0; i < it->associations_.size(); i++ ) {

			MultiResolutionSoftSurfelRegistration::SurfelAssociations::Surfel2SurfelAssociation& assoc = it->associations_[i];

			if( !assoc.match )
				continue;


			float nweight = 1.0;//it->n_src_->value_.assocWeight_ * assoc.n_dst_->value_.assocWeight_;
			float weight = nweight * assoc.weight;


			const Eigen::Matrix< double, 6, 3 > JtW = weight * assoc.dh_dx.transpose() * assoc.W;

			df += JtW * (assoc.z - assoc.h);
			d2f += JtW * assoc.dh_dx;

			sumError += weight * assoc.error;
			sumWeight += weight;
			numMatches += 1.0;//nweight;



			if( correspondences_source_points_ ) {

				pcl::PointXYZRGB& p1 = correspondences_source_points_->points[cidx];
				pcl::PointXYZRGB& p2 = correspondences_target_points_->points[cidx];

				Eigen::Vector4f pos1 = assoc.n_dst_->getCenterPosition();
				Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

				p1.x = pos1(0);
				p1.y = pos1(1);
				p1.z = pos1(2);

				p1.r = nweight * 255.f;
				p1.g = 0;
				p1.b = (1.f-nweight) * 255.f;

				Eigen::Vector4d pos;
				pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
				pos(3,0) = 1.f;

				const Eigen::Vector4d pos_src = gf.currentTransform * pos;

				p2.x = pos_src[0];
				p2.y = pos_src[1];
				p2.z = pos_src[2];

				p2.r = nweight * 255.f;
				p2.g = 0;
				p2.b = (1.f-nweight) * 255.f;

				cidx++;
			}

		}

	}


	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(cidx);
		correspondences_target_points_->points.resize(cidx);
	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else if( numMatches < params_.registration_min_num_surfels_ ) {
		sumError = std::numeric_limits<double>::max();
		std::cout << "not enough surfels for robust matching " << numMatches << "\n";
		return false;
	}

	f = sumError / sumWeight * numMatches;
	df = df / sumWeight * numMatches;
	d2f = d2f / sumWeight * numMatches;



	if( params_.use_prior_pose_ ) {

		f += (x - params_.prior_pose_mean_).transpose() * params_.prior_pose_invcov_ * (x - params_.prior_pose_mean_);
		df += params_.prior_pose_invcov_ * (params_.prior_pose_mean_ - x);
		d2f += params_.prior_pose_invcov_;

	}


	return true;

}



bool MultiResolutionSoftSurfelRegistration::estimateTransformationLevenbergMarquardt( Eigen::Matrix4d& transform, int maxIterations ) {

	const double tau = 10e-5;
	const double min_delta = 1e-3; // was 1e-3

	float minResolution = std::min( params_.startResolution_, params_.stopResolution_ );
	float maxResolution = std::max( params_.startResolution_, params_.stopResolution_ );

	Eigen::Matrix4d currentTransform = transform;


	target_->buildSamplingMap();

	// initialize with current transform
	Eigen::Matrix< double, 6, 1 > x;
	Eigen::Quaterniond q( currentTransform.block<3,3>(0,0) );


	x(0) = currentTransform( 0, 3 );
	x(1) = currentTransform( 1, 3 );
	x(2) = currentTransform( 2, 3 );
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	lastWSign_ = q.w() / fabsf(q.w());


	pcl::StopWatch stopwatch;

	Eigen::Matrix< double, 6, 1 > df;
	Eigen::Matrix< double, 6, 6 > d2f;

	const Eigen::Matrix< double, 6, 6 > id6 = Eigen::Matrix< double, 6, 6 >::Identity();
	double mu = -1.0;
	double nu = 2;

	double last_error = std::numeric_limits<double>::max();

	MultiResolutionSoftSurfelRegistration::SurfelAssociationsList surfelAssociations;

	bool reassociate = true;

	bool reevaluateGradient = true;

	bool retVal = true;

	int iter = 0;
	while( iter < maxIterations ) {

		if( processTimeWatch.getTime() > params_.max_processing_time_ )
			return false;

		if( reevaluateGradient ) {

//			if( reassociate ) {
				target_->clearAssociations();
//			}

			float searchDistFactor = 2.f;
			float maxSearchDist = 2.f*maxResolution;

			stopwatch.reset();
			surfelAssociations.clear();
			associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, target_->samplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist );
			double deltat = stopwatch.getTime();

//			std::cout << "#assocs: " << surfelAssociations.size() << std::endl;

//			std::cout << "assoc took: " << deltat << "\n";

			stopwatch.reset();
			retVal = registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( x, lastWSign_, last_error, df, d2f, surfelAssociations );
			double deltat2 = stopwatch.getTime();
//			std::cout << "reg deriv took: " << deltat2 << "\n";

		}

		reevaluateGradient = false;

		if( !retVal ) {
			std::cout << "registration failed\n";
			return false;
		}


		if( mu < 0 ) {
			mu = tau * std::max( d2f.maxCoeff(), -d2f.minCoeff() );
		}

		Eigen::Matrix< double, 6, 1 > delta_x = Eigen::Matrix< double, 6, 1 >::Zero();
		Eigen::Matrix< double, 6, 6 > d2f_inv = Eigen::Matrix< double, 6, 6 >::Zero();
		if( fabsf( d2f.determinant() ) > std::numeric_limits<double>::epsilon() ) {

			d2f_inv = (d2f + mu * id6).inverse();

			delta_x = d2f_inv * df;

		}

		if( delta_x.norm() < min_delta ) {

			if( reassociate ) {
				std::cout << "reassociate break\n";
				break;
			}

			reassociate = true;
			reevaluateGradient = true;
		}
		else
			reassociate = false;


		double qx = x( 3 );
		double qy = x( 4 );
		double qz = x( 5 );
		double qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);


		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = x( 0 );
		currentTransform(1,3) = x( 1 );
		currentTransform(2,3) = x( 2 );


		qx = delta_x( 3 );
		qy = delta_x( 4 );
		qz = delta_x( 5 );
		qw = sqrt(1.0-qx*qx-qy*qy-qz*qz);

		Eigen::Matrix4d deltaTransform = Eigen::Matrix4d::Identity();
		deltaTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		deltaTransform(0,3) = delta_x( 0 );
		deltaTransform(1,3) = delta_x( 1 );
		deltaTransform(2,3) = delta_x( 2 );

		Eigen::Matrix4d newTransform = deltaTransform * currentTransform;

		Eigen::Matrix< double, 6, 1 > x_new;
		x_new( 0 ) = newTransform(0,3);
		x_new( 1 ) = newTransform(1,3);
		x_new( 2 ) = newTransform(2,3);

		Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
		x_new( 3 ) = q_new.x();
		x_new( 4 ) = q_new.y();
		x_new( 5 ) = q_new.z();
		double newWSign = q_new.w() / fabsf(q_new.w());

		double new_error = 0.0;
		bool retVal2 = registrationErrorFunctionLM( x_new, newWSign, new_error, surfelAssociations );

		double rho = (last_error - new_error) / (delta_x.transpose() * (mu * delta_x + df));

		if( !retVal2 ) {

			rho = -1;

//			return false;
		}



		if( rho > 0 ) {

			x = x_new;
			lastWSign_ = newWSign;

			mu *= std::max( 0.333, 1.0 - pow( 2.0*rho-1.0, 3.0 ) );
			nu = 2;

			reevaluateGradient = true;

		}
		else {

			mu *= nu; nu *= 2.0;

		}



		qx = x( 3 );
		qy = x( 4 );
		qz = x( 5 );
		qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);



		if( boost::math::isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f ) {
			return false;
		}


		transform.setIdentity();
		transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		transform(0,3) = x( 0 );
		transform(1,3) = x( 1 );
		transform(2,3) = x( 2 );


		iter++;

	}


	return retVal;

}



bool MultiResolutionSoftSurfelRegistration::estimateTransformation( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesTargetPoints, int iterations ) {

	processTimeWatch.reset();

	params_.startResolution_ = startResolution;
	params_.stopResolution_ = stopResolution;

	source_ = &source;
	target_ = &target;


	correspondences_source_points_ = correspondencesSourcePoints;
	correspondences_target_points_ = correspondencesTargetPoints;

	// estimate transformation from maps
	target.clearAssociations();


	bool retVal = estimateTransformationLevenbergMarquardt( transform, iterations );

	if( !retVal )
		std::cout << "levenberg marquardt failed\n";

	return retVal;

}



bool MultiResolutionSoftSurfelRegistration::estimatePoseCovarianceLM( Eigen::Matrix< double, 6, 6 >& poseCov, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, SurfelAssociationsList* surfelAssociationsArg, bool knownAssociations ) {

	MultiResolutionSoftSurfelRegistration::SurfelAssociationsList surfelAssociations;


	if( !knownAssociations ) {
		target.clearAssociations();

		float minResolution = std::min( startResolution, stopResolution );
		float maxResolution = std::max( startResolution, stopResolution );

		target.buildSamplingMap();

		associateMapsBreadthFirstParallel( surfelAssociations, source, target, target.samplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, 2.f, 2.f*maxResolution );

	}
	else if( surfelAssociationsArg )
		surfelAssociations = *surfelAssociationsArg;


	double sumWeight = 0.0;

	Eigen::Quaterniond q( transform.block<3,3>(0,0) );

	const double tx = transform(0,3);
	const double ty = transform(1,3);
	const double tz = transform(2,3);
	const double qx = q.x();
	const double qy = q.y();
	const double qz = q.z();
	const double qw = q.w();

	MultiResolutionSoftSurfelRegistration::Params params = params_;
	SoftGradientFunctorLM gf( &surfelAssociations, params, tx, ty, tz, qx, qy, qz, qw, true );

	static tbb::affinity_partitioner ap;

	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	Eigen::Matrix< double, 6, 6 > d2f, JSzJ;
	d2f.setZero();
	JSzJ.setZero();

	for( MultiResolutionSoftSurfelRegistration::SurfelAssociationsList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		for( unsigned int i = 0; i < it->associations_.size(); i++ ) {

			MultiResolutionSoftSurfelRegistration::SurfelAssociations::Surfel2SurfelAssociation& assoc = it->associations_[i];

			if( !assoc.match )
				continue;

			d2f += assoc.dh_dx.transpose() * assoc.W * assoc.dh_dx;
			sumWeight += 1.0;

		}

	}


	if( sumWeight <= 1e-10 ) {
		poseCov.setIdentity();
		return false;
	}
	else if( sumWeight < params_.registration_min_num_surfels_ ) {
		std::cout << "not enough surfels for robust matching\n";
		poseCov.setIdentity();
		return false;
	}


	poseCov.setZero();

	if( fabsf(d2f.determinant()) < 1e-8 ) {
		poseCov.setIdentity();
		return false;
	}

	poseCov = d2f.inverse();

	if( params_.use_prior_pose_ ) {
		poseCov = (poseCov.inverse().eval() + params_.prior_pose_invcov_).inverse().eval();
	}

	return true;


}



