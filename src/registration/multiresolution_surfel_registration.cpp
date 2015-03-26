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

#include "mrsmap/registration/multiresolution_surfel_registration.h"

#include <mrsmap/utilities/utilities.h>

#include <g2o/types/slam3d/dquat2mat.h>

#include <deque>

#include <fstream>

#include <tbb/tbb.h>

#include <cmath>

#include <mrsmap/utilities/logging.h>

using namespace mrsmap;

typedef MultiResolutionSurfelRegistration MRCSReg;
typedef MRCSReg::SurfelAssociationList MRCSRSAL;
typedef MRCSReg::FeatureAssociationList MRCSRFAL;
typedef MultiResolutionSurfelMap MRCSMap;


inline MultiResolutionSurfelMap::Surfel* bestSurfelMatch( spatialaggregate::OcTreeNode<float, MultiResolutionSurfelMap::NodeValue>* node,
                                Eigen::Vector4d & viewDir, unsigned int & surfelIndex  ) {
    const double & x = viewDir(0);
    const double & y = viewDir(1);
    const double & z = viewDir[2];
    const double x_abs = fabsf( x );
    const double y_abs = fabsf( y );
    const double z_abs = fabsf( z );
    surfelIndex = 0;

    if ( x_abs > y_abs && x_abs > z_abs ) {
        surfelIndex = x >= 0 ? 0 : 1;
    } else if ( y_abs > z_abs ) {
        surfelIndex = y >= 0 ? 2 : 3;
    } else {
        surfelIndex = z >= 0 ? 4 : 5;
    }

    return &( node->value_.surfels_[surfelIndex] );
}



MultiResolutionSurfelRegistration::Params::Params() {
	init();
}

void MultiResolutionSurfelRegistration::Params::init() {

	// defaults
	registerSurfels_ = true;
	registerFeatures_ = false;

	max_processing_time_ = std::numeric_limits<double>::max();


	use_prior_pose_ = false;
	prior_pose_mean_ = Eigen::Matrix< double, 6, 1 >::Zero();
	prior_pose_invcov_ = Eigen::Matrix< double, 6, 6 >::Identity();

	add_smooth_pos_covariance_ = true;
	smooth_surface_cov_factor_ = 0.001f;

	interpolation_cov_factor_ = 20.0;

	surfel_match_angle_threshold_ = 0.5;
	registration_min_num_surfels_ = 0;
	max_feature_dist2_ = 0.1;
	use_features_ = true;

	match_likelihood_use_color_ = true;
	luminance_damp_diff_ = 0.5;
	color_damp_diff_ = 0.1;
	model_visibility_max_depth_ = 12;

	registration_use_color_ = true;
	luminance_reg_threshold_ = 0.5;
	color_reg_threshold_ = 0.1;

	occlusion_z_similarity_factor_ = 0.02f;
	image_border_range_ = 0;

	parallel_ = true;

	recover_associations_ = true;

	startResolution_ = 0.0125f;
	stopResolution_ = 0.2f;


	pointFeatureMatchingNumNeighbors_= 3;
	pointFeatureMatchingThreshold_ = 40;
	pointFeatureMatchingCoarseImagePosMahalDist_ = 1000.0;
	pointFeatureMatchingFineImagePosMahalDist_ = 48.0;
	pointFeatureWeight_ = 0.05; // weighting relative to surfels
	pointFeatureMinNumMatches_ = 50;
	// matchings beyond this threshold are considered outliers
	debugFeatures_ = false;
	calibration_f_ = 525.f;
	calibration_c1_ = 319.5f;
	calibration_c2_ = 239.5f;
	K_.setIdentity();
	K_(0, 0) = K_(1, 1) = calibration_f_;
	K_(0, 2) = calibration_c1_;
	K_(1, 2) = calibration_c2_;
	KInv_ = K_.inverse();

}

std::string MultiResolutionSurfelRegistration::Params::toString() {

	std::stringstream retVal;

	retVal << "use_prior_pose: " << (use_prior_pose_ ? 1 : 0) << std::endl;
	retVal << "prior_pose_mean: " << prior_pose_mean_.transpose() << std::endl;
	retVal << "prior_pose_invcov: " << prior_pose_invcov_ << std::endl;

	return retVal.str();

}



MultiResolutionSurfelRegistration::MultiResolutionSurfelRegistration() {

}


MultiResolutionSurfelRegistration::MultiResolutionSurfelRegistration( const Params& params ) {

	params_ = params;

}


void MultiResolutionSurfelRegistration::setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances ) {

	params_.use_prior_pose_ = enabled;
	params_.prior_pose_mean_ = prior_pose_mean;
	params_.prior_pose_invcov_ = Eigen::DiagonalMatrix< double, 6 >( prior_pose_variances ).inverse();

}


void MultiResolutionSurfelRegistration::setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 6 >& prior_pose_cov ) {

	params_.use_prior_pose_ = enabled;
	params_.prior_pose_mean_ = prior_pose_mean;
	params_.prior_pose_invcov_ = prior_pose_cov.inverse();

}



spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* MultiResolutionSurfelRegistration::getOccluder2( const Eigen::Vector4f& p, const MultiResolutionSurfelMap& target, double z_similarity_factor ) {

	if( boost::math::isnan( p(0) ) )
		return NULL;

	int px = 525.0 * p(0) / p(2) + 319.5;
	int py = 525.0 * p(1) / p(2) + 239.5;


	if( px < 0 || px >= 640 || py < 0 || py >= 480 ) {
		return NULL;
	}

	if( !target.imageAllocator_->node_set_.empty() ) {

		unsigned int idx = py * 640 + px;
		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = target.imageAllocator_->node_image_[idx];
		if( n ) {
			double z_dist = std::max( 0.f, p(2) - n->getCenterPosition()(2) );
			if( z_dist > fabsf( z_similarity_factor * p(2) ) )
				return n;

//			double z_dist = p(2) - n->getCenterPosition()(2);
//			if( fabsf( z_dist ) <= fabsf( 0.4 ) || std::max( 0.0, z_dist ) > fabsf( z_similarity_factor * p(2) ) )
//				return n;
		}

	}
	else {

		std::cout << "WARNING: mrsmap not created with node image! occlusion check disabled.\n";

	}


	return NULL;

}



bool pointOccluded( const Eigen::Vector4f& p, const MultiResolutionSurfelMap& target, double z_similarity_factor ) {

	if( boost::math::isnan( p(0) ) )
		return false;

	int px = 525.0 * p(0) / p(2) + 319.5;
	int py = 525.0 * p(1) / p(2) + 239.5;


	if( px < 0 || px >= 640 || py < 0 || py >= 480 ) {
		return false;
	}

	if( !target.imageAllocator_->node_set_.empty() ) {

		unsigned int idx = py * 640 + px;
		const spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = target.imageAllocator_->node_image_[idx];
		if( n ) {
			double z_dist = std::max( 0.f, p(2) - n->getCenterPosition()(2) );
			if( z_dist > fabsf( z_similarity_factor * p(2) ) )
				return true;
		}

	}
	else {

		std::cout << "WARNING: mrsmap not created with node image! occlusion check disabled.\n";

	}


	return false;

}


// the "reverse" of pointOccluded
bool pointSeenThrough( const Eigen::Vector4f& p, const MultiResolutionSurfelMap& target, double z_similarity_factor, bool markSeenThrough = false ) {

	if( boost::math::isnan( p(0) ) )
		return false;

	int px = 525.0 * p(0) / p(2) + 319.5;
	int py = 525.0 * p(1) / p(2) + 239.5;


	if( px < 0 || px >= 640 || py < 0 || py >= 480 ) {
		return false;
	}

	if( !target.imageAllocator_->node_set_.empty() ) {

		unsigned int idx = py * 640 + px;
		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = target.imageAllocator_->node_image_[idx];
		if( n ) {
			double z_dist = std::max( 0.f, n->getCenterPosition()(2) - p(2) );
			if( z_dist > fabsf( z_similarity_factor * p(2) ) ) {
				if( markSeenThrough ) {
					spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n2 = n;
					while( n2 != NULL ) {
//						for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ )
							n2->value_.surfels_[4].seenThrough_ = true;
						n2 = n2->parent_;
					}
				}
				return true;
			}
		}

	}


	return false;

}


spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* MultiResolutionSurfelRegistration::getOccluder( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, const MultiResolutionSurfelMap& target, const Eigen::Matrix4d& transform ) {

	// determine corresponding node in target..
	Eigen::Vector4f npos = node->getPosition();
	npos(3) = 1.0;
	Eigen::Vector4f npos_match_src = transform.cast<float>() * npos;

	return getOccluder2( npos_match_src, target, params_.occlusion_z_similarity_factor_ );

}


spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* MultiResolutionSurfelRegistration::calculateNegLogLikelihoodFeatureScoreN( double& logLikelihood, double& featureScore, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, const MultiResolutionSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate ) {

	// for each surfel in node with applyUpdate set and sufficient points, transform to target using transform,
	// then measure negative log likelihood

//	const double spatial_z_cov_factor = 0.04;
//	const double color_z_cov = 0.0001;
//	const double normalStd = 0.25*M_PI;

	featureScore = std::numeric_limits<double>::max();
	logLikelihood = std::numeric_limits<double>::max();

	Eigen::Matrix3d rotation = transform.block<3,3>(0,0);

	// determine corresponding node in target..
	Eigen::Vector4f npos = node->getPosition();
	npos(3) = 1.0;
	Eigen::Vector4f npos_match_src = transform.cast<float>() * npos;

	if( !pointInImage( npos_match_src, params_.image_border_range_ ) )
		outOfImage = true;

	// also check if point is occluded (project into image and compare z coordinate at some threshold)
	occluded = pointOccluded( npos_match_src, target, params_.occlusion_z_similarity_factor_ );
//	if( occluded )
//		return NULL;

	std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* > neighbors;
	neighbors.reserve(50);
	float searchRange = 2.f;
	Eigen::Vector4f minPosition = npos_match_src - Eigen::Vector4f( searchRange*node->resolution(), searchRange*node->resolution(), searchRange*node->resolution(), 0.f );
	Eigen::Vector4f maxPosition = npos_match_src + Eigen::Vector4f( searchRange*node->resolution(), searchRange*node->resolution(), searchRange*node->resolution(), 0.f );

	target.octree_->getAllNodesInVolumeOnDepth( neighbors, minPosition, maxPosition, node->depth_, true );

	if( neighbors.size() == 0 ) {
		return NULL;
	}

	spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_matched = NULL;
	MultiResolutionSurfelMap::Surfel* srcSurfel = NULL;
	MultiResolutionSurfelMap::Surfel* matchedSurfel = NULL;
	int matchedSurfelIdx = -1;
	double bestDist = std::numeric_limits<double>::max();

	// get closest node in neighbor list
	for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

		MultiResolutionSurfelMap::Surfel& surfel = node->value_.surfels_[i];

		// border points are returned but must be handled later!
		if( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
			continue;
		}
//		if( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ || !surfel.applyUpdate_ ) {
//			continue;
//		}

		Eigen::Vector4d pos;
		pos.block<3,1>(0,0) = surfel.mean_.block<3,1>(0,0);
		pos(3,0) = 1.f;

		Eigen::Vector4d pos_match_src = transform * pos;
		Eigen::Vector3d dir_match_src = rotation * surfel.initial_view_dir_;

		for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >::iterator it = neighbors.begin(); it != neighbors.end(); it++ ) {

			if( (*it)->value_.border_ != node->value_.border_ )
				continue;

			MultiResolutionSurfelMap::Surfel* bestMatchSurfel = NULL;
			int bestMatchSurfelIdx = -1;
			double bestMatchDist = -1.f;
			for( unsigned int k = 0; k < MultiResolutionSurfelMap::NodeValue::num_surfels_; k++ ) {

				const MultiResolutionSurfelMap::Surfel& srcSurfel2 = (*it)->value_.surfels_[k];

				if( srcSurfel2.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
					continue;
				}

				const double dist = dir_match_src.dot( srcSurfel2.initial_view_dir_ );
				if( dist >= params_.surfel_match_angle_threshold_ && dist >= bestMatchDist ) {
					bestMatchSurfel = &((*it)->value_.surfels_[k]);
					bestMatchDist = dist;
					bestMatchSurfelIdx = k;
				}
			}

			if( bestMatchSurfel ) {
				// use distance between means
				double dist = (pos_match_src.block<3,1>(0,0) - bestMatchSurfel->mean_.block<3,1>(0,0)).norm();
				if( dist < bestDist ) {
					bestDist = dist;
					srcSurfel = &surfel;
					n_matched = *it;
					matchedSurfel = bestMatchSurfel;
					matchedSurfelIdx = bestMatchSurfelIdx;
				}
			}
		}

	}

	// border points are returned but must be handled later!
//	if( !n_matched || !matchedSurfel->applyUpdate_ ) {
	if( !n_matched ) {
		return NULL;
	}

	if( !srcSurfel->applyUpdate_ || !matchedSurfel->applyUpdate_ )
		virtualBorder = true;

//	if( !matchedSurfel->applyUpdate_ )
//		virtualBorder = true;
//	if( !srcSurfel->applyUpdate_ )
//		return NULL;

	featureScore = 0;//srcSurfel->agglomerated_shape_texture_features_.distance( matchedSurfel->agglomerated_shape_texture_features_ );

	Eigen::Vector4d pos;
	pos.block<3,1>(0,0) = srcSurfel->mean_.block<3,1>(0,0);
	pos(3,0) = 1.f;

	Eigen::Vector4d pos_match_src = transform * pos;

	double l = 0;


	if( params_.match_likelihood_use_color_ ) {

//		Eigen::Matrix< double, 6, 6 > rotation6 = Eigen::Matrix< double, 6, 6 >::Identity();
//		rotation6.block<3,3>(0,0) = rotation;

		Eigen::Matrix< double, 6, 6 > cov1;
		Eigen::Matrix< double, 6, 1 > dstMean;

		bool in_interpolation_range = true;

		if( interpolate ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center

			// associate with neighbors for which distance to the node center is smaller than resolution

			dstMean.setZero();
			cov1.setZero();

			double sumWeight = 0.f;
			double sumWeight2 = 0.f;

			const float resolution = n_matched->resolution();

			for( int s = 0; s < 27; s++ ) {

				spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_dst_n = n_matched->neighbors_[s];

				if(!n_dst_n)
					continue;

				MultiResolutionSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[matchedSurfelIdx];
				if( dst_n->num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
					continue;

				Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_match_src.block<3,1>(0,0);
				const double dx = resolution - fabsf(centerDiff_n(0));
				const double dy = resolution - fabsf(centerDiff_n(1));
				const double dz = resolution - fabsf(centerDiff_n(2));

				if( dx > 0 && dy > 0 && dz > 0 ) {

					const double weight = dx*dy*dz;

					dstMean += weight * dst_n->mean_;
					cov1 += weight*weight * (dst_n->cov_);

					sumWeight += weight;
					sumWeight2 += weight*weight;

				}


			}

			// numerically stable?
			if( sumWeight > resolution* 1e-6 ) {
				dstMean /= sumWeight;
				cov1 /= sumWeight2;

			}
			else
				in_interpolation_range = false;

		}

		if( !interpolate || !in_interpolation_range ) {

			dstMean = matchedSurfel->mean_;
			cov1 = matchedSurfel->cov_;

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
//		cov1 *= params_.interpolation_cov_factor_;
//		const Eigen::Matrix< double, 6, 6 > cov2 = params_.interpolation_cov_factor_ * srcSurfel->cov_;

		const Eigen::Matrix< double, 6, 6 > cov2 = srcSurfel->cov_;

		Eigen::Matrix< double, 6, 1 > diff;
		diff.block<3,1>(0,0) = dstMean.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0);
		diff.block<3,1>(3,0) = dstMean.block<3,1>(3,0) - srcSurfel->mean_.block<3,1>(3,0);
		if( fabs(diff(3)) < params_.luminance_damp_diff_ )
			diff(3) = 0;
		if( fabs(diff(4)) < params_.color_damp_diff_ )
			diff(4) = 0;
		if( fabs(diff(5)) < params_.color_damp_diff_ )
			diff(5) = 0;

		if( diff(3) < 0 )
			diff(3) += params_.luminance_damp_diff_;
		if( diff(4) < 0 )
			diff(4) += params_.color_damp_diff_;
		if( diff(5) < 0 )
			diff(5) += params_.color_damp_diff_;

		if( diff(3) > 0 )
			diff(3) -= params_.luminance_damp_diff_;
		if( diff(4) > 0 )
			diff(4) -= params_.color_damp_diff_;
		if( diff(5) > 0 )
			diff(5) -= params_.color_damp_diff_;


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		const Eigen::Matrix3d cov1_ss = cov1.block<3,3>(0,0);
		const Eigen::Matrix3d cov2_ss = cov2.block<3,3>(0,0);

		const Eigen::Matrix3d Rcov2_ss = rotation * cov2_ss;

		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * rotation.transpose() + spatial_z_cov_factor*node->resolution()*node->resolution() * Eigen::Matrix3d::Identity();
//		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * rotation.transpose() + spatial_z_cov_factor*0.01 * Eigen::Matrix3d::Identity();
//		const Eigen::Matrix3d cov_ss = node->resolution()*node->resolution() * Eigen::Matrix3d::Identity();
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();

		const Eigen::Vector3d invcov_ss_diff_s = invcov_ss * diff.block<3,1>(0,0);

//		l = log( cov_ss.determinant() ) + diff.block<3,1>(0,0).dot(invcov_ss_diff_s);
		l = diff.block<3,1>(0,0).dot(invcov_ss_diff_s);
//		l = std::max( 0.0, l - 9.0 );
//		if( l < 9 )
//			l = 0;
		if( l > 48.0 )
			l = 48.0;

//		std::cout << "s:\n";
//		std::cout << diff.block<3,1>(0,0) << "\n";
//		std::cout << cov_ss.block<3,3>(0,0) << "\n";
//		std::cout << log( cov_ss.determinant() ) << "\n";
//		std::cout << l << "\n";

		const Eigen::Matrix3d cov_cc = cov1.block<3,3>(3,3) + cov2.block<3,3>(3,3) + color_z_cov * Eigen::Matrix3d::Identity();
//		l += log( cov_cc.determinant() ) + diff.block<3,1>(3,0).dot( cov_cc.inverse() * diff.block<3,1>(3,0) );
		double color_loglikelihood = diff.block<3,1>(3,0).dot( (cov_cc.inverse() * diff.block<3,1>(3,0)).eval() );
		if( color_loglikelihood > 48.0 )
			color_loglikelihood = 48.0;
		l += color_loglikelihood;


	}
	else {

		Eigen::Matrix3d cov1_ss;
		Eigen::Vector3d dstMean;

		bool in_interpolation_range = true;

		if( interpolate ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center
			const float resolution = node->resolution();

			// associate with neighbors for which distance to the node center is smaller than resolution

			dstMean.setZero();
			cov1_ss.setZero();

			double sumWeight = 0.f;
			double sumWeight2 = 0.f;

			for( int s = 0; s < 27; s++ ) {

				spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_dst_n = n_matched->neighbors_[s];

				if(!n_dst_n)
					continue;

				MultiResolutionSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[matchedSurfelIdx];
				if( dst_n->num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
					continue;

				Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_match_src.block<3,1>(0,0);
				const double dx = resolution - fabsf(centerDiff_n(0));
				const double dy = resolution - fabsf(centerDiff_n(1));
				const double dz = resolution - fabsf(centerDiff_n(2));

				if( dx > 0 && dy > 0 && dz > 0 ) {

					const double weight = dx*dy*dz;

					dstMean += weight * dst_n->mean_.block<3,1>(0,0);
					cov1_ss += weight*weight * (dst_n->cov_.block<3,3>(0,0));

					sumWeight += weight;
					sumWeight2 += weight*weight;

				}


			}

			// numerically stable?
			if( sumWeight > resolution* 1e-6 ) {
				dstMean /= sumWeight;
				cov1_ss /= sumWeight2;

			}
			else
				in_interpolation_range = false;

		}

		if( !interpolate || !in_interpolation_range ) {

			dstMean = matchedSurfel->mean_.block<3,1>(0,0);
			cov1_ss = matchedSurfel->cov_.block<3,3>(0,0);

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		cov1_ss *= params_.interpolation_cov_factor_;
		const Eigen::Matrix3d cov2_ss = params_.interpolation_cov_factor_ * srcSurfel->cov_.block<3,3>(0,0);

		const Eigen::Vector3d diff_s = dstMean - pos_match_src.block<3,1>(0,0);

		const Eigen::Matrix3d Rcov2_ss = rotation * cov2_ss;

		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * rotation.transpose() + spatial_z_cov_factor*node->resolution()*node->resolution() * Eigen::Matrix3d::Identity();
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();

		const Eigen::Vector3d invcov_ss_diff_s = invcov_ss * diff_s;

//		l = log( cov_ss.determinant() ) + diff_s.dot(invcov_ss_diff_s);
		l = diff_s.dot(invcov_ss_diff_s);
		if( l > 48.0 )
			l = 48.0;

	}


	// also consider normal orientation in the likelihood
//	// TODO curvature-dependency should be made nicer
//	if( srcSurfel->surface_curvature_ < 0.05 ) {
		Eigen::Vector4d normal_src;
		normal_src.block<3,1>(0,0) = srcSurfel->normal_;
		normal_src(3,0) = 0.0;
		normal_src = (transform * normal_src).eval();

		double normalError = acos( normal_src.block<3,1>(0,0).dot( matchedSurfel->normal_ ) );
		double normalExponent = std::min( 4.0, normalError * normalError / normal_z_cov );
	//	double normalLogLikelihood = log( 2.0 * M_PI * normalStd ) + normalExponent;

		l += normalExponent;
	//	l += normalLogLikelihood;

	//	std::cout << "n:\n";
	//	std::cout << normalError << "\n";
	//	std::cout << normalExponent << "\n\n";
//	}

	logLikelihood = std::min( l, logLikelihood );

	return n_matched;

}


spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* MultiResolutionSurfelRegistration::calculateNegLogLikelihoodN( double& logLikelihood, bool& outOfImage, bool& virtualBorder, bool& occluded, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, const MultiResolutionSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate ) {

	double featureScore = 0.0;

	return calculateNegLogLikelihoodFeatureScoreN( logLikelihood, featureScore, outOfImage, virtualBorder, occluded, node, target, transform, spatial_z_cov_factor, color_z_cov, normal_z_cov, interpolate );

}


bool MultiResolutionSurfelRegistration::calculateNegLogLikelihood( double& logLikelihood, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, const MultiResolutionSurfelMap& target, const Eigen::Matrix4d& transform, double spatial_z_cov_factor, double color_z_cov, double normal_z_cov, bool interpolate ) {

	bool outOfImage = false;
	bool virtualBorder = false;
	bool occluded = false;
	if( calculateNegLogLikelihoodN( logLikelihood, outOfImage, virtualBorder, occluded, node, target, transform, spatial_z_cov_factor, color_z_cov, normal_z_cov, interpolate ) != NULL )
		return true;
	else
		return false;

}


// transform from src to tgt
double MultiResolutionSurfelRegistration::calculateInPlaneLogLikelihood( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_src, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_tgt, const Eigen::Matrix4d& transform, double normal_z_cov ) {

	double bestLogLikelihood = 18.0;
	for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

		MultiResolutionSurfelMap::Surfel& s_src = n_src->value_.surfels_[i];
		MultiResolutionSurfelMap::Surfel& s_tgt = n_tgt->value_.surfels_[i];

		if( s_src.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
			continue;
		}

		if( s_tgt.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
			continue;
		}

		// measure variance along normal direction of reference surfel
		const Eigen::Vector3d mean_src = s_src.mean_.block<3,1>(0,0);
		const Eigen::Vector3d mean_tgt = s_tgt.mean_.block<3,1>(0,0);
		const Eigen::Vector4d mean_src4( mean_src(0), mean_src(1), mean_src(2), 1.0 );
		const Eigen::Vector3d mean_src_transformed = (transform * mean_src4).block<3,1>(0,0);

		const Eigen::Matrix3d rot_src = transform.block<3,3>(0,0);
		const Eigen::Matrix3d cov_src_transformed = rot_src * (s_src.cov_.block<3,3>(0,0)) * rot_src.transpose();
		const Eigen::Matrix3d cov_tgt = s_tgt.cov_.block<3,3>(0,0);

		const Eigen::Vector3d n_tgt = s_tgt.normal_;

		double var_n_src = n_tgt.transpose() * cov_src_transformed * n_tgt;
		double var_n_tgt = n_tgt.transpose() * cov_tgt * n_tgt;
		double var_n = var_n_src + var_n_tgt;

		double diff_n = n_tgt.dot( mean_tgt - mean_src_transformed );

		double logLikelihood = diff_n*diff_n / var_n;


		// also consider normal orientation for the likelihood
		Eigen::Vector4d normal_src;
		normal_src.block<3,1>(0,0) = s_src.normal_;
		normal_src(3,0) = 0.0;
		normal_src = (transform * normal_src).eval();

		double normalError = acos( normal_src.block<3,1>(0,0).dot( s_tgt.normal_ ) );
		double normalExponent = std::min( 9.0, normalError * normalError / normal_z_cov );

		logLikelihood += normalExponent;


		bestLogLikelihood = std::min( bestLogLikelihood, logLikelihood );

	}

	return bestLogLikelihood;

}


void MultiResolutionSurfelRegistration::associateMapsBreadthFirstParallel( MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, algorithm::OcTreeSamplingVectorMap< float, MultiResolutionSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution, double searchDistFactor, double maxSearchDist, bool useFeatures ) {


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

	for( int d = maxDepth; d >= 0; d-- ) {

		const float processResolution = source.octree_->volumeSizeForDepth( d );

		if( processResolution < minResolution || processResolution > maxResolution ) {
			continue;
		}

		associateNodeListParallel( surfelAssociations, source, target, targetSamplingMap[d], d, transform, searchDistFactor, maxSearchDist, useFeatures );

	}


}


class AssociateFunctor {
public:
	AssociateFunctor( tbb::concurrent_vector< MultiResolutionSurfelRegistration::SurfelAssociation >* associations, const MultiResolutionSurfelRegistration::Params& params, MultiResolutionSurfelMap* source, MultiResolutionSurfelMap* target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >* nodes, const Eigen::Matrix4d& transform, int processDepth, double searchDistFactor, double maxSearchDist, bool useFeatures ) {
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

		use_features_ = useFeatures;

		num_vol_queries_ = 0;
		num_finds_ = 0;
		num_neighbors_ = 0;


	}

	~AssociateFunctor() {}

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

//			if( numAssociatedChildren > 0 )
//				n->value_.associated_ = 1;
//
//			if( numChildren > 0 && numChildren == numAssociatedChildren )
//				return;

			if( numChildren > 0 && numAssociatedChildren > 0 ) {
				n->value_.associated_ = 1;
				return;
			}
		}

//		if( !n->value_.associated_ )
//			return;


		// check if surfels exist and can be associated by view direction
		// use only one best association per node
		float bestAssocDist = std::numeric_limits<float>::max();
		float bestAssocFeatureDist = std::numeric_limits<float>::max();
		MultiResolutionSurfelRegistration::SurfelAssociation bestAssoc;

		bool hasSurfel = false;

		// TODO: collect features for view directions (surfels)
		// once a representative node is chosen, search for feature correspondences by sweeping up the tree up to a maximum search distance.
		// check compatibility using inverse depth parametrization

		// check if a surfels exist
		for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

			// if image border points fall into this node, we must check the children_
			if( !n->value_.surfels_[i].applyUpdate_ ) {
				continue;
			}

//			if( n->value_.surfels_[i].cleared_ ) {
//				continue;
//			}

			if( n->value_.surfels_[i].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
				continue;
			}

			hasSurfel = true;
		}

		if( hasSurfel ) {

			spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_src_last = NULL;
			std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* > neighbors;

			// association of this node exists from a previous iteration?
			char surfelSrcIdx = -1;
			char surfelDstIdx = -1;
			if( n->value_.association_ ) {
				n_src_last = n->value_.association_;
				surfelSrcIdx = n->value_.assocSurfelIdx_;
				surfelDstIdx = n->value_.assocSurfelDstIdx_;
				n_src_last->getNeighbors( neighbors );
			}

			// does association of parent exist from a previous iteration?
			if( !n_src_last ) {

				if( false && n->parent_ && n->parent_->value_.association_ ) {

					n_src_last = n->parent_->value_.association_;
					surfelSrcIdx = n->parent_->value_.assocSurfelIdx_;
					surfelDstIdx = n->parent_->value_.assocSurfelDstIdx_;

					Eigen::Vector4f npos = n->getCenterPosition();
					npos(3) = 1.f;
					Eigen::Vector4f npos_match_src = transformf_ * npos;

					n_src_last = n_src_last->findRepresentative( npos_match_src, process_depth_ );

					if( n_src_last )
						n_src_last->getNeighbors( neighbors );

				}
				else  {

					neighbors.reserve(50);

					Eigen::Vector4f npos = n->getCenterPosition();
					npos(3) = 1.f;
					Eigen::Vector4f npos_match_src = transformf_ * npos;

					// if direct look-up fails, perform a region query
					// in case there is nothing within the volume, the query will exit early

					Eigen::Vector4f minPosition = npos_match_src - search_dist_vec_;
					Eigen::Vector4f maxPosition = npos_match_src + search_dist_vec_;

					source_->octree_->getAllNodesInVolumeOnDepth( neighbors, minPosition, maxPosition, process_depth_, false );

				}

			}

			if( neighbors.size() == 0 ) {

				n->value_.association_ = NULL;
				n->value_.associated_ = 0;

				return;
			}


			if( surfelSrcIdx >= 0 && surfelDstIdx >= 0 ) {

				const MultiResolutionSurfelMap::Surfel& surfel = n->value_.surfels_[surfelSrcIdx];

				if( surfel.num_points_ >= MultiResolutionSurfelMap::Surfel::min_points_ ) {

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


						// find matching surfel for the view direction, but allow to use a slightly worse fit,
						// when it is the only one with sufficient points for matching
						MultiResolutionSurfelMap::Surfel& dstSurfel = n_src->value_.surfels_[surfelDstIdx];

						if( dstSurfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
							continue;

						const double dist = dir_match_src.dot( dstSurfel.initial_view_dir_ );

						MultiResolutionSurfelMap::Surfel* bestMatchSurfel = NULL;
						int bestMatchSurfelIdx = -1;
						double bestMatchDist = -1.f;

						if( dist >= params_.surfel_match_angle_threshold_ ) {
							bestMatchSurfel = &dstSurfel;
							bestMatchDist = dist;
							bestMatchSurfelIdx = surfelDstIdx;
						}

						if( !bestMatchSurfel ) {
							continue;
						}

						// calculate error metric for matching surfels
						double dist_pos2 = (bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0)).squaredNorm();

						if( dist_pos2 > search_dist2_ )
							continue;


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


						// check local descriptor in any case
						float featureDist = 0.0;
						if( use_features_ )
							featureDist = surfel.agglomerated_shape_texture_features_.distance( bestMatchSurfel->agglomerated_shape_texture_features_ );
						if( use_features_ && featureDist > params_.max_feature_dist2_ )
							continue;




						float assocDist = sqrtf(dist_pos2);// + process_resolution_*process_resolution_*(bestMatchSurfel->mean_.block<3,1>(3,0) - surfel.mean_.block<3,1>(3,0)).squaredNorm());

						if( use_features_ )
							assocDist *= featureDist;

						if( assocDist < bestAssocDist ) {
							bestAssocDist = assocDist;
							bestAssocFeatureDist = featureDist;
							n->value_.surfels_[surfelSrcIdx].assocDist_ = assocDist;

//								bestAssoc = MultiResolutionSurfelRegistration::SurfelAssociation( n, &n->value_.surfels_[surfelSrcIdx], surfelSrcIdx, n_src, bestMatchSurfel, bestMatchSurfelIdx );
							bestAssoc.n_src_ = n;
							bestAssoc.src_ = &n->value_.surfels_[surfelSrcIdx];
							bestAssoc.src_idx_ = surfelSrcIdx;
							bestAssoc.n_dst_ = n_src;
							bestAssoc.dst_ = bestMatchSurfel;
							bestAssoc.dst_idx_ = bestMatchSurfelIdx;
							bestAssoc.match = 1;

							if( use_features_ )
								bestAssoc.weight = params_.max_feature_dist2_ - featureDist;
							else
								bestAssoc.weight = (1+numChildren) * 1.f;

						}

					}

				}

			}
			else {


				for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

					const MultiResolutionSurfelMap::Surfel& surfel = n->value_.surfels_[i];

					if( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
						continue;
					}

//					if( surfel.cleared_ ) {
//						continue;
//					}


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

						// find matching surfel for the view direction, but allow to use a slightly worse fit,
						// when it is the only one with sufficient points for matching
						MultiResolutionSurfelMap::Surfel* bestMatchSurfel = NULL;
						int bestMatchSurfelIdx = -1;
						double bestMatchDist = -1.f;
						for( unsigned int k = 0; k < MultiResolutionSurfelMap::NodeValue::num_surfels_; k++ ) {

							const MultiResolutionSurfelMap::Surfel& srcSurfel = n_src->value_.surfels_[k];

							if( srcSurfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
								continue;

							const double dist = dir_match_src.dot( srcSurfel.initial_view_dir_ );
							if( dist >= params_.surfel_match_angle_threshold_ && dist >= bestMatchDist ) {
								bestMatchSurfel = &n_src->value_.surfels_[k];
								bestMatchDist = dist;
								bestMatchSurfelIdx = k;
							}
						}

						if( !bestMatchSurfel ) {
							continue;
						}

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

						float assocDist = sqrtf(dist_pos2);

						if( use_features_ )
							assocDist *= featureDist;

						if( assocDist < bestAssocDist ) {
							bestAssocDist = assocDist;
							bestAssocFeatureDist = featureDist;
							n->value_.surfels_[i].assocDist_ = assocDist;

							bestAssoc.n_src_ = n;
							bestAssoc.src_ = &n->value_.surfels_[i];
							bestAssoc.src_idx_ = i;
							bestAssoc.n_dst_ = n_src;
							bestAssoc.dst_ = bestMatchSurfel;
							bestAssoc.dst_idx_ = bestMatchSurfelIdx;
							bestAssoc.match = 1;

							if( use_features_ )
								bestAssoc.weight = params_.max_feature_dist2_ - featureDist;
							else
								bestAssoc.weight = (1+numChildren) * 1.f;
						}

					}

				}

			}

		}

		if( bestAssocDist != std::numeric_limits<float>::max() ) {


//			bestAssoc.weight *= n->value_.assocWeight_;
//			bestAssoc.weight = 1.f;

			associations_->push_back( bestAssoc );
			n->value_.association_ = bestAssoc.n_dst_;
			n->value_.associated_ = 1;
			n->value_.assocSurfelIdx_ = bestAssoc.src_idx_;
			n->value_.assocSurfelDstIdx_ = bestAssoc.dst_idx_;
		}
		else {
			n->value_.association_ = NULL;
			n->value_.associated_ = 0;
		}


	}


	tbb::concurrent_vector< MultiResolutionSurfelRegistration::SurfelAssociation >* associations_;
	MultiResolutionSurfelRegistration::Params params_;
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


void MultiResolutionSurfelRegistration::associateNodeListParallel( MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >& nodes, int processDepth, Eigen::Matrix4d& transform, double searchDistFactor, double maxSearchDist, bool useFeatures ) {

	tbb::concurrent_vector< MultiResolutionSurfelRegistration::SurfelAssociation > depthAssociations;
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



	AssociateFunctor af( &depthAssociations, params_, &source, &target, &activeNodes, transform, processDepth, searchDistFactor, maxSearchDist, useFeatures );

	if( params_.parallel_ )
		tbb::parallel_for_each( activeNodes.begin(), activeNodes.end(), af );
	else
		std::for_each( activeNodes.begin(), activeNodes.end(), af );


	surfelAssociations.insert( surfelAssociations.end(), depthAssociations.begin(), depthAssociations.end() );

}


MultiResolutionSurfelRegistration::SurfelAssociationList MultiResolutionSurfelRegistration::revertSurfelAssociations( const MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	MultiResolutionSurfelRegistration::SurfelAssociationList retVal = surfelAssociations;

	for( unsigned int i = 0; i < retVal.size(); i++ ) {

		retVal[i].revert();

	}

	return retVal;

}



void MultiResolutionSurfelRegistration::associatePointFeatures() {

	pcl::StopWatch sw;
	sw.reset();

	const int numNeighbors = params_.pointFeatureMatchingNumNeighbors_;
	const int distanceThreshold = params_.pointFeatureMatchingThreshold_;

	featureAssociations_.clear();

	if (!target_->lsh_index_ || !source_->lsh_index_)
		return;

	featureAssociations_.reserve(std::min(source_->features_.size(), target_->features_.size()));

	// find associations from source to target
	// build up query matrix
	flann::Matrix<unsigned char> sourceQuery(source_->descriptors_.data, source_->descriptors_.rows, source_->descriptors_.cols);
	flann::Matrix<unsigned char> targetQuery(target_->descriptors_.data, target_->descriptors_.rows, target_->descriptors_.cols);

	// indices in source features for target features
	flann::Matrix<int> sourceIndices(new int[targetQuery.rows * numNeighbors], targetQuery.rows, numNeighbors);
	flann::Matrix<int> sourceDists(new int[targetQuery.rows * numNeighbors], targetQuery.rows, numNeighbors);

	// indices in target features for source features
	flann::Matrix<int> targetIndices(new int[sourceQuery.rows * numNeighbors], sourceQuery.rows, numNeighbors);
	flann::Matrix<int> targetDists(new int[sourceQuery.rows * numNeighbors], sourceQuery.rows, numNeighbors);

	target_->lsh_index_->knnSearch(sourceQuery, targetIndices, targetDists, numNeighbors, flann::SearchParams());
	source_->lsh_index_->knnSearch(targetQuery, sourceIndices, sourceDists, numNeighbors, flann::SearchParams());

	if( params_.debugFeatures_ )
		std::cout << "flann query took: " << sw.getTime() << "\n";
	sw.reset();

	// find mutually consistent matches within distance threshold
	for (unsigned int i = 0; i < sourceQuery.rows; i++) {

		// check if source feature is among nearest neighbors of matched target feature
		for (unsigned int n = 0; n < numNeighbors; n++) {

			if (targetDists.ptr()[i * numNeighbors + n] > distanceThreshold)
				continue;

			int targetIdx = targetIndices.ptr()[i * numNeighbors + n];

			if (targetIdx < 0 || targetIdx >= sourceIndices.rows)
				continue;

			for (unsigned int n2 = 0; n2 < numNeighbors; n2++) {

				if (sourceDists.ptr()[targetIdx * numNeighbors + n2] > distanceThreshold)
					continue;

				int sourceIdx = sourceIndices.ptr()[targetIdx * numNeighbors + n2];

				if (sourceIdx < 0 || sourceIdx >= targetIndices.rows)
					continue;

				if (sourceIdx == i) {
					MultiResolutionSurfelRegistration::FeatureAssociation assoc( i, targetIdx);
					featureAssociations_.push_back(assoc);
//		    			consistentMatches.push_back( std::pair< int, int >( i, targetIdx ) );
					break;
				}

			}

		}

	}

	if( params_.debugFeatures_ )
		std::cout << "consistent match search took: " << sw.getTime() << "\n";


	delete[] sourceIndices.ptr();
	delete[] targetIndices.ptr();
	delete[] sourceDists.ptr();
	delete[] targetDists.ptr();



    if( params_.debugFeatures_ ) {
		cv::Mat outimg;

		std::vector< cv::KeyPoint > keypoints_src;
		for( unsigned int i = 0; i < source_->features_.size(); i++ ) {
			cv::KeyPoint kp;
			kp.pt.x = source_->features_[i].invzpos_(0);
			kp.pt.y = source_->features_[i].invzpos_(1);
			keypoints_src.push_back( kp );
		}

		std::vector< cv::KeyPoint > keypoints_tgt;
		for( unsigned int i = 0; i < target_->features_.size(); i++ ) {
			cv::KeyPoint kp;
			kp.pt.x = target_->features_[i].invzpos_(0);
			kp.pt.y = target_->features_[i].invzpos_(1);
			keypoints_tgt.push_back( kp );
		}


		std::vector< cv::DMatch > matches;
		for( unsigned int i = 0; i < featureAssociations_.size(); i++ ) {
			cv::DMatch match;
			match.trainIdx = featureAssociations_[i].dst_idx_;
			match.queryIdx = featureAssociations_[i].src_idx_;
			matches.push_back( match );
		}

		cv::drawMatches( source_->img_rgb_, keypoints_src, target_->img_rgb_, keypoints_tgt, matches, outimg );
		cv::imshow( "matches", outimg );
		cv::waitKey(1);
    }

}


class GradientFunctor {
public:
	GradientFunctor( MultiResolutionSurfelRegistration::SurfelAssociationList* assocList, const MultiResolutionSurfelRegistration::Params& params, double tx, double ty, double tz, double qx, double qy, double qz, double qw, bool relativeDerivatives, bool deriv2 = false, bool interpolate_neighbors = true, bool derivZ = false ) {

		assocList_ = assocList;

		params_ = params;

		const double inv_qw = 1.0 / qw;

		relativeDerivatives_ = relativeDerivatives;
		deriv2_ = deriv2;
		derivZ_ = derivZ;
		interpolate_neighbors_ = interpolate_neighbors;

		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = tx;
		currentTransform(1,3) = ty;
		currentTransform(2,3) = tz;


//		cov_cc_add.setIdentity();
//		cov_cc_add *= SMOOTH_COLOR_COVARIANCE;

		currentRotation = Eigen::Matrix3d( currentTransform.block<3,3>(0,0) );
		currentRotationT = currentRotation.transpose();
		currentTranslation = Eigen::Vector3d( currentTransform.block<3,1>(0,3) );


		// build up derivatives of rotation and translation for the transformation variables
		dt_tx(0) = 1.f; dt_tx(1) = 0.f; dt_tx(2) = 0.f;
		dt_ty(0) = 0.f; dt_ty(1) = 1.f; dt_ty(2) = 0.f;
		dt_tz(0) = 0.f; dt_tz(1) = 0.f; dt_tz(2) = 1.f;


		if( relativeDerivatives_ ) {

			dR_qx.setZero();
			dR_qx(1,2) = -2;
			dR_qx(2,1) = 2;

//			dR_qx = (dR_qx * currentRotation).eval();


			dR_qy.setZero();
			dR_qy(0,2) = 2;
			dR_qy(2,0) = -2;

//			dR_qy = (dR_qy * currentRotation).eval();


			dR_qz.setZero();
			dR_qz(0,1) = -2;
			dR_qz(1,0) = 2;

//			dR_qz = (dR_qz * currentRotation).eval();

		}
		else {

			// matrix(
			//  [ 0,
			//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy),
			//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qx,
			//    2*(qx^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz),
			//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qx^2/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qx ]
			// )
			dR_qx(0,0) = 0.0;
			dR_qx(0,1) = 2.0*((qx*qz)*inv_qw+qy);
			dR_qx(0,2) = 2.0*(qz-(qx*qy)*inv_qw);
			dR_qx(1,0) = 2.0*(qy-(qx*qz)*inv_qw);
			dR_qx(1,1) = -4.0*qx;
			dR_qx(1,2) = 2.0*(qx*qx*inv_qw-qw);
			dR_qx(2,0) = 2.0*((qx*qy)*inv_qw+qz);
			dR_qx(2,1) = 2.0*(qw-qx*qx*inv_qw);
			dR_qx(2,2) = -4.0*qx;

			// matrix(
			//  [ -4*qy,
			//    2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
			//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qy^2/sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    0,
			//    2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz) ],
			//  [ 2*(qy^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
			//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qy ]
			// )

			dR_qy(0,0) = -4.0*qy;
			dR_qy(0,1) = 2.0*((qy*qz)*inv_qw+qx);
			dR_qy(0,2) = 2.0*(qw-qy*qy*inv_qw);
			dR_qy(1,0) = 2.0*(qx-(qy*qz)*inv_qw);
			dR_qy(1,1) = 0.0;
			dR_qy(1,2) = 2.0*((qx*qy)*inv_qw+qz);
			dR_qy(2,0) = 2.0*(qy*qy*inv_qw-qw);
			dR_qy(2,1) = 2.0*(qz-(qx*qy)*inv_qw);
			dR_qy(2,2) = -4.0*qy;

			// matrix(
			//  [ -4*qz,
			//    2*(qz^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
			//    2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*(sqrt(-qz^2-qy^2-qx^2+1)-qz^2/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qz,
			//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy) ],
			//  [ 2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
			//    2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    0 ]
			// )
			dR_qz(0,0) = -4.0*qz;
			dR_qz(0,1) = 2.0*(qz*qz*inv_qw-qw);
			dR_qz(0,2) = 2.0*(qx-(qy*qz)*inv_qw);
			dR_qz(1,0) = 2.0*(qw-qz*qz*inv_qw);
			dR_qz(1,1) = -4.0*qz;
			dR_qz(1,2) = 2.0*((qx*qz)*inv_qw+qy);
			dR_qz(2,0) = 2.0*((qy*qz)*inv_qw+qx);
			dR_qz(2,1) = 2.0*(qy-(qx*qz)*inv_qw);
			dR_qz(2,2) = 0.0;

		}


		dR_qxT = dR_qx.transpose();
		dR_qyT = dR_qy.transpose();
		dR_qzT = dR_qz.transpose();


		ddiff_s_tx.block<3,1>(0,0) = -dt_tx;
		ddiff_s_ty.block<3,1>(0,0) = -dt_ty;
		ddiff_s_tz.block<3,1>(0,0) = -dt_tz;

		if( deriv2_ ) {

			if( relativeDerivatives_ ) {

				d2R_qxx( 0, 0 ) = 0;
				d2R_qxx( 0, 1 ) = 0;
				d2R_qxx( 0, 2 ) = 0;
				d2R_qxx( 1, 0 ) = 0;
				d2R_qxx( 1, 1 ) = -4.0;
				d2R_qxx( 1, 2 ) = 0;
				d2R_qxx( 2, 0 ) = 0;
				d2R_qxx( 2, 1 ) = 0;
				d2R_qxx( 2, 2 ) = -4.0;

//				d2R_qxx = (d2R_qxx * currentRotation).eval();


				d2R_qxy( 0, 0 ) = 0.0;
				d2R_qxy( 0, 1 ) = 2;
				d2R_qxy( 0, 2 ) = 0;
				d2R_qxy( 1, 0 ) = 2;
				d2R_qxy( 1, 1 ) = 0.0;
				d2R_qxy( 1, 2 ) = 0;
				d2R_qxy( 2, 0 ) = 0;
				d2R_qxy( 2, 1 ) = 0;
				d2R_qxy( 2, 2 ) = 0.0;

//				d2R_qxy = (d2R_qxy * currentRotation).eval();


				d2R_qxz( 0, 0 ) = 0.0;
				d2R_qxz( 0, 1 ) = 0;
				d2R_qxz( 0, 2 ) = 2;
				d2R_qxz( 1, 0 ) = 0;
				d2R_qxz( 1, 1 ) = 0.0;
				d2R_qxz( 1, 2 ) = 0;
				d2R_qxz( 2, 0 ) = 2;
				d2R_qxz( 2, 1 ) = 0;
				d2R_qxz( 2, 2 ) = 0.0;

//				d2R_qxz = (d2R_qxz * currentRotation).eval();


				d2R_qyy( 0, 0 ) = -4.0;
				d2R_qyy( 0, 1 ) = 0;
				d2R_qyy( 0, 2 ) = 0;
				d2R_qyy( 1, 0 ) = 0;
				d2R_qyy( 1, 1 ) = 0.0;
				d2R_qyy( 1, 2 ) = 0;
				d2R_qyy( 2, 0 ) = 0;
				d2R_qyy( 2, 1 ) = 0;
				d2R_qyy( 2, 2 ) = -4.0;

//				d2R_qyy = (d2R_qyy * currentRotation).eval();


				d2R_qyz( 0, 0 ) = 0.0;
				d2R_qyz( 0, 1 ) = 0;
				d2R_qyz( 0, 2 ) = 0;
				d2R_qyz( 1, 0 ) = 0;
				d2R_qyz( 1, 1 ) = 0.0;
				d2R_qyz( 1, 2 ) = 2;
				d2R_qyz( 2, 0 ) = 0;
				d2R_qyz( 2, 1 ) = 2;
				d2R_qyz( 2, 2 ) = 0.0;

//				d2R_qyz = (d2R_qyz * currentRotation).eval();


				d2R_qzz( 0, 0 ) = -4.0;
				d2R_qzz( 0, 1 ) = 0;
				d2R_qzz( 0, 2 ) = 0;
				d2R_qzz( 1, 0 ) = 0;
				d2R_qzz( 1, 1 ) = -4.0;
				d2R_qzz( 1, 2 ) = 0;
				d2R_qzz( 2, 0 ) = 0;
				d2R_qzz( 2, 1 ) = 0;
				d2R_qzz( 2, 2 ) = 0.0;

//				d2R_qzz = (d2R_qzz * currentRotation).eval();


			}
			else {

				const double inv_qw3 = inv_qw*inv_qw*inv_qw;

				// matrix(
				// [ 0,
				//   2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4,
				//   2*((3*qx)/sqrt(-qz^2-qy^2-qx^2+1)+qx^3/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-(3*qx)/sqrt(-qz^2-qy^2-qx^2+1)-qx^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4 ] )
				d2R_qxx( 0, 0 ) = 0;
				d2R_qxx( 0, 1 ) = 2.0*(qz*inv_qw+qx*qx*qz*inv_qw3);
				d2R_qxx( 0, 2 ) = 2.0*(-qy*inv_qw-qx*qx*qy*inv_qw3);
				d2R_qxx( 1, 0 ) = 2.0*(-qz*inv_qw-qx*qx*qz*inv_qw3);
				d2R_qxx( 1, 1 ) = -4.0;
				d2R_qxx( 1, 2 ) = 2.0*(3.0*qx*inv_qw+qx*qx*qx*inv_qw3);
				d2R_qxx( 2, 0 ) = 2.0*(qy*inv_qw+qx*qx*qy*inv_qw3);
				d2R_qxx( 2, 1 ) = 2.0*(-3.0*qx*inv_qw-qx*qx*qx*inv_qw3);
				d2R_qxx( 2, 2 ) = -4.0;


				// matrix(
				// [ 0,
				//   2*((qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)+1),
				//   2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(1-(qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0 ] )
				d2R_qxy( 0, 0 ) = 0.0;
				d2R_qxy( 0, 1 ) = 2.0*(qx*qy*qz*inv_qw3+1.0);
				d2R_qxy( 0, 2 ) = 2.0*(-qx*inv_qw-qx*qy*qy*inv_qw3);
				d2R_qxy( 1, 0 ) = 2.0*(1.0-qx*qy*qz*inv_qw3);
				d2R_qxy( 1, 1 ) = 0.0;
				d2R_qxy( 1, 2 ) = 2.0*(qy*inv_qw+qx*qx*qy*inv_qw3);
				d2R_qxy( 2, 0 ) = 2.0*(qx*inv_qw+qx*qy*qy*inv_qw3);
				d2R_qxy( 2, 1 ) = 2.0*(-qy*inv_qw-qx*qx*qy*inv_qw3);
				d2R_qxy( 2, 2 ) = 0.0;


				// matrix(
				// [ 0,
				//   2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(1-(qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*((qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)+1),
				//   2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				// 0 ])
				d2R_qxz( 0, 0 ) = 0.0;
				d2R_qxz( 0, 1 ) = 2.0*(qx*inv_qw+qx*qz*qz*inv_qw3);
				d2R_qxz( 0, 2 ) = 2.0*(1.0-qx*qy*qz*inv_qw3);
				d2R_qxz( 1, 0 ) = 2.0*(-qx*inv_qw-qx*qz*qz*inv_qw3);
				d2R_qxz( 1, 1 ) = 0.0;
				d2R_qxz( 1, 2 ) = 2.0*(qz*inv_qw+qx*qx*qz*inv_qw3);
				d2R_qxz( 2, 0 ) = 2.0*(qx*qy*qz*inv_qw3+1.0);
				d2R_qxz( 2, 1 ) = 2.0*(-qz*inv_qw-qx*qx*qz*inv_qw3);
				d2R_qxz( 2, 2 ) = 0.0;

				// matrix(
				// [ -4,
				//   2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-(3*qy)/sqrt(-qz^2-qy^2-qx^2+1)-qy^3/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*((3*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qy^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4 ])
				d2R_qyy( 0, 0 ) = -4.0;
				d2R_qyy( 0, 1 ) = 2.0*(qz*inv_qw+qy*qy*qz*inv_qw3);
				d2R_qyy( 0, 2 ) = 2.0*(-3.0*qy*inv_qw-qy*qy*qy*inv_qw3);
				d2R_qyy( 1, 0 ) = 2.0*(-qz*inv_qw-qy*qy*qz*inv_qw3);
				d2R_qyy( 1, 1 ) = 0.0;
				d2R_qyy( 1, 2 ) = 2.0*(qx*inv_qw+qx*qy*qy*inv_qw3);
				d2R_qyy( 2, 0 ) = 2.0*(3.0*qy*inv_qw+qy*qy*qy*inv_qw3);
				d2R_qyy( 2, 1 ) = 2.0*(-qx*inv_qw-qx*qy*qy*inv_qw3);
				d2R_qyy( 2, 2 ) = -4.0;

				// matrix(
				// [ 0,
				//   2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*((qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)+1) ],
				// [ 2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(1-(qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0 ])
				d2R_qyz( 0, 0 ) = 0.0;
				d2R_qyz( 0, 1 ) = 2.0*(qy*inv_qw+qy*qz*qz*inv_qw3);
				d2R_qyz( 0, 2 ) = 2.0*(-qz*inv_qw-qy*qy*qz*inv_qw3);
				d2R_qyz( 1, 0 ) = 2.0*(-qy*inv_qw-qy*qz*qz*inv_qw3);
				d2R_qyz( 1, 1 ) = 0.0;
				d2R_qyz( 1, 2 ) = 2.0*(qx*qy*qz*inv_qw3+1.0);
				d2R_qyz( 2, 0 ) = 2.0*(qz*inv_qw+qy*qy*qz*inv_qw3);
				d2R_qyz( 2, 1 ) = 2.0*(1.0-qx*qy*qz*inv_qw3);
				d2R_qyz( 2, 2 ) = 0.0;

				// matrix(
				// [ -4,
				//   2*((3*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qz^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-(3*qz)/sqrt(-qz^2-qy^2-qx^2+1)-qz^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4,
				//   2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0 ])
				d2R_qzz( 0, 0 ) = -4.0;
				d2R_qzz( 0, 1 ) = 2.0*(3.0*qz*inv_qw+qz*qz*qz*inv_qw3);
				d2R_qzz( 0, 2 ) = 2.0*(-qy*inv_qw-qy*qz*qz*inv_qw3);
				d2R_qzz( 1, 0 ) = 2.0*(-3.0*qz*inv_qw-qz*qz*qz*inv_qw3);
				d2R_qzz( 1, 1 ) = -4.0;
				d2R_qzz( 1, 2 ) = 2.0*(qx*inv_qw+qx*qz*qz*inv_qw3);
				d2R_qzz( 2, 0 ) = 2.0*(qy*inv_qw+qy*qz*qz*inv_qw3);
				d2R_qzz( 2, 1 ) = 2.0*(-qx*inv_qw-qx*qz*qz*inv_qw3);
				d2R_qzz( 2, 2 ) = 0.0;

			}

			d2R_qxxT = d2R_qxx.transpose();
			d2R_qxyT = d2R_qxy.transpose();
			d2R_qxzT = d2R_qxz.transpose();
			d2R_qyyT = d2R_qyy.transpose();
			d2R_qyzT = d2R_qyz.transpose();
			d2R_qzzT = d2R_qzz.transpose();


			if( derivZ_ ) {

				// needed for the derivatives for the measurements

				ddiff_dzmx = Eigen::Vector3d( 1.0, 0.0, 0.0 );
				ddiff_dzmy = Eigen::Vector3d( 0.0, 1.0, 0.0 );
				ddiff_dzmz = Eigen::Vector3d( 0.0, 0.0, 1.0 );

				ddiff_dzsx = -currentRotation * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				ddiff_dzsy = -currentRotation * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				ddiff_dzsz = -currentRotation * Eigen::Vector3d( 0.0, 0.0, 1.0 );

				d2diff_qx_zsx = -dR_qx * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				d2diff_qx_zsy = -dR_qx * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				d2diff_qx_zsz = -dR_qx * Eigen::Vector3d( 0.0, 0.0, 1.0 );
				d2diff_qy_zsx = -dR_qy * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				d2diff_qy_zsy = -dR_qy * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				d2diff_qy_zsz = -dR_qy * Eigen::Vector3d( 0.0, 0.0, 1.0 );
				d2diff_qz_zsx = -dR_qz * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				d2diff_qz_zsy = -dR_qz * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				d2diff_qz_zsz = -dR_qz * Eigen::Vector3d( 0.0, 0.0, 1.0 );

			}

		}

	}

	~GradientFunctor() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*assocList_)[i]);
	}



	void operator()( MultiResolutionSurfelRegistration::SurfelAssociation& assoc ) const {


		if( assoc.match == 0 || !assoc.src_->applyUpdate_ || !assoc.dst_->applyUpdate_ ) {
			assoc.match = 0;
			return;
		}

		const float processResolution = assoc.n_src_->resolution();
		double weight = assoc.weight;

		Eigen::Vector4d pos;
		pos.block<3,1>(0,0) = assoc.src_->mean_.block<3,1>(0,0);
		pos(3,0) = 1.f;

		const Eigen::Vector4d pos_src = currentTransform * pos;

		double error = 0;

		double de_tx = 0;
		double de_ty = 0;
		double de_tz = 0;
		double de_qx = 0;
		double de_qy = 0;
		double de_qz = 0;

		Eigen::Matrix< double, 6, 6 > d2J_pp;
		Eigen::Matrix< double, 6, 6 > JSzJ;
		if( deriv2_ ) {
			d2J_pp.setZero();
			JSzJ.setZero();
		}



		// spatial component, marginalized

		Eigen::Matrix3d cov_ss_add;
		cov_ss_add.setZero();
		if( params_.add_smooth_pos_covariance_ ) {
			cov_ss_add.setIdentity();
			cov_ss_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
		}

		Eigen::Matrix3d cov1_ss;
		Eigen::Matrix3d cov2_ss = assoc.src_->cov_.block<3,3>(0,0) + cov_ss_add;

		Eigen::Vector3d dstMean;
		Eigen::Vector3d srcMean = assoc.src_->mean_.block<3,1>(0,0);

		bool in_interpolation_range = false;

		if( interpolate_neighbors_ ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center
			const float resolution = processResolution;
			Eigen::Vector3d centerDiff = assoc.n_dst_->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
			if( resolution - fabsf(centerDiff(0)) > 0  && resolution - fabsf(centerDiff(1)) > 0  && resolution - fabsf(centerDiff(2)) > 0 ) {

				in_interpolation_range = true;

				// associate with neighbors for which distance to the node center is smaller than resolution

				dstMean.setZero();
				cov1_ss.setZero();

				double sumWeight = 0.f;
				double sumWeight2 = 0.f;

				for( int s = 0; s < 27; s++ ) {

					spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_dst_n = assoc.n_dst_->neighbors_[s];

					if(!n_dst_n)
						continue;

					MultiResolutionSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[assoc.dst_idx_];
					if( dst_n->num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
						continue;

					Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
					const double dx = resolution - fabsf(centerDiff_n(0));
					const double dy = resolution - fabsf(centerDiff_n(1));
					const double dz = resolution - fabsf(centerDiff_n(2));

					if( dx > 0 && dy > 0 && dz > 0 ) {

						const double weight = dx*dy*dz;

						dstMean += weight * dst_n->mean_.block<3,1>(0,0);
						cov1_ss += weight*weight * (dst_n->cov_.block<3,3>(0,0));

						sumWeight += weight;
						sumWeight2 += weight*weight;

					}


				}

				// numerically stable?
				if( sumWeight > resolution* 1e-6 ) {
					dstMean /= sumWeight;
					cov1_ss /= sumWeight2;

				}
				else
					in_interpolation_range = false;

				cov1_ss += cov_ss_add;


			}

		}

		if( !interpolate_neighbors_ || !in_interpolation_range ) {

			dstMean = assoc.dst_->mean_.block<3,1>(0,0);
			cov1_ss = assoc.dst_->cov_.block<3,3>(0,0) + cov_ss_add;

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		cov1_ss *= params_.interpolation_cov_factor_;
		cov2_ss *= params_.interpolation_cov_factor_;

		const Eigen::Vector3d TsrcMean = pos_src.block<3,1>(0,0);
		const Eigen::Vector3d diff_s = dstMean - TsrcMean;

		const Eigen::Matrix3d Rcov2_ss = currentRotation * cov2_ss;
		const Eigen::Matrix3d Rcov2_ssT = Rcov2_ss.transpose();

		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * currentRotationT;
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();
		const Eigen::Vector3d invcov_ss_diff_s = invcov_ss * diff_s;

		error = log( cov_ss.determinant() ) + diff_s.dot(invcov_ss_diff_s);


		if( relativeDerivatives_ ) {

//			const Eigen::Matrix3d dRR_qx = dR_qx * currentRotation;
//			const Eigen::Matrix3d dRR_qy = dR_qy * currentRotation;
//			const Eigen::Matrix3d dRR_qz = dR_qz * currentRotation;
//
//			const Eigen::Matrix3d d2RR_qxx = d2R_qxx * currentRotation;
//			const Eigen::Matrix3d d2RR_qxy = d2R_qxy * currentRotation;
//			const Eigen::Matrix3d d2RR_qxz = d2R_qxz * currentRotation;
//			const Eigen::Matrix3d d2RR_qyy = d2R_qyy * currentRotation;
//			const Eigen::Matrix3d d2RR_qyz = d2R_qyz * currentRotation;
//			const Eigen::Matrix3d d2RR_qzz = d2R_qzz * currentRotation;

			const Eigen::Matrix3d Rcov2R_ss = Rcov2_ss * currentRotationT;

			const Eigen::Vector3d ddiff_s_qx = -dR_qx * TsrcMean;
			const Eigen::Vector3d ddiff_s_qy = -dR_qy * TsrcMean;
			const Eigen::Vector3d ddiff_s_qz = -dR_qz * TsrcMean;



			const Eigen::Matrix3d dcov_ss_qx = dR_qx * Rcov2R_ss + Rcov2R_ss * dR_qx.transpose();
			const Eigen::Matrix3d dcov_ss_qy = dR_qy * Rcov2R_ss + Rcov2R_ss * dR_qy.transpose();
			const Eigen::Matrix3d dcov_ss_qz = dR_qz * Rcov2R_ss + Rcov2R_ss * dR_qz.transpose();

			const Eigen::Matrix3d dinvcov_ss_qx = -invcov_ss * dcov_ss_qx * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qy = -invcov_ss * dcov_ss_qy * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qz = -invcov_ss * dcov_ss_qz * invcov_ss;

			const Eigen::Vector3d dinvcov_ss_qx_diff_s = dinvcov_ss_qx * diff_s;
			const Eigen::Vector3d dinvcov_ss_qy_diff_s = dinvcov_ss_qy * diff_s;
			const Eigen::Vector3d dinvcov_ss_qz_diff_s = dinvcov_ss_qz * diff_s;


			de_tx = 2.0 * ddiff_s_tx.dot(invcov_ss_diff_s);
			de_ty = 2.0 * ddiff_s_ty.dot(invcov_ss_diff_s);
			de_tz = 2.0 * ddiff_s_tz.dot(invcov_ss_diff_s);
			de_qx = 2.0 * ddiff_s_qx.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qx_diff_s);
			de_qy = 2.0 * ddiff_s_qy.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qy_diff_s);
			de_qz = 2.0 * ddiff_s_qz.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qz_diff_s);

			// second term: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
			// -log( (2pi)^-(3/2) (det(cov))^(-1/2) )
			// = - log( (2pi)^-(3/2) ) - log( (det(cov))^(-1/2) )
			// = const. - (-0.5) * log( det(cov) )
			// = 0.5 * log( det(cov) ) => 0.5 factor can be left out also for the exp part...
			// d(log(det(cov)))/dq = 1/det(cov) * det(cov) * tr( cov^-1 * dcov/dq )
			// = tr( cov^-1 * dcov/dq )
			de_qx += (invcov_ss * dcov_ss_qx).trace();
			de_qy += (invcov_ss * dcov_ss_qy).trace();
			de_qz += (invcov_ss * dcov_ss_qz).trace();


			if( deriv2_ ) {

				const Eigen::Vector3d d2diff_s_qxx = -d2R_qxx * TsrcMean;
				const Eigen::Vector3d d2diff_s_qxy = -d2R_qxy * TsrcMean;
				const Eigen::Vector3d d2diff_s_qxz = -d2R_qxz * TsrcMean;
				const Eigen::Vector3d d2diff_s_qyy = -d2R_qyy * TsrcMean;
				const Eigen::Vector3d d2diff_s_qyz = -d2R_qyz * TsrcMean;
				const Eigen::Vector3d d2diff_s_qzz = -d2R_qzz * TsrcMean;

				const Eigen::Matrix3d d2cov_ss_qxx = d2R_qxx * Rcov2R_ss + 2.0 * dR_qx * Rcov2R_ss * dR_qxT + Rcov2R_ss * d2R_qxxT;
				const Eigen::Matrix3d d2cov_ss_qxy = d2R_qxy * Rcov2R_ss + dR_qx * Rcov2R_ss * dR_qyT + dR_qy * Rcov2R_ss * dR_qxT + Rcov2R_ss * d2R_qxyT;
				const Eigen::Matrix3d d2cov_ss_qxz = d2R_qxz * Rcov2R_ss + dR_qx * Rcov2R_ss * dR_qzT + dR_qz * Rcov2R_ss * dR_qxT + Rcov2R_ss * d2R_qxzT;
				const Eigen::Matrix3d d2cov_ss_qyy = d2R_qyy * Rcov2R_ss + 2.0 * dR_qy * Rcov2R_ss * dR_qyT + Rcov2R_ss * d2R_qyyT;
				const Eigen::Matrix3d d2cov_ss_qyz = d2R_qyz * Rcov2R_ss + dR_qy * Rcov2R_ss * dR_qzT + dR_qz * Rcov2R_ss * dR_qyT + Rcov2R_ss * d2R_qyzT;
				const Eigen::Matrix3d d2cov_ss_qzz = d2R_qzz * Rcov2R_ss + 2.0 * dR_qz * Rcov2R_ss * dR_qzT + Rcov2R_ss * d2R_qzzT;

				const Eigen::Matrix3d d2invcov_ss_qxx = -dinvcov_ss_qx * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxx * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qx;
				const Eigen::Matrix3d d2invcov_ss_qxy = -dinvcov_ss_qy * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxy * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qxz = -dinvcov_ss_qz * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxz * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qyy = -dinvcov_ss_qy * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyy * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qyz = -dinvcov_ss_qz * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyz * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qzz = -dinvcov_ss_qz * dcov_ss_qz * invcov_ss - invcov_ss * d2cov_ss_qzz * invcov_ss - invcov_ss * dcov_ss_qz * dinvcov_ss_qz;

				const Eigen::Vector3d invcov_ss_ddiff_s_tx = invcov_ss * ddiff_s_tx;
				const Eigen::Vector3d invcov_ss_ddiff_s_ty = invcov_ss * ddiff_s_ty;
				const Eigen::Vector3d invcov_ss_ddiff_s_tz = invcov_ss * ddiff_s_tz;
				const Eigen::Vector3d invcov_ss_ddiff_s_qx = invcov_ss * ddiff_s_qx;
				const Eigen::Vector3d invcov_ss_ddiff_s_qy = invcov_ss * ddiff_s_qy;
				const Eigen::Vector3d invcov_ss_ddiff_s_qz = invcov_ss * ddiff_s_qz;

				d2J_pp(0,0) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tx );
				d2J_pp(0,1) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(0,2) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(0,3) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(0,4) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(0,5) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(1,2) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(1,3) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(1,4) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(1,5) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) = 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(2,3) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(2,4) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(2,5) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(3,0) = d2J_pp(0,3);
				d2J_pp(3,1) = d2J_pp(1,3);
				d2J_pp(3,2) = d2J_pp(2,3);
				d2J_pp(3,3) = 2.0 * d2diff_s_qxx.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qx ) + diff_s.dot( d2invcov_ss_qxx * diff_s );
				d2J_pp(3,4) = 2.0 * d2diff_s_qxy.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxy * diff_s );
				d2J_pp(3,5) = 2.0 * d2diff_s_qxz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxz * diff_s );

				d2J_pp(4,0) = d2J_pp(0,4);
				d2J_pp(4,1) = d2J_pp(1,4);
				d2J_pp(4,2) = d2J_pp(2,4);
				d2J_pp(4,3) = d2J_pp(3,4);
				d2J_pp(4,4) = 2.0 * d2diff_s_qyy.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qy ) + diff_s.dot( d2invcov_ss_qyy * diff_s );
				d2J_pp(4,5) = 2.0 * d2diff_s_qyz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qy_diff_s ) + diff_s.dot( d2invcov_ss_qyz * diff_s );

				d2J_pp(5,0) = d2J_pp(0,5);
				d2J_pp(5,1) = d2J_pp(1,5);
				d2J_pp(5,2) = d2J_pp(2,5);
				d2J_pp(5,3) = d2J_pp(3,5);
				d2J_pp(5,4) = d2J_pp(4,5);
				d2J_pp(5,5) = 2.0 * d2diff_s_qzz.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qz.dot( invcov_ss_ddiff_s_qz ) + diff_s.dot( d2invcov_ss_qzz * diff_s );


				// further terms: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
				// = dtr( cov^-1 * dcov/dq ) / dq
				// = tr( d( cov^-1 * dcov/dq ) / dq )
				// = tr( dcov^-1/dq * dcov/dq + cov^-1 * d2cov/dqq )
				d2J_pp(0,0) += (dinvcov_ss_qx * dcov_ss_qx + invcov_ss * d2cov_ss_qxx).trace();
				d2J_pp(0,1) += (dinvcov_ss_qy * dcov_ss_qx + invcov_ss * d2cov_ss_qxy).trace();
				d2J_pp(0,2) += (dinvcov_ss_qz * dcov_ss_qx + invcov_ss * d2cov_ss_qxz).trace();
				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) += (dinvcov_ss_qy * dcov_ss_qy + invcov_ss * d2cov_ss_qyy).trace();
				d2J_pp(1,2) += (dinvcov_ss_qz * dcov_ss_qy + invcov_ss * d2cov_ss_qyz).trace();
				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) += (dinvcov_ss_qz * dcov_ss_qz + invcov_ss * d2cov_ss_qzz).trace();


				if( derivZ_ ) {

					// structure: pose along rows; first model coordinates, then scene
					Eigen::Matrix< double, 6, 3 > d2J_pzm, d2J_pzs;
					d2J_pzm(0,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(1,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(2,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(3,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qx_diff_s );
					d2J_pzs(3,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(3,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(3,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(4,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qy_diff_s );
					d2J_pzs(4,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(4,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(4,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(5,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qz_diff_s );
					d2J_pzs(5,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(5,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(5,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsz.dot( invcov_ss_diff_s );


					JSzJ += d2J_pzm * cov1_ss * d2J_pzm.transpose();
					JSzJ += d2J_pzs * cov2_ss * d2J_pzs.transpose();

				}

			}

		}
		else {

			const Eigen::Vector3d ddiff_s_qx = -dR_qx * srcMean;
			const Eigen::Vector3d ddiff_s_qy = -dR_qy * srcMean;
			const Eigen::Vector3d ddiff_s_qz = -dR_qz * srcMean;

			const Eigen::Matrix3d dcov_ss_qx = dR_qx * Rcov2_ssT + Rcov2_ss * dR_qxT;
			const Eigen::Matrix3d dcov_ss_qy = dR_qy * Rcov2_ssT + Rcov2_ss * dR_qyT;
			const Eigen::Matrix3d dcov_ss_qz = dR_qz * Rcov2_ssT + Rcov2_ss * dR_qzT;

			const Eigen::Matrix3d dinvcov_ss_qx = -invcov_ss * dcov_ss_qx * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qy = -invcov_ss * dcov_ss_qy * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qz = -invcov_ss * dcov_ss_qz * invcov_ss;

			const Eigen::Vector3d dinvcov_ss_qx_diff_s = dinvcov_ss_qx * diff_s;
			const Eigen::Vector3d dinvcov_ss_qy_diff_s = dinvcov_ss_qy * diff_s;
			const Eigen::Vector3d dinvcov_ss_qz_diff_s = dinvcov_ss_qz * diff_s;


			de_tx = 2.0 * ddiff_s_tx.dot(invcov_ss_diff_s);
			de_ty = 2.0 * ddiff_s_ty.dot(invcov_ss_diff_s);
			de_tz = 2.0 * ddiff_s_tz.dot(invcov_ss_diff_s);
			de_qx = 2.0 * ddiff_s_qx.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qx_diff_s);
			de_qy = 2.0 * ddiff_s_qy.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qy_diff_s);
			de_qz = 2.0 * ddiff_s_qz.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qz_diff_s);

			// second term: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
			// -log( (2pi)^-(3/2) (det(cov))^(-1/2) )
			// = - log( (2pi)^-(3/2) ) - log( (det(cov))^(-1/2) )
			// = const. - (-0.5) * log( det(cov) )
			// = 0.5 * log( det(cov) ) => 0.5 factor can be left out also for the exp part...
			// d(log(det(cov)))/dq = 1/det(cov) * det(cov) * tr( cov^-1 * dcov/dq )
			// = tr( cov^-1 * dcov/dq )
			de_qx += (invcov_ss * dcov_ss_qx).trace();
			de_qy += (invcov_ss * dcov_ss_qy).trace();
			de_qz += (invcov_ss * dcov_ss_qz).trace();


			if( deriv2_ ) {

				const Eigen::Vector3d d2diff_s_qxx = -d2R_qxx * srcMean;
				const Eigen::Vector3d d2diff_s_qxy = -d2R_qxy * srcMean;
				const Eigen::Vector3d d2diff_s_qxz = -d2R_qxz * srcMean;
				const Eigen::Vector3d d2diff_s_qyy = -d2R_qyy * srcMean;
				const Eigen::Vector3d d2diff_s_qyz = -d2R_qyz * srcMean;
				const Eigen::Vector3d d2diff_s_qzz = -d2R_qzz * srcMean;

				const Eigen::Matrix3d d2cov_ss_qxx = d2R_qxx * Rcov2_ssT + 2.0 * dR_qx * cov2_ss * dR_qxT + Rcov2_ss * d2R_qxxT;
				const Eigen::Matrix3d d2cov_ss_qxy = d2R_qxy * Rcov2_ssT + dR_qx * cov2_ss * dR_qyT + dR_qy * cov2_ss * dR_qxT + Rcov2_ss * d2R_qxyT;
				const Eigen::Matrix3d d2cov_ss_qxz = d2R_qxz * Rcov2_ssT + dR_qx * cov2_ss * dR_qzT + dR_qz * cov2_ss * dR_qxT + Rcov2_ss * d2R_qxzT;
				const Eigen::Matrix3d d2cov_ss_qyy = d2R_qyy * Rcov2_ssT + 2.0 * dR_qy * cov2_ss * dR_qyT + Rcov2_ss * d2R_qyyT;
				const Eigen::Matrix3d d2cov_ss_qyz = d2R_qyz * Rcov2_ssT + dR_qy * cov2_ss * dR_qzT + dR_qz * cov2_ss * dR_qyT + Rcov2_ss * d2R_qyzT;
				const Eigen::Matrix3d d2cov_ss_qzz = d2R_qzz * Rcov2_ssT + 2.0 * dR_qz * cov2_ss * dR_qzT + Rcov2_ss * d2R_qzzT;

				const Eigen::Matrix3d d2invcov_ss_qxx = -dinvcov_ss_qx * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxx * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qx;
				const Eigen::Matrix3d d2invcov_ss_qxy = -dinvcov_ss_qy * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxy * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qxz = -dinvcov_ss_qz * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxz * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qyy = -dinvcov_ss_qy * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyy * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qyz = -dinvcov_ss_qz * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyz * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qzz = -dinvcov_ss_qz * dcov_ss_qz * invcov_ss - invcov_ss * d2cov_ss_qzz * invcov_ss - invcov_ss * dcov_ss_qz * dinvcov_ss_qz;

				const Eigen::Vector3d invcov_ss_ddiff_s_tx = invcov_ss * ddiff_s_tx;
				const Eigen::Vector3d invcov_ss_ddiff_s_ty = invcov_ss * ddiff_s_ty;
				const Eigen::Vector3d invcov_ss_ddiff_s_tz = invcov_ss * ddiff_s_tz;
				const Eigen::Vector3d invcov_ss_ddiff_s_qx = invcov_ss * ddiff_s_qx;
				const Eigen::Vector3d invcov_ss_ddiff_s_qy = invcov_ss * ddiff_s_qy;
				const Eigen::Vector3d invcov_ss_ddiff_s_qz = invcov_ss * ddiff_s_qz;

				d2J_pp(0,0) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tx );
				d2J_pp(0,1) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(0,2) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(0,3) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(0,4) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(0,5) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(1,2) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(1,3) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(1,4) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(1,5) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) = 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(2,3) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(2,4) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(2,5) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(3,0) = d2J_pp(0,3);
				d2J_pp(3,1) = d2J_pp(1,3);
				d2J_pp(3,2) = d2J_pp(2,3);
				d2J_pp(3,3) = 2.0 * d2diff_s_qxx.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qx ) + diff_s.dot( d2invcov_ss_qxx * diff_s );
				d2J_pp(3,4) = 2.0 * d2diff_s_qxy.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxy * diff_s );
				d2J_pp(3,5) = 2.0 * d2diff_s_qxz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxz * diff_s );

				d2J_pp(4,0) = d2J_pp(0,4);
				d2J_pp(4,1) = d2J_pp(1,4);
				d2J_pp(4,2) = d2J_pp(2,4);
				d2J_pp(4,3) = d2J_pp(3,4);
				d2J_pp(4,4) = 2.0 * d2diff_s_qyy.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qy ) + diff_s.dot( d2invcov_ss_qyy * diff_s );
				d2J_pp(4,5) = 2.0 * d2diff_s_qyz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qy_diff_s ) + diff_s.dot( d2invcov_ss_qyz * diff_s );

				d2J_pp(5,0) = d2J_pp(0,5);
				d2J_pp(5,1) = d2J_pp(1,5);
				d2J_pp(5,2) = d2J_pp(2,5);
				d2J_pp(5,3) = d2J_pp(3,5);
				d2J_pp(5,4) = d2J_pp(4,5);
				d2J_pp(5,5) = 2.0 * d2diff_s_qzz.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qz.dot( invcov_ss_ddiff_s_qz ) + diff_s.dot( d2invcov_ss_qzz * diff_s );


				// further terms: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
				// = dtr( cov^-1 * dcov/dq ) / dq
				// = tr( d( cov^-1 * dcov/dq ) / dq )
				// = tr( dcov^-1/dq * dcov/dq + cov^-1 * d2cov/dqq )
				d2J_pp(0,0) += (dinvcov_ss_qx * dcov_ss_qx + invcov_ss * d2cov_ss_qxx).trace();
				d2J_pp(0,1) += (dinvcov_ss_qy * dcov_ss_qx + invcov_ss * d2cov_ss_qxy).trace();
				d2J_pp(0,2) += (dinvcov_ss_qz * dcov_ss_qx + invcov_ss * d2cov_ss_qxz).trace();
				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) += (dinvcov_ss_qy * dcov_ss_qy + invcov_ss * d2cov_ss_qyy).trace();
				d2J_pp(1,2) += (dinvcov_ss_qz * dcov_ss_qy + invcov_ss * d2cov_ss_qyz).trace();
				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) += (dinvcov_ss_qz * dcov_ss_qz + invcov_ss * d2cov_ss_qzz).trace();


				if( derivZ_ ) {

					// structure: pose along rows; first model coordinates, then scene
					Eigen::Matrix< double, 6, 3 > d2J_pzm, d2J_pzs;
					d2J_pzm(0,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(1,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(2,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(3,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qx_diff_s );
					d2J_pzs(3,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(3,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(3,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(4,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qy_diff_s );
					d2J_pzs(4,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(4,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(4,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(5,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qz_diff_s );
					d2J_pzs(5,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(5,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(5,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsz.dot( invcov_ss_diff_s );


					JSzJ += d2J_pzm * cov1_ss * d2J_pzm.transpose();
					JSzJ += d2J_pzs * cov2_ss * d2J_pzs.transpose();

				}

			}

		}


		assoc.df_dx(0) = de_tx;
		assoc.df_dx(1) = de_ty;
		assoc.df_dx(2) = de_tz;
		assoc.df_dx(3) = de_qx;
		assoc.df_dx(4) = de_qy;
		assoc.df_dx(5) = de_qz;

		if( deriv2_ ) {
			assoc.d2f = d2J_pp;

			if( derivZ_ )
				assoc.JSzJ = JSzJ;
		}

		assoc.error = error;
		assoc.weight = weight;
		assoc.match = 1;

		assert( !boost::math::isnan(error) );




	}


	MultiResolutionSurfelRegistration::Params params_;

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

	MultiResolutionSurfelRegistration::SurfelAssociationList* assocList_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


class GradientFunctorLM {
public:


	GradientFunctorLM( MultiResolutionSurfelRegistration::SurfelAssociationList* assocList, const MultiResolutionSurfelRegistration::Params& params, double tx, double ty, double tz, double qx, double qy, double qz, double qw, bool derivs, bool derivZ = false, bool interpolate_neighbors = false ) {

		derivs_ = derivs;
		derivZ_ = derivZ;

		interpolate_neighbors_ = interpolate_neighbors;

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

			const double inv_qw = 1.0 / qw;

			// build up derivatives of rotation and translation for the transformation variables
			dt_tx(0) = 1.f; dt_tx(1) = 0.f; dt_tx(2) = 0.f;
			dt_ty(0) = 0.f; dt_ty(1) = 1.f; dt_ty(2) = 0.f;
			dt_tz(0) = 0.f; dt_tz(1) = 0.f; dt_tz(2) = 1.f;

			if( !derivZ ) {
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
			else {

                // matrix(
                //  [ 0,
                //    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy),
                //    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)) ],
                //  [ 2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
                //    -4*qx,
                //    2*(qx^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)) ],
                //  [ 2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz),
                //    2*(sqrt(-qz^2-qy^2-qx^2+1)-qx^2/sqrt(-qz^2-qy^2-qx^2+1)),
                //    -4*qx ]
                // )
                dR_qx(0,0) = 0.0;
                dR_qx(0,1) = 2.0*((qx*qz)*inv_qw+qy);
                dR_qx(0,2) = 2.0*(qz-(qx*qy)*inv_qw);
                dR_qx(1,0) = 2.0*(qy-(qx*qz)*inv_qw);
                dR_qx(1,1) = -4.0*qx;
                dR_qx(1,2) = 2.0*(qx*qx*inv_qw-qw);
                dR_qx(2,0) = 2.0*((qx*qy)*inv_qw+qz);
                dR_qx(2,1) = 2.0*(qw-qx*qx*inv_qw);
                dR_qx(2,2) = -4.0*qx;

                // matrix(
                //  [ -4*qy,
                //    2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
                //    2*(sqrt(-qz^2-qy^2-qx^2+1)-qy^2/sqrt(-qz^2-qy^2-qx^2+1)) ],
                //  [ 2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
                //    0,
                //    2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz) ],
                //  [ 2*(qy^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
                //    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)),
                //    -4*qy ]
                // )

                dR_qy(0,0) = -4.0*qy;
                dR_qy(0,1) = 2.0*((qy*qz)*inv_qw+qx);
                dR_qy(0,2) = 2.0*(qw-qy*qy*inv_qw);
                dR_qy(1,0) = 2.0*(qx-(qy*qz)*inv_qw);
                dR_qy(1,1) = 0.0;
                dR_qy(1,2) = 2.0*((qx*qy)*inv_qw+qz);
                dR_qy(2,0) = 2.0*(qy*qy*inv_qw-qw);
                dR_qy(2,1) = 2.0*(qz-(qx*qy)*inv_qw);
                dR_qy(2,2) = -4.0*qy;


                // matrix(
                //  [ -4*qz,
                //    2*(qz^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
                //    2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)) ],
                //  [ 2*(sqrt(-qz^2-qy^2-qx^2+1)-qz^2/sqrt(-qz^2-qy^2-qx^2+1)),
                //    -4*qz,
                //    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy) ],
                //  [ 2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
                //    2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
                //    0 ]
                // )
                dR_qz(0,0) = -4.0*qz;
                dR_qz(0,1) = 2.0*(qz*qz*inv_qw-qw);
                dR_qz(0,2) = 2.0*(qx-(qy*qz)*inv_qw);
                dR_qz(1,0) = 2.0*(qw-qz*qz*inv_qw);
                dR_qz(1,1) = -4.0*qz;
                dR_qz(1,2) = 2.0*((qx*qz)*inv_qw+qy);
                dR_qz(2,0) = 2.0*((qy*qz)*inv_qw+qx);
                dR_qz(2,1) = 2.0*(qy-(qx*qz)*inv_qw);
                dR_qz(2,2) = 0.0;


                // needed for the derivatives for the measurements

                ddiff_dzmx = Eigen::Vector3d( 1.0, 0.0, 0.0 );
                ddiff_dzmy = Eigen::Vector3d( 0.0, 1.0, 0.0 );
                ddiff_dzmz = Eigen::Vector3d( 0.0, 0.0, 1.0 );

                ddiff_dzsx = -currentRotation * Eigen::Vector3d( 1.0, 0.0, 0.0 );
                ddiff_dzsy = -currentRotation * Eigen::Vector3d( 0.0, 1.0, 0.0 );
                ddiff_dzsz = -currentRotation * Eigen::Vector3d( 0.0, 0.0, 1.0 );

                d2diff_qx_zsx = -dR_qx * Eigen::Vector3d( 1.0, 0.0, 0.0 );
                d2diff_qx_zsy = -dR_qx * Eigen::Vector3d( 0.0, 1.0, 0.0 );
                d2diff_qx_zsz = -dR_qx * Eigen::Vector3d( 0.0, 0.0, 1.0 );
                d2diff_qy_zsx = -dR_qy * Eigen::Vector3d( 1.0, 0.0, 0.0 );
                d2diff_qy_zsy = -dR_qy * Eigen::Vector3d( 0.0, 1.0, 0.0 );
                d2diff_qy_zsz = -dR_qy * Eigen::Vector3d( 0.0, 0.0, 1.0 );
                d2diff_qz_zsx = -dR_qz * Eigen::Vector3d( 1.0, 0.0, 0.0 );
                d2diff_qz_zsy = -dR_qz * Eigen::Vector3d( 0.0, 1.0, 0.0 );
                d2diff_qz_zsz = -dR_qz * Eigen::Vector3d( 0.0, 0.0, 1.0 );


			}

		}

	}

	~GradientFunctorLM() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*assocList_)[i]);
	}



	void operator()( MultiResolutionSurfelRegistration::SurfelAssociation& assoc ) const {


		if( assoc.match == 0 || !assoc.src_->applyUpdate_ || !assoc.dst_->applyUpdate_ ) {
			assoc.match = 0;
			return;
		}


		const float processResolution = assoc.n_src_->resolution();

		Eigen::Matrix3d cov_ss_add;
		cov_ss_add.setZero();
		if( params_.add_smooth_pos_covariance_ ) {
			cov_ss_add.setIdentity();
			cov_ss_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
		}

		Eigen::Matrix3d cov1_ss;
		Eigen::Matrix3d cov2_ss = assoc.src_->cov_.block<3,3>(0,0) + cov_ss_add;

		Eigen::Vector3d dstMean;
		Eigen::Vector3d srcMean = assoc.src_->mean_.block<3,1>(0,0);

		Eigen::Vector4d pos;
		pos.block<3,1>(0,0) = srcMean;
		pos(3,0) = 1.f;

		const Eigen::Vector4d pos_src = currentTransform * pos;

		bool in_interpolation_range = false;

		if( interpolate_neighbors_ ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center
			const float resolution = processResolution;
			Eigen::Vector3d centerDiff = assoc.n_dst_->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
			if( resolution - fabsf(centerDiff(0)) > 0  && resolution - fabsf(centerDiff(1)) > 0  && resolution - fabsf(centerDiff(2)) > 0 ) {

				in_interpolation_range = true;

				// associate with neighbors for which distance to the node center is smaller than resolution

				dstMean.setZero();
				cov1_ss.setZero();

				double sumWeight = 0.f;
				double sumWeight2 = 0.f;

				for( int s = 0; s < 27; s++ ) {

					spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_dst_n = assoc.n_dst_->neighbors_[s];

					if(!n_dst_n)
						continue;

					MultiResolutionSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[assoc.dst_idx_];
					if( dst_n->num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
						continue;

					Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
					const double dx = resolution - fabsf(centerDiff_n(0));
					const double dy = resolution - fabsf(centerDiff_n(1));
					const double dz = resolution - fabsf(centerDiff_n(2));

					if( dx > 0 && dy > 0 && dz > 0 ) {

						const double weight = dx*dy*dz;

						dstMean += weight * dst_n->mean_.block<3,1>(0,0);
						cov1_ss += weight*weight * (dst_n->cov_.block<3,3>(0,0));

						sumWeight += weight;
						sumWeight2 += weight*weight;

					}


				}

				// numerically stable?
				if( sumWeight > resolution* 1e-6 ) {
					dstMean /= sumWeight;
					cov1_ss /= sumWeight2;

				}
				else
					in_interpolation_range = false;

				cov1_ss += cov_ss_add;


			}

		}

		if( !interpolate_neighbors_ || !in_interpolation_range ) {

			dstMean = assoc.dst_->mean_.block<3,1>(0,0);
			cov1_ss = assoc.dst_->cov_.block<3,3>(0,0) + cov_ss_add;

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		cov1_ss *= params_.interpolation_cov_factor_;
		cov2_ss *= params_.interpolation_cov_factor_;


////		const float processResolution = assoc.n_src_->resolution();
////
////		Eigen::Matrix3d cov_ss_add;
////		cov_ss_add.setZero();
////		if( params_.add_smooth_pos_covariance_ ) {
////			cov_ss_add.setIdentity();
////			cov_ss_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
////		}
//
//		const Eigen::Matrix3d cov1_ss = assoc.dst_->cov_.block<3,3>(0,0);// + cov_ss_add;
//		const Eigen::Matrix3d cov2_ss = assoc.src_->cov_.block<3,3>(0,0);// + cov_ss_add;
//
//		const Eigen::Vector3d dstMean = assoc.dst_->mean_.block<3,1>(0,0);
//		const Eigen::Vector3d srcMean = assoc.src_->mean_.block<3,1>(0,0);



		const Eigen::Vector3d p_s = pos_src.block<3,1>(0,0);
		const Eigen::Vector3d diff_s = dstMean - p_s;

		const Eigen::Matrix3d cov_ss = params_.interpolation_cov_factor_ * (cov1_ss + currentRotation * cov2_ss * currentRotationT);
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();

		assoc.error = diff_s.dot(invcov_ss * diff_s);

		assoc.z = dstMean;
		assoc.h = p_s;

		if( derivs_ ) {

			assoc.dh_dx.block<3,1>(0,0) = dt_tx;
			assoc.dh_dx.block<3,1>(0,1) = dt_ty;
			assoc.dh_dx.block<3,1>(0,2) = dt_tz;
//			assoc.df_dx.block<3,1>(0,3) = dR_qx * srcMean;
//			assoc.df_dx.block<3,1>(0,4) = dR_qy * srcMean;
//			assoc.df_dx.block<3,1>(0,5) = dR_qz * srcMean;
			assoc.dh_dx.block<3,1>(0,3) = dR_qx * pos_src.block<3,1>(0,0);
			assoc.dh_dx.block<3,1>(0,4) = dR_qy * pos_src.block<3,1>(0,0);
			assoc.dh_dx.block<3,1>(0,5) = dR_qz * pos_src.block<3,1>(0,0);

			assoc.W = invcov_ss;

            if( derivZ_ ) {

                    // ddiff_s_X = df_dX.transpose * (z-f)
                    // ddiff_dzmx: simple
                    const Eigen::Vector3d invcov_ss_diff_s = assoc.W * diff_s;
                    const Eigen::Vector3d invcov_ss_ddiff_s_tx = assoc.W * dt_tx;
                    const Eigen::Vector3d invcov_ss_ddiff_s_ty = assoc.W * dt_ty;
                    const Eigen::Vector3d invcov_ss_ddiff_s_tz = assoc.W * dt_tz;
                    const Eigen::Vector3d invcov_ss_ddiff_s_qx = assoc.W * assoc.dh_dx.block<3,1>(0,3);
                    const Eigen::Vector3d invcov_ss_ddiff_s_qy = assoc.W * assoc.dh_dx.block<3,1>(0,4);
                    const Eigen::Vector3d invcov_ss_ddiff_s_qz = assoc.W * assoc.dh_dx.block<3,1>(0,5);


                    // structure: pose along rows; first model coordinates, then scene
                    Eigen::Matrix< double, 6, 3 > d2J_pzm, d2J_pzs;
                    d2J_pzm(0,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzm(0,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzm(0,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzs(0,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzs(0,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzs(0,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_tx );
                    d2J_pzm(1,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzm(1,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzm(1,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzs(1,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzs(1,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzs(1,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_ty );
                    d2J_pzm(2,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzm(2,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzm(2,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzs(2,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzs(2,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzs(2,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_tz );
                    d2J_pzm(3,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_qx );
                    d2J_pzm(3,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_qx );
                    d2J_pzm(3,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_qx );
                    d2J_pzs(3,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_qx ) + d2diff_qx_zsx.dot( invcov_ss_diff_s );
                    d2J_pzs(3,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_qx ) + d2diff_qx_zsy.dot( invcov_ss_diff_s );
                    d2J_pzs(3,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_qx ) + d2diff_qx_zsz.dot( invcov_ss_diff_s );
                    d2J_pzm(4,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_qy );
                    d2J_pzm(4,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_qy );
                    d2J_pzm(4,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_qy );
                    d2J_pzs(4,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_qy ) + d2diff_qy_zsx.dot( invcov_ss_diff_s );
                    d2J_pzs(4,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_qy ) + d2diff_qy_zsy.dot( invcov_ss_diff_s );
                    d2J_pzs(4,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_qy ) + d2diff_qy_zsz.dot( invcov_ss_diff_s );
                    d2J_pzm(5,0) = ddiff_dzmx.dot( invcov_ss_ddiff_s_qz );
                    d2J_pzm(5,1) = ddiff_dzmy.dot( invcov_ss_ddiff_s_qz );
                    d2J_pzm(5,2) = ddiff_dzmz.dot( invcov_ss_ddiff_s_qz );
                    d2J_pzs(5,0) = ddiff_dzsx.dot( invcov_ss_ddiff_s_qz ) + d2diff_qz_zsx.dot( invcov_ss_diff_s );
                    d2J_pzs(5,1) = ddiff_dzsy.dot( invcov_ss_ddiff_s_qz ) + d2diff_qz_zsy.dot( invcov_ss_diff_s );
                    d2J_pzs(5,2) = ddiff_dzsz.dot( invcov_ss_ddiff_s_qz ) + d2diff_qz_zsz.dot( invcov_ss_diff_s );

                    assoc.JSzJ.setZero();
                    assoc.JSzJ += d2J_pzm * cov1_ss * d2J_pzm.transpose();
                    assoc.JSzJ += d2J_pzs * cov2_ss * d2J_pzs.transpose();

            }


		}


		assoc.match = 1;

//		assert( !boost::math::isnan(assoc.error) );


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


	MultiResolutionSurfelRegistration::SurfelAssociationList* assocList_;

	MultiResolutionSurfelRegistration::Params params_;

	bool derivs_, derivZ_, interpolate_neighbors_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


class GradientFunctorPointFeature {
public:

	// dh/dx
	inline Eigen::Matrix<double, 3, 6> dh_dx(const Eigen::Vector3d& m,const Eigen::Matrix3d& rot, const Eigen::Vector3d& transl) const {
		Eigen::Vector3d Phi = reg_->phi(m);
		Eigen::Vector3d alpha = rot * Phi + transl;
		Eigen::Matrix<double, 3, 6> J;
		J.setZero(3, 6);

		// init d PhiInv / d Alpha
		Eigen::Matrix3d dPI_da = Eigen::Matrix3d::Zero();
		Eigen::Vector3d tmp = params_.K_ * alpha;
		double a2 = alpha(2) * alpha(2);
		dPI_da(0, 0) = params_.calibration_f_ / alpha(2);
		dPI_da(1, 1) = params_.calibration_f_ / alpha(2);
		tmp(0) /= -a2;
		tmp(1) /= -a2;
		tmp(2) /= -0.5 * alpha(2) * a2;
		tmp(0) += params_.calibration_c1_ / alpha(2);
		tmp(1) += params_.calibration_c2_ / alpha(2);
		tmp(2) += 1 / a2;
		dPI_da.block<3, 1>(0, 2) = tmp;

		Eigen::Matrix<double, 3, 3> da_dq;
		da_dq.block<3, 1>(0, 0) = dR_qx * rot * Phi;
		da_dq.block<3, 1>(0, 1) = dR_qy * rot * Phi;
		da_dq.block<3, 1>(0, 2) = dR_qz * rot * Phi;

		Eigen::Matrix<double, 3, 6> dh_dx;
		dh_dx.block<3, 3>(0, 0) = dPI_da;
		dh_dx.block<3, 3>(0, 3) = dPI_da * da_dq;

		return dh_dx;
	}

	// dh/dm
	inline Eigen::Matrix3d dh_dm(const Eigen::Vector3d& m, const Eigen::Matrix3d& rot, const Eigen::Vector3d& transl) const {

		const Eigen::Vector3d Phi = reg_->phi(m);
		const Eigen::Vector3d alpha = rot * Phi + transl;
//		Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();

		// init d PhiInv / d Alpha
		Eigen::Matrix3d dPI_da = Eigen::Matrix3d::Zero();
		Eigen::Vector3d tmp = params_.K_ * alpha;
		const double inv_a2 = 1.0 / alpha(2);
		const double inv_a22 = 1.0 / (alpha(2)*alpha(2));
//		double a2 = alpha(2) * alpha(2);
		dPI_da(0, 0) = params_.calibration_f_ * inv_a2;// / alpha(2);
		dPI_da(1, 1) = params_.calibration_f_ * inv_a2;// / alpha(2);
		tmp(0) *= -inv_a22;// /= -a2;
		tmp(1) *= -inv_a22;// /= -a2;
		tmp(2) *= -2.0 * inv_a2 * inv_a22;// /= -0.5 * alpha(2) * a2;
		tmp(0) += params_.calibration_c1_ * inv_a2;// / alpha(2);
		tmp(1) += params_.calibration_c2_ * inv_a2;// / alpha(2);
		tmp(2) += inv_a22;//1 / a2;
		dPI_da.block<3, 1>(0, 2) = tmp;

		Eigen::Matrix<double, 3, 3> da_dm = Eigen::Matrix<double, 3, 3>::Zero();
		const double inv_m22 = 1.0 / (m(2) * m(2));
		const double inv_m2 = 1.0 / m(2);
		const double inv_f = 1.0 / params_.calibration_f_;
		da_dm(0, 0) = inv_m2 * inv_f;// 1.0 / m(2) / params_.calibration_f_;
		da_dm(1, 1) = inv_m2 * inv_f;// 1.0 / m(2) / params_.calibration_f_;
		tmp(0) = -m(0) * inv_m22;
		tmp(1) = -m(1) * inv_m22;
		tmp(2) = -inv_m22;
		da_dm.block<3, 1>(0, 2) = params_.KInv_ * tmp;

		return dPI_da * da_dm;
	}

	GradientFunctorPointFeature(MultiResolutionSurfelMap* source,
			MultiResolutionSurfelMap* target,
			MultiResolutionSurfelRegistration::FeatureAssociationList* assocList,
			const MultiResolutionSurfelRegistration::Params& params,
			MultiResolutionSurfelRegistration* reg, double tx,
			double ty, double tz, double qx, double qy, double qz, double qw ) {

		source_ = source;
		target_ = target;
		assocList_ = assocList;
		params_ = params;
		reg_ = reg;

		const double inv_qw = 1.0 / qw;

		currentTransform.setIdentity();
		currentTransform.block<3, 3>(0, 0) = Eigen::Matrix3d(
				Eigen::Quaterniond(qw, qx, qy, qz));
		currentTransform(0, 3) = tx;
		currentTransform(1, 3) = ty;
		currentTransform(2, 3) = tz;


		dR_qx.setZero();
		dR_qx(1,2) = -2;
		dR_qx(2,1) = 2;

		dR_qy.setZero();
		dR_qy(0,2) = 2;
		dR_qy(2,0) = -2;

		dR_qz.setZero();
		dR_qz(0,1) = -2;
		dR_qz(1,0) = 2;

//		// matrix(
//		//  [ 0,
//		//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy),
//		//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)) ],
//		//  [ 2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    -4*qx,
//		//    2*(qx^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)) ],
//		//  [ 2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz),
//		//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qx^2/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    -4*qx ]
//		// )
//		dR_qx(0, 0) = 0.0;
//		dR_qx(0, 1) = 2.0 * ((qx * qz) * inv_qw + qy);
//		dR_qx(0, 2) = 2.0 * (qz - (qx * qy) * inv_qw);
//		dR_qx(1, 0) = 2.0 * (qy - (qx * qz) * inv_qw);
//		dR_qx(1, 1) = -4.0 * qx;
//		dR_qx(1, 2) = 2.0 * (qx * qx * inv_qw - qw);
//		dR_qx(2, 0) = 2.0 * ((qx * qy) * inv_qw + qz);
//		dR_qx(2, 1) = 2.0 * (qw - qx * qx * inv_qw);
//		dR_qx(2, 2) = -4.0 * qx;
//
//		// matrix(
//		//  [ -4*qy,
//		//    2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
//		//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qy^2/sqrt(-qz^2-qy^2-qx^2+1)) ],
//		//  [ 2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    0,
//		//    2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz) ],
//		//  [ 2*(qy^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
//		//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    -4*qy ]
//		// )
//
//		dR_qy(0, 0) = -4.0 * qy;
//		dR_qy(0, 1) = 2.0 * ((qy * qz) * inv_qw + qx);
//		dR_qy(0, 2) = 2.0 * (qw - qy * qy * inv_qw);
//		dR_qy(1, 0) = 2.0 * (qx - (qy * qz) * inv_qw);
//		dR_qy(1, 1) = 0.0;
//		dR_qy(1, 2) = 2.0 * ((qx * qy) * inv_qw + qz);
//		dR_qy(2, 0) = 2.0 * (qy * qy * inv_qw - qw);
//		dR_qy(2, 1) = 2.0 * (qz - (qx * qy) * inv_qw);
//		dR_qy(2, 2) = -4.0 * qy;
//
//		// matrix(
//		//  [ -4*qz,
//		//    2*(qz^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
//		//    2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)) ],
//		//  [ 2*(sqrt(-qz^2-qy^2-qx^2+1)-qz^2/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    -4*qz,
//		//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy) ],
//		//  [ 2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
//		//    2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
//		//    0 ]
//		// )
//		dR_qz(0, 0) = -4.0 * qz;
//		dR_qz(0, 1) = 2.0 * (qz * qz * inv_qw - qw);
//		dR_qz(0, 2) = 2.0 * (qx - (qy * qz) * inv_qw);
//		dR_qz(1, 0) = 2.0 * (qw - qz * qz * inv_qw);
//		dR_qz(1, 1) = -4.0 * qz;
//		dR_qz(1, 2) = 2.0 * ((qx * qz) * inv_qw + qy);
//		dR_qz(2, 0) = 2.0 * ((qy * qz) * inv_qw + qx);
//		dR_qz(2, 1) = 2.0 * (qy - (qx * qz) * inv_qw);
//		dR_qz(2, 2) = 0.0;


	}

	~GradientFunctorPointFeature() {
	}


	double tx, ty, tz, qx, qy, qz, qw;
	Eigen::Matrix4d currentTransform;
	Eigen::Matrix3d dR_qx, dR_qy, dR_qz;

	MultiResolutionSurfelRegistration::FeatureAssociationList* assocList_;
	MultiResolutionSurfelRegistration::Params params_;

	MultiResolutionSurfelMap* source_;
	MultiResolutionSurfelMap* target_;

	MultiResolutionSurfelRegistration* reg_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


double MultiResolutionSurfelRegistration::preparePointFeatureDerivatives( const Eigen::Matrix<double, 6, 1>& x, double qw, double mahaldist ) {
	double error = 0;

	Eigen::Matrix3d id = Eigen::Matrix3d::Identity();
	Eigen::Vector3d null = Eigen::Vector3d::Zero();

	MRCSRFAL& fal = featureAssociations_;
	Eigen::Matrix3d rot = Eigen::Matrix3d(Eigen::Quaterniond(qw, x(3), x(4), x(5)));
	Eigen::Vector3d trnsl(x(0), x(1), x(2));
	GradientFunctorPointFeature gff(source_, target_, &fal, params_, this, x(0), x(1), x(2), x(3), x(4), x(5), qw);

	for (MRCSRFAL::iterator it = fal.begin(); it != fal.end(); ++it) {

		const PointFeature& f = source_->features_[it->src_idx_];
		const PointFeature& f2 = target_->features_[it->dst_idx_];

		it->match = 1;

		const Eigen::Vector3d lmPos = it->landmark_pos; // 3D pos in target frame
		const Eigen::Vector3d lmPosA = rot * lmPos + trnsl; // 3D pos in src frame
		const Eigen::Vector3d lm25D_A = phiInv(lmPosA);
		const Eigen::Vector3d lm25D_B = phiInv(lmPos);
		const Eigen::Matrix3d SigmaA = f.invzinvcov_;
		const Eigen::Matrix3d SigmaB = f2.invzinvcov_;
		const Eigen::Vector3d z25D_A = f.invzpos_;
		const Eigen::Vector3d z25D_B = f2.invzpos_;
		const Eigen::Vector3d diffA = -(z25D_A - lm25D_A);
		const Eigen::Vector3d diffB = -(z25D_B - lm25D_B);

		const Eigen::Vector3d SdiffA = SigmaA * diffA;
		const Eigen::Vector3d SdiffB = SigmaB * diffB;

		double src_error = diffA.transpose() * SdiffA;
		double dst_error = diffB.transpose() * SdiffB;

		if (src_error > mahaldist) {
			src_error = mahaldist;
			it->match = 0;
		}

		if (dst_error > mahaldist) {
			dst_error = mahaldist;
			it->match = 0;
		}

		error += (src_error + dst_error);

		if (!it->match)
			continue;

		const Eigen::Matrix<double, 3, 6> dhA_dx = gff.dh_dx(lm25D_B, rot, trnsl);
		const Eigen::Matrix3d dhA_dm = gff.dh_dm(lm25D_B, rot, trnsl);
		const Eigen::Matrix3d dhB_dm = gff.dh_dm(lm25D_B, id, null);

		const Eigen::Matrix3d dhA_dmS = dhA_dm.transpose() * SigmaA;
		const Eigen::Matrix3d dhB_dmS = dhB_dm.transpose() * SigmaB;
		const Eigen::Matrix<double, 6, 3> dhA_dx_S = dhA_dx.transpose() * SigmaA;

		it->Hpp = dhA_dx_S * dhA_dx;
		it->Hpl = dhB_dm.transpose() * SigmaA * dhA_dx;
		it->Hll = (dhA_dmS * dhA_dm + dhB_dmS * dhB_dm);
		it->bp = dhA_dx_S * diffA;
		it->bl = (dhA_dmS * diffA + dhB_dmS * diffB);

	}
	return error;
}


bool MultiResolutionSurfelRegistration::registrationErrorFunctionWithFirstDerivative( const Eigen::Matrix< double, 6, 1 >& x, double qwSign, double& f, Eigen::Matrix< double, 6, 1 >& df_dx, MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	df_dx.setZero();

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = qwSign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctor gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, false, false, interpolate_neighbors_ );


	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );


	double numMatches = 0;
	for( MultiResolutionSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;

		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		df_dx += weight * it->df_dx;
		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;

	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else if( numMatches < params_.registration_min_num_surfels_ ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else {
		sumError = sumError / sumWeight * numMatches;
		df_dx = df_dx / sumWeight * numMatches;
	}

	if( params_.use_prior_pose_ ) {

		df_dx += 2.0 * params_.prior_pose_invcov_ * (x - params_.prior_pose_mean_);
	}

	f = sumError;
	return true;



}



bool MultiResolutionSurfelRegistration::registrationErrorFunctionWithFirstAndSecondDerivative( const Eigen::Matrix< double, 6, 1 >& x, double qwSign, bool relativeDerivative, double& f, Eigen::Matrix< double, 6, 1 >& df_dx, Eigen::Matrix< double, 6, 6 >& d2f_dx2, MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	f = 0;
	df_dx.setZero();
	d2f_dx2.setZero();

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = qwSign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctor gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, relativeDerivative, true, interpolate_neighbors_ );

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
	for( MultiResolutionSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;


		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		df_dx += weight * it->df_dx;
		d2f_dx2 += weight * it->d2f;
		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;



		if( correspondences_source_points_ ) {

			pcl::PointXYZRGB& p1 = correspondences_source_points_->points[cidx];
			pcl::PointXYZRGB& p2 = correspondences_target_points_->points[cidx];

			Eigen::Vector4f pos1 = it->n_dst_->getCenterPosition();
			Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

			p1.x = pos1(0);
			p1.y = pos1(1);
			p1.z = pos1(2);


//			p1.x = it->dst_->mean_[0];
//			p1.y = it->dst_->mean_[1];
//			p1.z = it->dst_->mean_[2];

			p1.r = nweight * 255.f;
			p1.g = 0;
			p1.b = (1.f-nweight) * 255.f;

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
//			pos.block<3,1>(0,0) = it->src_->mean_.block<3,1>(0,0);
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


	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(cidx);
		correspondences_target_points_->points.resize(cidx);
	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
//			ROS_INFO("no surfel match!");
		return false;
	}
	else if( numMatches < params_.registration_min_num_surfels_ ) {
		sumError = std::numeric_limits<double>::max();
		std::cout << "not enough surfels for robust matching " << numMatches << "\n";
		return false;
	}
	else {
		sumError = sumError / sumWeight * numMatches;
		df_dx = df_dx / sumWeight * numMatches;
		d2f_dx2 = d2f_dx2 / sumWeight * numMatches;
	}


	if( params_.use_prior_pose_ ) {

		df_dx += 2.0 * params_.prior_pose_invcov_ * (x - params_.prior_pose_mean_);
		d2f_dx2 += 2.0 * params_.prior_pose_invcov_;
	}



	f = sumError;
	return true;

}



bool MultiResolutionSurfelRegistration::registrationErrorFunctionLM( const Eigen::Matrix< double, 6, 1 >& x, double qwSign, double& f, MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionSurfelRegistration::FeatureAssociationList& featureAssociations, double mahaldist ) {

	double sumFeatureError	= 0.0;
	double sumFeatureWeight = 0.0;
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
	const double qw = qwSign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	bool retVal = false;

	if (params_.registerSurfels_) {

		GradientFunctorLM gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, false, false, interpolate_neighbors_ );

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
		for( MultiResolutionSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

			if( !it->match )
				continue;


			float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
			float weight = nweight * it->weight;

			sumSurfelError += weight * it->error;
			sumSurfelWeight += weight;
			numMatches += 1.0;//nweight;



			if( correspondences_source_points_ ) {

				pcl::PointXYZRGB& p1 = correspondences_source_points_->points[cidx];
				pcl::PointXYZRGB& p2 = correspondences_target_points_->points[cidx];

				Eigen::Vector4f pos1 = it->n_dst_->getCenterPosition();
				Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

				p1.x = pos1(0);
				p1.y = pos1(1);
				p1.z = pos1(2);


	//			p1.x = it->dst_->mean_[0];
	//			p1.y = it->dst_->mean_[1];
	//			p1.z = it->dst_->mean_[2];

				p1.r = nweight * 255.f;
				p1.g = 0;
				p1.b = (1.f-nweight) * 255.f;

				Eigen::Vector4d pos;
				pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
	//			pos.block<3,1>(0,0) = it->src_->mean_.block<3,1>(0,0);
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


		if( correspondences_source_points_ ) {
			correspondences_source_points_->points.resize(cidx);
			correspondences_target_points_->points.resize(cidx);
		}

		if( sumSurfelWeight <= 1e-10 ) {
			sumSurfelError = std::numeric_limits<double>::max();
	//			ROS_INFO("no surfel match!");
		}
		else if( numMatches < params_.registration_min_num_surfels_ ) {
			sumSurfelError = std::numeric_limits<double>::max();
			std::cout << "not enough surfels for robust matching " << numMatches << "\n";
		}
		else {
			retVal = true;
		}


		f = sumSurfelError / sumSurfelWeight * numMatches;

	}

	if (params_.registerFeatures_) {

		Eigen::Matrix3d rot = Eigen::Matrix3d(Eigen::Quaterniond(qw, qx, qy, qz));
		Eigen::Vector3d trnsl(tx, ty, tz);
		Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
		transform.block<3, 3>(0, 0) = rot;
		transform.block<3, 1>(0, 3) = trnsl;

		for (MRCSReg::FeatureAssociationList::iterator it = featureAssociations.begin(); it != featureAssociations.end(); ++it) {

//			if( !it->match )
//				continue;
			it->match = 1;

			Eigen::Vector3d src = source_->features_[it->src_idx_].invzpos_.block<3, 1>(0,0);
			Eigen::Vector3d dst = target_->features_[it->dst_idx_].invzpos_.block<3, 1>(0,0);

			Eigen::Vector3d lm = it->landmark_pos;

			double src_error = (src - phiInv(rot * lm + trnsl)).transpose() * source_->features_[it->src_idx_].invzcov_.block<3, 3>(0,0).inverse() * (src - phiInv(rot * lm + trnsl));
			double dst_error = (dst - phiInv(lm)).transpose() * target_->features_[it->dst_idx_].invzcov_.block<3, 3>(0,0).inverse() * (dst - phiInv(lm));

			if (src_error > mahaldist) {
				it->match = 0;
				src_error = mahaldist;
			}

			if (dst_error > mahaldist) {
				it->match = 0;
				dst_error = mahaldist;
			}

			it->error = src_error + dst_error;
			sumFeatureError += it->error;
			sumFeatureWeight += 1.0;
		}

		if( sumFeatureWeight > params_.pointFeatureMinNumMatches_ ) {
			retVal = true;
		}

		f += params_.pointFeatureWeight_ * sumFeatureError;// / (double) featureAssociations.size();
	}

	if( params_.use_prior_pose_ ) {
		f += (params_.prior_pose_mean_ - x).transpose() * params_.prior_pose_invcov_ * (params_.prior_pose_mean_ - x);
	}


	return true;

}



bool MultiResolutionSurfelRegistration::registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( const Eigen::Matrix< double, 6, 1 >& x, double qwSign, double& f, Eigen::Matrix< double, 6, 1 >& df, Eigen::Matrix< double, 6, 6 >& d2f, MultiResolutionSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

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
	const double qw = qwSign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctorLM gf( &surfelAssociations, params_, tx, ty, tz, qx, qy, qz, qw, true, false, interpolate_neighbors_ );

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
	for( MultiResolutionSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;


		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		const Eigen::Matrix< double, 6, 3 > JtW = weight * it->dh_dx.transpose() * it->W;

		df += JtW * (it->z - it->h);
		d2f += JtW * it->dh_dx;

		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;



		if( correspondences_source_points_ ) {

			pcl::PointXYZRGB& p1 = correspondences_source_points_->points[cidx];
			pcl::PointXYZRGB& p2 = correspondences_target_points_->points[cidx];

			Eigen::Vector4f pos1 = it->n_dst_->getCenterPosition();
			Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

			p1.x = pos1(0);
			p1.y = pos1(1);
			p1.z = pos1(2);


//			p1.x = it->dst_->mean_[0];
//			p1.y = it->dst_->mean_[1];
//			p1.z = it->dst_->mean_[2];

			p1.r = nweight * 255.f;
			p1.g = 0;
			p1.b = (1.f-nweight) * 255.f;

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
//			pos.block<3,1>(0,0) = it->src_->mean_.block<3,1>(0,0);
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


	if( correspondences_source_points_ ) {
		correspondences_source_points_->points.resize(cidx);
		correspondences_target_points_->points.resize(cidx);
	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
//			ROS_INFO("no surfel match!");
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



bool MultiResolutionSurfelRegistration::estimateTransformationNewton( Eigen::Matrix4d& transform, int coarseToFineIterations, int fineIterations ) {

	Eigen::Matrix4d initialTransform = transform;

	// coarse alignment with features
	// fine alignment without features

	float minResolution = std::min( params_.startResolution_, params_.stopResolution_ );
	float maxResolution = std::max( params_.startResolution_, params_.stopResolution_ );

	const double step_max = 0.1;
	const double step_size_coarse = 1.0;
	const double step_size_fine = 1.0;

	Eigen::Matrix4d currentTransform = transform;

	const int maxIterations = coarseToFineIterations + fineIterations;


	Eigen::Matrix< double, 6, 1 > x, last_x, df, best_x, best_g;
	Eigen::Matrix< double, 6, 6 > d2f;

	// initialize with current transform
	Eigen::Quaterniond q( currentTransform.block<3,3>(0,0) );

	x(0) = currentTransform( 0, 3 );
	x(1) = currentTransform( 1, 3 );
	x(2) = currentTransform( 2, 3 );
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	double lastWSign_ = q.w() / fabsf(q.w());


	last_x = x;


	target_->clearAssociations();

	double best_f = std::numeric_limits<double>::max();
	Eigen::Matrix4d bestTransform;
	bestTransform.setIdentity();
	best_x = x;
	best_g.setZero();


	pcl::StopWatch stopwatch;

	transform.setIdentity();
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz), qx, qy, qz ) );
	transform(0,3) = x( 0 );
	transform(1,3) = x( 1 );
	transform(2,3) = x( 2 );

	double associateTime = 0;
	double gradientTime = 0;

	MultiResolutionSurfelRegistration::SurfelAssociationList surfelAssociations;

	bool retVal = true;


	int iter = 0;
	while( iter < maxIterations ) {

		if( processTimeWatch.getTime() > params_.max_processing_time_ )
			return false;

		// stays at minresolution after coarseToFineIterations
		float searchDistFactor = 2.f;//std::max( 1.f, 1.f + 1.f * (((float)(fineIterations / 2 - iter)) / (float)(fineIterations / 2)) );
		float maxSearchDist = 2.f*maxResolution;//(minResolution + (maxResolution-minResolution) * ((float)(maxIterations - iter)) / (float)maxIterations);

		MultiResolutionSurfelRegistration::SurfelAssociationList surfelAssociations;
		if( iter < coarseToFineIterations ) {
			stopwatch.reset();
			associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, params_.use_features_ );
			double deltat = stopwatch.getTimeSeconds() * 1000.0;
			associateTime += deltat;
			interpolate_neighbors_ = false;

		}
		else {
			if( iter == coarseToFineIterations ) {
				target_->clearAssociations();
			}

			stopwatch.reset();
			associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, false );
			double deltat = stopwatch.getTimeSeconds() * 1000.0;
			associateTime += deltat;
			interpolate_neighbors_ = true;
		}


		// evaluate function and derivative
		double f = 0.0;
		stopwatch.reset();
		retVal = registrationErrorFunctionWithFirstAndSecondDerivative( x, lastWSign_, true, f, df, d2f, surfelAssociations );

		if( !retVal ) {
			df.setZero();
			d2f.setIdentity();
		}

		double deltat2 = stopwatch.getTimeSeconds() * 1000.0;
		gradientTime += deltat2;

		if( f < best_f ) {
			best_f = f;
			bestTransform = transform;
		}



		Eigen::Matrix< double, 6, 1 > lastX = x;
		Eigen::Matrix< double, 6, 6 > d2f_inv;
		d2f_inv.setZero();
		if( fabsf( d2f.determinant() ) > std::numeric_limits<double>::epsilon() ) {

			double step_size_i = step_size_fine;

			d2f_inv = d2f.inverse();
			Eigen::Matrix< double, 6, 1 > deltaX = -step_size_i * d2f_inv * df;

			last_x = x;


			double qx = x( 3 );
			double qy = x( 4 );
			double qz = x( 5 );
			double qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);

			currentTransform.setIdentity();
			currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
			currentTransform(0,3) = x( 0 );
			currentTransform(1,3) = x( 1 );
			currentTransform(2,3) = x( 2 );


			qx = deltaX( 3 );
			qy = deltaX( 4 );
			qz = deltaX( 5 );
			qw = sqrt(1.0-qx*qx-qy*qy-qz*qz);

			Eigen::Matrix4d deltaTransform = Eigen::Matrix4d::Identity();
			deltaTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
			deltaTransform(0,3) = deltaX( 0 );
			deltaTransform(1,3) = deltaX( 1 );
			deltaTransform(2,3) = deltaX( 2 );

			Eigen::Matrix4d newTransform = deltaTransform * currentTransform;

			x( 0 ) = newTransform(0,3);
			x( 1 ) = newTransform(1,3);
			x( 2 ) = newTransform(2,3);

			Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
			x( 3 ) = q_new.x();
			x( 4 ) = q_new.y();
			x( 5 ) = q_new.z();
			lastWSign_ = q_new.w() / fabsf(q_new.w());

		}


		double qx = x( 3 );
		double qy = x( 4 );
		double qz = x( 5 );
		double qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);



		if( boost::math::isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f ) {
			x = last_x;
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





bool MultiResolutionSurfelRegistration::estimateTransformationLevenbergMarquardt( Eigen::Matrix4d& transform, int maxIterations, SurfelAssociationList* surfelAssociationsArg, bool knownAssociations, bool interpolate ) {

	const bool useFeatures = params_.use_features_;

	const double tau = 10e-5;
//	const double min_gradient_size = 1e-4;
	const double min_delta = 1e-3; // was 1e-3
	const double min_error = 1e-6;

	Eigen::Matrix4d initialTransform = transform;

	float minResolution = std::min( params_.startResolution_, params_.stopResolution_ );
	float maxResolution = std::max( params_.startResolution_, params_.stopResolution_ );

	Eigen::Matrix4d currentTransform = transform;



	// initialize with current transform
	Eigen::Matrix< double, 6, 1 > x;
	Eigen::Quaterniond q( currentTransform.block<3,3>(0,0) );

	x(0) = currentTransform( 0, 3 );
	x(1) = currentTransform( 1, 3 );
	x(2) = currentTransform( 2, 3 );
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	double lastWSign_ = q.w() / fabsf(q.w());


	pcl::StopWatch stopwatch;

	Eigen::Matrix< double, 6, 1 > df;
	Eigen::Matrix< double, 6, 6 > d2f;

	const Eigen::Matrix< double, 6, 6 > id6 = Eigen::Matrix< double, 6, 6 >::Identity();
	double mu = -1.0;
	double nu = 2;

	double last_error = std::numeric_limits<double>::max();

	MultiResolutionSurfelRegistration::SurfelAssociationList surfelAssociations;

	if( knownAssociations )
		surfelAssociations = *surfelAssociationsArg;

	bool reassociate = true;

	bool reevaluateGradient = true;

	bool retVal = true;

	int iter = 0;
	while( iter < maxIterations ) {

		if( processTimeWatch.getTime() > params_.max_processing_time_ )
			return false;

		if( reevaluateGradient ) {

			if( reassociate ) {
				target_->clearAssociations();
			}

			float searchDistFactor = 2.f;
			float maxSearchDist = 2.f*maxResolution;

			stopwatch.reset();
			surfelAssociations.clear();
			associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, useFeatures );
			double deltat = stopwatch.getTime();
			//std::cout << "assoc took: " << deltat << "\n";

			if( surfelAssociationsArg )
				*surfelAssociationsArg = surfelAssociations;

			interpolate_neighbors_ = interpolate;

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

//		double gradient_size = std::max( df.maxCoeff(), -df.minCoeff() );
//		if( last_error < min_error ) {
////			std::cout << "converged\n";
//			break;
//		}


		if( mu < 0 ) {
			mu = tau * std::max( d2f.maxCoeff(), -d2f.minCoeff() );
		}

//		std::cout << "mu: " << mu << "\n";


		Eigen::Matrix< double, 6, 1 > delta_x = Eigen::Matrix< double, 6, 1 >::Zero();
		Eigen::Matrix< double, 6, 6 > d2f_inv = Eigen::Matrix< double, 6, 6 >::Zero();
		if( fabsf( d2f.determinant() ) > std::numeric_limits<double>::epsilon() ) {

			d2f_inv = (d2f + mu * id6).inverse();

			delta_x = d2f_inv * df;

		}

		if( delta_x.norm() < min_delta ) {

			if( reassociate )
				break;

			reassociate = true;
			reevaluateGradient = true;
//			std::cout << "reassociating!\n";
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


//		Eigen::Matrix< double, 6, 1 > x_new = x + delta_x;
		Eigen::Matrix< double, 6, 1 > x_new;
		x_new( 0 ) = newTransform(0,3);
		x_new( 1 ) = newTransform(1,3);
		x_new( 2 ) = newTransform(2,3);

		Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
		x_new( 3 ) = q_new.x();
		x_new( 4 ) = q_new.y();
		x_new( 5 ) = q_new.z();
		double newWSign = q_new.w() / fabsf(q_new.w());


//		std::cout << "iter: " << iter << ": " << delta_x.norm() << "\n";

//		std::cout << x_new.transpose() << "\n";
//
		double new_error = 0.0;
		featureAssociations_.clear();
		bool retVal2 = registrationErrorFunctionLM( x_new, newWSign, new_error, surfelAssociations, featureAssociations_, 0 );

		if( !retVal2 )
			return false;

		double rho = (last_error - new_error) / (delta_x.transpose() * (mu * delta_x + df));


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


//		last_error = new_error;

		iter++;

	}


	return retVal;

}



bool MultiResolutionSurfelRegistration::estimateTransformationGaussNewton( Eigen::Matrix4d& transform, int maxIterations, SurfelAssociationList* surfelAssociationsArg, bool knownAssociations, bool interpolate ) {

	const bool useFeatures = params_.use_features_;

	const double min_delta = 1e-5; // was 1e-3

	Eigen::Matrix4d initialTransform = transform;

	float minResolution = std::min( params_.startResolution_, params_.stopResolution_ );
	float maxResolution = std::max( params_.startResolution_, params_.stopResolution_ );

	Eigen::Matrix4d currentTransform = transform;



	// initialize with current transform
	Eigen::Matrix< double, 6, 1 > x;
	Eigen::Quaterniond q( currentTransform.block<3,3>(0,0) );

	x(0) = currentTransform( 0, 3 );
	x(1) = currentTransform( 1, 3 );
	x(2) = currentTransform( 2, 3 );
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	double lastWSign_ = q.w() / fabsf(q.w());


	pcl::StopWatch stopwatch;

	Eigen::Matrix< double, 6, 1 > df;
	Eigen::Matrix< double, 6, 6 > d2f;

	const Eigen::Matrix< double, 6, 6 > id6 = Eigen::Matrix< double, 6, 6 >::Identity();

	double last_error = std::numeric_limits<double>::max();

	MultiResolutionSurfelRegistration::SurfelAssociationList surfelAssociations;

	if( knownAssociations )
		surfelAssociations = *surfelAssociationsArg;

	bool reassociate = true;

	bool reevaluateGradient = true;

	bool retVal = true;

	int iter = 0;
	while( iter < maxIterations ) {

		if( processTimeWatch.getTime() > params_.max_processing_time_ )
			return false;

		if( reevaluateGradient ) {

			if( reassociate ) {
				target_->clearAssociations();
			}

			float searchDistFactor = 2.f;
			float maxSearchDist = 2.f*maxResolution;

			stopwatch.reset();
			surfelAssociations.clear();
			associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, useFeatures );
			double deltat = stopwatch.getTime();
			//std::cout << "assoc took: " << deltat << "\n";

			if( surfelAssociationsArg )
				*surfelAssociationsArg = surfelAssociations;

			interpolate_neighbors_ = interpolate;

			stopwatch.reset();
			retVal = registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( x, lastWSign_, last_error, df, d2f, surfelAssociations );
			double deltat2 = stopwatch.getTime();
//			std::cout << "reg deriv took: " << deltat2 << "\n";


		}


		if( !retVal ) {
			std::cout << "GN registration failed\n";
			return false;
		}

//		double gradient_size = std::max( df.maxCoeff(), -df.minCoeff() );
//		if( last_error < min_error ) {
////			std::cout << "converged\n";
//			break;
//		}



		Eigen::Matrix< double, 6, 1 > delta_x = Eigen::Matrix< double, 6, 1 >::Zero();
		Eigen::Matrix< double, 6, 6 > d2f_inv = Eigen::Matrix< double, 6, 6 >::Zero();
		if( fabsf( d2f.determinant() ) > std::numeric_limits<double>::epsilon() ) {

			d2f_inv = (d2f).inverse();

			delta_x = d2f_inv * df;

		}

		if( delta_x.norm() < min_delta ) {
			std::cout << "GN registration converged\n";
			break;
		}


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


//		Eigen::Matrix< double, 6, 1 > x_new = x + delta_x;
		Eigen::Matrix< double, 6, 1 > x_new;
		x_new( 0 ) = newTransform(0,3);
		x_new( 1 ) = newTransform(1,3);
		x_new( 2 ) = newTransform(2,3);

		Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
		x_new( 3 ) = q_new.x();
		x_new( 4 ) = q_new.y();
		x_new( 5 ) = q_new.z();
		double newWSign = q_new.w() / fabsf(q_new.w());


//		std::cout << "iter: " << iter << ": " << delta_x.norm() << "\n";

//		std::cout << x_new.transpose() << "\n";
//
		double new_error = 0.0;
		featureAssociations_.clear();
		bool retVal2 = registrationErrorFunctionLM( x_new, newWSign, new_error, surfelAssociations, featureAssociations_, 0 );

		if( !retVal2 )
			return false;

		if( new_error >= last_error ) {
			std::cout << "GN registration converged " << iter << ": " << new_error << " " << last_error << "\n";
			break;
		}

		x = x_new;
		lastWSign_ = newWSign;

		reevaluateGradient = true;
		reassociate = true;



		qx = x( 3 );
		qy = x( 4 );
		qz = x( 5 );
		qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);



		if( boost::math::isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f ) {
			std::cout << "GN registration failed\n";
			return false;
		}


		transform.setIdentity();
		transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		transform(0,3) = x( 0 );
		transform(1,3) = x( 1 );
		transform(2,3) = x( 2 );


//		last_error = new_error;

		iter++;

	}


	return retVal;

}


bool MultiResolutionSurfelRegistration::estimateTransformationLevenbergMarquardtPF( Eigen::Matrix4d& transform, int maxIterations, double featureAssocMahalDist, double minDelta, bool resetFeatures, double& mu, double& nu ) {

	const double tau = 1e-4;
	const double min_delta = minDelta;//1e-4;
	const double min_error = 1e-6;

	if( resetFeatures ) {
		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it) {
			it->landmark_pos = target_->features_[it->dst_idx_].pos_.block<3, 1>(0,0);
		}
	}

	float minResolution = std::min( params_.startResolution_, params_.stopResolution_ );
	float maxResolution = std::max( params_.startResolution_, params_.stopResolution_ );

	MultiResolutionSurfelRegistration::SurfelAssociationList surfelAssociations;


	Eigen::Matrix4d initialTransform = transform;
	Eigen::Matrix4d currentTransform = transform;

	// initialize with current transform
	Eigen::Matrix<double, 6, 1> x;
	Eigen::Quaterniond q(currentTransform.block<3, 3>(0, 0));

	x(0) = currentTransform(0, 3);
	x(1) = currentTransform(1, 3);
	x(2) = currentTransform(2, 3);
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	double lastWSign_ = q.w() / fabsf(q.w());

	pcl::StopWatch stopwatch;

	Eigen::Matrix<double, 6, 6> compactH	= Eigen::Matrix<double, 6, 6>::Zero();
	Eigen::Matrix<double, 6, 1> rightSide	= Eigen::Matrix<double, 6, 1>::Zero();
	Eigen::Matrix<double, 6, 1> poseOFdf	= Eigen::Matrix<double, 6, 1>::Zero(); // df.block<6,1>(0,0), df = J^T * Sigma * diff

	const Eigen::Matrix<double, 6, 6> id6	= Eigen::Matrix<double, 6, 6>::Identity();
//	double mu = -1.0;
//	double nu = 2;

	double last_error = std::numeric_limits<double>::max();
	double new_error = 0.0;

	bool reassociate = true;

	bool reevaluateGradient = true;

	bool retVal = true;

	int iter = 0;
	while (iter < maxIterations) {

		if( processTimeWatch.getTime() > params_.max_processing_time_ )
			return false;

		if( params_.debugFeatures_ ) {
			// AreNo - test des BundleAdjustment
			cv::Mat sourceFrame = source_->img_rgb_.clone();
			cv::Mat targetFrame = target_->img_rgb_.clone();
			cv::Scalar color = 0;
			color.val[0] = 255;		// Blau
			cv::Scalar color2 = 0;
			color2.val[1] = 255;	// Grün
			cv::Scalar colorErr = 0;// schwarz
			cv::Scalar rot( 0, 0, 255, 0);
			Eigen::Matrix4d backTrnsf = transform.inverse();

			for( MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it ) {

				if( !it->match )
					continue;

				Eigen::Vector3d dst = target_->features_[ it->dst_idx_ ].invzpos_.block<3,1>(0,0);
				Eigen::Vector3d src = source_->features_[ it->src_idx_ ].invzpos_.block<3,1>(0,0);
		//			Eigen::Vector3d src = h( dst, backTrnsf.block<3,3>(0,0), backTrnsf.block<3,1>(0,3));
				cv::Point srcPoint( src(0) , src(1) );
				cv::Point dstPoint( dst(0) , dst(1) );

				// LM-Messungen
				cv::circle( sourceFrame, srcPoint, 2, color, 2);
				cv::circle( targetFrame, dstPoint, 2, color, 2);
	//			cv::line( frameA, srcPoint, dstPoint, rot, 1, 0, 0 );


				// LM-Schätzungen
				Eigen::Vector3d pixA = phiInv( transform.block<3,3>(0,0) * it->landmark_pos + transform.block<3,1>(0,3) );
				Eigen::Vector3d pixB = phiInv( it->landmark_pos );
				cv::Point srcLMPoint( pixA(0) , pixA(1) );
				cv::Point dstLMPoint( pixB(0) , pixB(1) );

				if ( (pixA - src).block<2,1>(0,0).norm() > 10 )
				{
					cv::circle( sourceFrame, srcLMPoint, 4, colorErr, 2);
					cv::line( sourceFrame, srcLMPoint, srcPoint, colorErr, 1, 0, 0 );
				}
				else
					cv::circle( sourceFrame, srcLMPoint, 4, color2, 2);

				if ( (pixB - dst).block<2,1>(0,0).norm() > 10 )
				{
					cv::circle( targetFrame, dstLMPoint, 4, colorErr, 2);
					cv::line( targetFrame, dstLMPoint, dstPoint, colorErr, 1, 0, 0 );
				}
				else
					cv::circle( targetFrame, dstLMPoint, 4, color2, 2);
			}
			cv::imshow( "TargetFrame", targetFrame);
			cv::imshow( "SourceFrame", sourceFrame);
			cv::waitKey(10);
//			while( 	cv::waitKey(10) == -1 );
		}


		Eigen::Matrix<double, 6, 1> deltaS		= Eigen::Matrix<double, 6, 1>::Zero();
		Eigen::Matrix<double, 6, 6> surfeld2f	= Eigen::Matrix<double, 6, 6>::Zero();
		Eigen::Matrix<double, 6, 1> surfeldf	= Eigen::Matrix<double, 6, 1>::Zero();

//		double coarsefactor = 1.0 - (double)iter / (double)maxIterations;
//		double featureAssocMahalDist = params_.pointFeatureMatchingFineImagePosMahalDist_ + coarsefactor * (params_.pointFeatureMatchingCoarseImagePosMahalDist_ - params_.pointFeatureMatchingFineImagePosMahalDist_);

//		int numMatches = 0;
//		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
//		{
//			if (it->match == 0)
//				continue;
//
//			numMatches++;
//		}
//
//		double unmatchedFraction = 1.0 - (double)numMatches / (double)featureAssociations_.size();
//		double featureAssocMahalDist = params_.pointFeatureMatchingFineImagePosMahalDist_ + unmatchedFraction * (params_.pointFeatureMatchingCoarseImagePosMahalDist_ - params_.pointFeatureMatchingFineImagePosMahalDist_);
//
//		std::cout << iter << " " << featureAssocMahalDist << "\n";

		if (reevaluateGradient) {

			if (reassociate) {
				target_->clearAssociations();
			}

			stopwatch.reset();

			last_error = 0.0;
			compactH.setZero();
			rightSide.setZero();
			poseOFdf.setZero();
			if (params_.registerFeatures_)
			{
				stopwatch.reset();
				last_error = preparePointFeatureDerivatives(x, q.w(), featureAssocMahalDist);

				int numMatches = 0;
				for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
				{
					if (it->match == 0)
						continue;

					numMatches++;

					Eigen::Matrix<double, 6, 3> tmp = it->Hpl.transpose() * it->Hll.inverse();
					compactH += it->Hpp - tmp * it->Hpl;
					rightSide += tmp * it->bl - it->bp;
					poseOFdf += -it->bp;

				}

				last_error *= params_.pointFeatureWeight_;
				compactH *= params_.pointFeatureWeight_;
				rightSide *= params_.pointFeatureWeight_;
				poseOFdf *= params_.pointFeatureWeight_;

				if( params_.debugFeatures_ )
					std::cout << "matched " << numMatches << "/" << featureAssociations_.size() << " associated pointfeatures\n";

				if( params_.debugFeatures_ )
					std::cout << "feature preparation took: " << stopwatch.getTime() << "\n";
			}

			if (params_.registerSurfels_)
			{
				float searchDistFactor = 2.f;
				float maxSearchDist = 2.f * maxResolution;

				stopwatch.reset();

				surfelAssociations.clear();
				associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, minResolution, maxResolution, searchDistFactor, maxSearchDist, false);

				double surfelError = 0;
				double deltat = stopwatch.getTime();
				// std::cout << "assoc took: " << deltat << "\n";

				interpolate_neighbors_ = true;

				if (!registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( x, lastWSign_, surfelError, surfeldf, surfeld2f, surfelAssociations))
				{
					std::cout << "Surfelregistration failed ------\n";
				}
				else
				{
					compactH += surfeld2f;
					rightSide += surfeldf;
					last_error += surfelError;
					poseOFdf += surfeldf;
				}
			}

		}

		reevaluateGradient = false;

		if (!retVal) {
			std::cout << "registration failed\n";
			return false;
		}

		if (mu < 0) {
			mu = tau * std::max(compactH.maxCoeff(), -compactH.minCoeff());
		}

		Eigen::Matrix<double, 6, 1> delta_x	= Eigen::Matrix<double, 6, 1>::Zero();
		Eigen::Matrix<double, 6, 6> d2f		= compactH + mu * Eigen::Matrix<double, 6, 6>::Identity();

		// delta_x für feature
		if (fabsf(d2f.determinant()) > std::numeric_limits<double>::epsilon())
		{
			delta_x = d2f.inverse() * rightSide;
		}
		else {
			std::cout << "Det(d2f) =\t" << d2f.determinant() << "\n";
		}


		if (delta_x.norm() < min_delta) {

			if (reassociate) {
				break;
			}

			reassociate = true;
			reevaluateGradient = true;
		} else
			reassociate = false;


		double qx = x( 3 );
		double qy = x( 4 );
		double qz = x( 5 );
		double qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);

		Eigen::Matrix4d currentTransform;
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


//		FeatureAssociationList FALcopy;
//		FALcopy.reserve( featureAssociations_.size() );
//		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
//		{
//			MultiResolutionSurfelRegistration::FeatureAssociation assoc( it->src_idx_, it->dst_idx_ );
//
//			if (it->match == 0) {
//				assoc.landmark_pos = it->landmark_pos;
//				FALcopy.push_back(assoc);
//				continue;
//			}
//
//			Eigen::Vector3d deltaLM = (it->Hll + mu * Eigen::Matrix3d::Identity()).inverse() * (-it->bl - it->Hpl * deltaS);
//
//			assoc.landmark_pos = phi(phiInv(it->landmark_pos) + deltaLM);
//
//			FALcopy.push_back(assoc);
//		}

		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
		{

			it->tmp_landmark_pos = it->landmark_pos;

			if (it->match == 0) {
				continue;
			}

			Eigen::Vector3d deltaLM = (it->Hll + mu * Eigen::Matrix3d::Identity()).inverse() * (-it->bl - it->Hpl * deltaS);

			it->landmark_pos = phi(phiInv(it->landmark_pos) + deltaLM);

		}

		stopwatch.reset();

		new_error = 0.0;
		bool retVal2 = registrationErrorFunctionLM(x_new, newWSign, new_error, surfelAssociations, featureAssociations_, featureAssocMahalDist);
		if (!retVal2)
		{
			std::cout << "2nd ErrorFunction for AreNo and MultiResolutionSurfelMap::Surfel failed\n";
			return false;
		}

		if( params_.debugFeatures_ )
			std::cout << "feature error function eval took: " << stopwatch.getTime() << "\n";

		double rho = (last_error - new_error) / (delta_x.transpose() * (mu * delta_x + poseOFdf));

		if (rho > 0) {

			x = x_new;
			lastWSign_ = newWSign;

//			MRCSRFAL::iterator it2 = FALcopy.begin();
//			for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it) {
//				it->landmark_pos = it2->landmark_pos;
//				it2++;
//			}

			mu *= std::max(0.333, 1.0 - pow(2.0 * rho - 1.0, 3.0));
			nu = 2;

			reevaluateGradient = true;

		} else {

			mu *= nu;
			nu *= 2.0;

			for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it) {
				it->landmark_pos = it->tmp_landmark_pos;
			}
		}

		qx = x(3);
		qy = x(4);
		qz = x(5);
		qw = lastWSign_ * sqrt(1.0 - qx * qx - qy * qy - qz * qz);

		if (boost::math::isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f) {
			return false;
		}

		transform.setIdentity();
		transform.block<3, 3>(0, 0) = Eigen::Matrix3d(
				Eigen::Quaterniond(qw, qx, qy, qz));
		transform(0, 3) = x(0);
		transform(1, 3) = x(1);
		transform(2, 3) = x(2);

		iter++;
	}
	return true;
}



bool MultiResolutionSurfelRegistration::estimateTransformationGaussNewtonPF( Eigen::Matrix4d& transform, int maxIterations, double featureAssocMahalDist, double minDelta, bool resetFeatures ) {

	const double min_delta = minDelta;//1e-4;

	if( resetFeatures ) {
		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it) {
			it->landmark_pos = target_->features_[it->dst_idx_].pos_.block<3, 1>(0,0);
		}
	}

	float minResolution = std::min( params_.startResolution_, params_.stopResolution_ );
	float maxResolution = std::max( params_.startResolution_, params_.stopResolution_ );

	MultiResolutionSurfelRegistration::SurfelAssociationList surfelAssociations;


	Eigen::Matrix4d initialTransform = transform;
	Eigen::Matrix4d currentTransform = transform;

	// initialize with current transform
	Eigen::Matrix<double, 6, 1> x;
	Eigen::Quaterniond q(currentTransform.block<3, 3>(0, 0));

	x(0) = currentTransform(0, 3);
	x(1) = currentTransform(1, 3);
	x(2) = currentTransform(2, 3);
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	double lastWSign_ = q.w() / fabsf(q.w());

	pcl::StopWatch stopwatch;

	Eigen::Matrix<double, 6, 6> compactH	= Eigen::Matrix<double, 6, 6>::Zero();
	Eigen::Matrix<double, 6, 1> rightSide	= Eigen::Matrix<double, 6, 1>::Zero();
	Eigen::Matrix<double, 6, 1> poseOFdf	= Eigen::Matrix<double, 6, 1>::Zero(); // df.block<6,1>(0,0), df = J^T * Sigma * diff

	const Eigen::Matrix<double, 6, 6> id6	= Eigen::Matrix<double, 6, 6>::Identity();

	bool reassociate = true;
	bool reevaluateGradient = true;

	bool retVal = true;

	int iter = 0;
	while (iter < maxIterations) {

		if( processTimeWatch.getTime() > params_.max_processing_time_ )
			return false;

		if( params_.debugFeatures_ ) {
			// AreNo - test des BundleAdjustment
			cv::Mat sourceFrame = source_->img_rgb_.clone();
			cv::Mat targetFrame = target_->img_rgb_.clone();
			cv::Scalar color = 0;
			color.val[0] = 255;		// Blau
			cv::Scalar color2 = 0;
			color2.val[1] = 255;	// Grün
			cv::Scalar colorErr = 0;// schwarz
			cv::Scalar rot( 0, 0, 255, 0);
			Eigen::Matrix4d backTrnsf = transform.inverse();

			for( MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it ) {

				if( !it->match )
					continue;

				Eigen::Vector3d dst = target_->features_[ it->dst_idx_ ].invzpos_.block<3,1>(0,0);
				Eigen::Vector3d src = source_->features_[ it->src_idx_ ].invzpos_.block<3,1>(0,0);
		//			Eigen::Vector3d src = h( dst, backTrnsf.block<3,3>(0,0), backTrnsf.block<3,1>(0,3));
				cv::Point srcPoint( src(0) , src(1) );
				cv::Point dstPoint( dst(0) , dst(1) );

				// LM-Messungen
				cv::circle( sourceFrame, srcPoint, 2, color, 2);
				cv::circle( targetFrame, dstPoint, 2, color, 2);
	//			cv::line( frameA, srcPoint, dstPoint, rot, 1, 0, 0 );


				// LM-Schätzungen
				Eigen::Vector3d pixA = phiInv( transform.block<3,3>(0,0) * it->landmark_pos + transform.block<3,1>(0,3) );
				Eigen::Vector3d pixB = phiInv( it->landmark_pos );
				cv::Point srcLMPoint( pixA(0) , pixA(1) );
				cv::Point dstLMPoint( pixB(0) , pixB(1) );

				if ( (pixA - src).block<2,1>(0,0).norm() > 10 )
				{
					cv::circle( sourceFrame, srcLMPoint, 4, colorErr, 2);
					cv::line( sourceFrame, srcLMPoint, srcPoint, colorErr, 1, 0, 0 );
				}
				else
					cv::circle( sourceFrame, srcLMPoint, 4, color2, 2);

				if ( (pixB - dst).block<2,1>(0,0).norm() > 10 )
				{
					cv::circle( targetFrame, dstLMPoint, 4, colorErr, 2);
					cv::line( targetFrame, dstLMPoint, dstPoint, colorErr, 1, 0, 0 );
				}
				else
					cv::circle( targetFrame, dstLMPoint, 4, color2, 2);
			}
			cv::imshow( "TargetFrame", targetFrame);
			cv::imshow( "SourceFrame", sourceFrame);
			cv::waitKey(10);
//			while( 	cv::waitKey(10) == -1 );
		}


		Eigen::Matrix<double, 6, 1> deltaS		= Eigen::Matrix<double, 6, 1>::Zero();
		Eigen::Matrix<double, 6, 6> surfeld2f	= Eigen::Matrix<double, 6, 6>::Zero();
		Eigen::Matrix<double, 6, 1> surfeldf	= Eigen::Matrix<double, 6, 1>::Zero();

//		double coarsefactor = 1.0 - (double)iter / (double)maxIterations;
//		double featureAssocMahalDist = params_.pointFeatureMatchingFineImagePosMahalDist_ + coarsefactor * (params_.pointFeatureMatchingCoarseImagePosMahalDist_ - params_.pointFeatureMatchingFineImagePosMahalDist_);

//		int numMatches = 0;
//		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
//		{
//			if (it->match == 0)
//				continue;
//
//			numMatches++;
//		}
//
//		double unmatchedFraction = 1.0 - (double)numMatches / (double)featureAssociations_.size();
//		double featureAssocMahalDist = params_.pointFeatureMatchingFineImagePosMahalDist_ + unmatchedFraction * (params_.pointFeatureMatchingCoarseImagePosMahalDist_ - params_.pointFeatureMatchingFineImagePosMahalDist_);
//
//		std::cout << iter << " " << featureAssocMahalDist << "\n";

		if (reevaluateGradient) {

			if (reassociate) {
				target_->clearAssociations();
			}

			stopwatch.reset();

			double last_error = 0.0;
			compactH.setZero();
			rightSide.setZero();
			poseOFdf.setZero();
			if (params_.registerFeatures_)
			{
				stopwatch.reset();
				last_error = preparePointFeatureDerivatives(x, q.w(), featureAssocMahalDist);

				int numMatches = 0;
				for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
				{
					if (it->match == 0)
						continue;

					numMatches++;

					Eigen::Matrix<double, 6, 3> tmp = it->Hpl.transpose() * it->Hll.inverse();
					compactH += it->Hpp - tmp * it->Hpl;
					rightSide += tmp * it->bl - it->bp;
					poseOFdf += -it->bp;

				}

				last_error *= params_.pointFeatureWeight_;
				compactH *= params_.pointFeatureWeight_;
				rightSide *= params_.pointFeatureWeight_;
				poseOFdf *= params_.pointFeatureWeight_;

				if( params_.debugFeatures_ )
					std::cout << "matched " << numMatches << "/" << featureAssociations_.size() << " associated pointfeatures\n";

				if( params_.debugFeatures_ )
					std::cout << "feature preparation took: " << stopwatch.getTime() << "\n";
			}

			if (params_.registerSurfels_)
			{
				float searchDistFactor = 2.f;
				float maxSearchDist = 2.f * maxResolution;

				stopwatch.reset();

				surfelAssociations.clear();
				associateMapsBreadthFirstParallel( surfelAssociations, *source_, *target_, targetSamplingMap_, transform, minResolution, maxResolution, searchDistFactor, maxSearchDist, false);

				double surfelError = 0;
				double deltat = stopwatch.getTime();
				// std::cout << "assoc took: " << deltat << "\n";

				interpolate_neighbors_ = true;

				if (!registrationSurfelErrorFunctionWithFirstAndSecondDerivativeLM( x, lastWSign_, surfelError, surfeldf, surfeld2f, surfelAssociations))
				{
					std::cout << "Surfelregistration failed ------\n";
				}
				else
				{
					compactH += surfeld2f;
					rightSide += surfeldf;
					last_error += surfelError;
					poseOFdf += surfeldf;
				}
			}

		}

		if (!retVal) {
			std::cout << "registration failed\n";
			return false;
		}

		Eigen::Matrix<double, 6, 1> delta_x	= Eigen::Matrix<double, 6, 1>::Zero();
		Eigen::Matrix<double, 6, 6> d2f		= compactH;

		// delta_x für feature
		if (fabsf(d2f.determinant()) > std::numeric_limits<double>::epsilon())
		{
			delta_x = d2f.inverse() * rightSide;
		}
		else {
			std::cout << "Det(d2f) =\t" << d2f.determinant() << "\n";
		}


		if (delta_x.norm() < min_delta) {
			break;
		}

		double qx = x( 3 );
		double qy = x( 4 );
		double qz = x( 5 );
		double qw = lastWSign_*sqrt(1.0-qx*qx-qy*qy-qz*qz);

		Eigen::Matrix4d currentTransform;
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


//		FeatureAssociationList FALcopy;
//		FALcopy.reserve( featureAssociations_.size() );
//		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
//		{
//			MultiResolutionSurfelRegistration::FeatureAssociation assoc( it->src_idx_, it->dst_idx_ );
//
//			if (it->match == 0) {
//				assoc.landmark_pos = it->landmark_pos;
//				FALcopy.push_back(assoc);
//				continue;
//			}
//
//			Eigen::Vector3d deltaLM = (it->Hll + mu * Eigen::Matrix3d::Identity()).inverse() * (-it->bl - it->Hpl * deltaS);
//
//			assoc.landmark_pos = phi(phiInv(it->landmark_pos) + deltaLM);
//
//			FALcopy.push_back(assoc);
//		}

		for (MRCSRFAL::iterator it = featureAssociations_.begin(); it != featureAssociations_.end(); ++it)
		{

			it->tmp_landmark_pos = it->landmark_pos;

			if (it->match == 0) {
				continue;
			}

			Eigen::Vector3d deltaLM = (it->Hll).inverse() * (-it->bl - it->Hpl * deltaS);

			it->landmark_pos = phi(phiInv(it->landmark_pos) + deltaLM);

		}

		stopwatch.reset();

		if( params_.debugFeatures_ )
			std::cout << "feature error function eval took: " << stopwatch.getTime() << "\n";

		x = x_new;
		lastWSign_ = newWSign;

		qx = x(3);
		qy = x(4);
		qz = x(5);
		qw = lastWSign_ * sqrt(1.0 - qx * qx - qy * qy - qz * qz);

		if (boost::math::isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f) {
			return false;
		}

		transform.setIdentity();
		transform.block<3, 3>(0, 0) = Eigen::Matrix3d(
				Eigen::Quaterniond(qw, qx, qy, qz));
		transform(0, 3) = x(0);
		transform(1, 3) = x(1);
		transform(2, 3) = x(2);

		iter++;
	}
	return true;
}


bool MultiResolutionSurfelRegistration::estimateTransformation( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGB >::Ptr correspondencesTargetPoints, int gradientIterations, int coarseToFineIterations, int fineIterations ) {

	processTimeWatch.reset();

	params_.startResolution_ = startResolution;
	params_.stopResolution_ = stopResolution;

	source_ = &source;
	target_ = &target;

	correspondences_source_points_ = correspondencesSourcePoints;
	correspondences_target_points_ = correspondencesTargetPoints;

	// estimate transformation from maps
	target.clearAssociations();

	targetSamplingMap_ = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	bool retVal = true;
	if( params_.registerFeatures_ ) {

		pcl::StopWatch stopwatch;
		stopwatch.reset();
		associatePointFeatures();
		if( params_.debugFeatures_ )
			std::cout << "pf association took: " << stopwatch.getTime() << "\n";

		double mu = -1.0;
		double nu = 2.0;
		bool retVal = estimateTransformationLevenbergMarquardtPF( transform, gradientIterations + coarseToFineIterations, params_.pointFeatureMatchingCoarseImagePosMahalDist_, 1e-2, true, mu, nu );
//		retVal = estimateTransformationLevenbergMarquardtPF( transform, gradientIterations + coarseToFineIterations, params_.pointFeatureMatchingFineImagePosMahalDist_, 1e-4, false, mu, nu );
		if( retVal ) {
//			retVal = estimateTransformationGaussNewtonPF( transform, fineIterations, params_.pointFeatureMatchingFineImagePosMahalDist_, 1e-4, true, mu, nu );
			retVal = estimateTransformationLevenbergMarquardtPF( transform, fineIterations, params_.pointFeatureMatchingFineImagePosMahalDist_, 1e-4, true, mu, nu );
		}

		if( !retVal )
			std::cout << "registration failed\n";

	}
	else {

		if( gradientIterations > 0 )
			retVal = estimateTransformationLevenbergMarquardt( transform, gradientIterations );//, NULL, false, true );

		if( !retVal )
			std::cout << "levenberg marquardt failed\n";

		Eigen::Matrix4d transformGradient = transform;

		if( retVal ) {

			bool retVal2 = estimateTransformationNewton( transform, coarseToFineIterations, fineIterations );
			if( !retVal2 ) {
				std::cout << "newton failed\n";
				transform = transformGradient;

				if( gradientIterations == 0 )
					retVal = false;
			}

		}

	}

	return retVal;

}


// exponents: exp( -0.5 * D * (k sigma^2) / sigma^2 )
// D=1:
// k=1: -0.5, k=2: -2, k=3:  -4.5, k=4:  -8
// D=3:
// k=1: -1.5, k=2: -6, k=3: -13.5, k=4: -24


class MatchLogLikelihoodFunctor {
public:

	MatchLogLikelihoodFunctor() {}
	MatchLogLikelihoodFunctor( MultiResolutionSurfelRegistration::NodeLogLikelihoodList* nodes, MultiResolutionSurfelRegistration::Params params, MultiResolutionSurfelMap* source, MultiResolutionSurfelMap* target, const Eigen::Matrix4d& transform, const Eigen::Matrix4d& lastTransform, const Eigen::Matrix< double, 6, 6 >& pcov, double delta_t, bool addPoseCov ) {
		nodes_ = nodes;
		params_ = params;
		source_ = source;
		target_ = target;
		transform_ = transform;
		add_pose_cov_ = addPoseCov;
		pcov_ = pcov;

		delta_t_ = delta_t;
		this->lastTransform = lastTransform;
		lastRotation = lastTransform.block<3,3>(0,0);

		normalStd = 0.125*M_PI;
		normalMinLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) - 4.5; // k=3
		normalMinLogLikelihoodSeenThrough = -0.5 * log( 2.0 * M_PI * normalStd ) - 8.0; // k=4

		targetToSourceTransform = transform_;
		currentRotation = Eigen::Matrix3d( targetToSourceTransform.block<3,3>(0,0) );
		currentRotationT = currentRotation.transpose();
		currentTranslation = Eigen::Vector3d( targetToSourceTransform.block<3,1>(0,3) );



		precomputeCovAdd( target );

	}

	~MatchLogLikelihoodFunctor() {}


	void precomputeCovAdd( MultiResolutionSurfelMap* target ) {


		for( unsigned int d = 0; d <= target->octree_->max_depth_; d++ ) {

			const double processResolution = target->octree_->resolutions_[d];

			Eigen::Matrix3d cov_add;
			cov_add.setZero();
			if( params_.add_smooth_pos_covariance_ ) {
				cov_add.setIdentity();
				cov_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
			}

			cov_add_[d] = cov_add;

		}

	}


	Eigen::Matrix< double, 3, 6 > diff_jac_for_pose( const Eigen::Vector3d& position ) const {

		// Jacobian of diff mu_tgt - T(x) mu_src is: - dT(x) mu_src in R^(3 x 6)

		Eigen::Matrix< double, 3, 6 > jac = Eigen::Matrix< double, 3, 6 >::Zero();

		jac.block<3,1>(0,0) = (lastTransform * Eigen::Vector4d::Unit( 0 )).block<3,1>(0,0);
		jac.block<3,1>(0,1) = (lastTransform * Eigen::Vector4d::Unit( 1 )).block<3,1>(0,0);
		jac.block<3,1>(0,2) = (lastTransform * Eigen::Vector4d::Unit( 2 )).block<3,1>(0,0);
		jac.block<3,1>(0,3) = (lastTransform * Eigen::Vector4d( 0, -position(2), position(1), 0 )).block<3,1>(0,0);
		jac.block<3,1>(0,4) = (lastTransform * Eigen::Vector4d( position(2), -position(0), 0, 0 )).block<3,1>(0,0);
		jac.block<3,1>(0,5) = (lastTransform * Eigen::Vector4d( -position(1), position(0), 0, 0 )).block<3,1>(0,0);

		jac *= -delta_t_;

		return jac;



//		Eigen::Matrix< double, 3, 6 > jac = Eigen::Matrix< double, 3, 6 >::Zero();
//
//		Eigen::Vector3d rotpos = currentRotation * position;
//
//		jac( 0, 0 ) = -1;
//		jac( 1, 1 ) = -1;
//		jac( 2, 2 ) = -1;
//
//		jac( 1, 3 ) = +rotpos(2);//(transform_.block<1,3>(2,0) * position)(0);
//		jac( 2, 3 ) = -rotpos(1);//(transform_.block<1,3>(1,0) * position)(0);
//
//		jac( 0, 4 ) = -rotpos(2);//(transform_.block<1,3>(2,0) * position)(0);
//		jac( 2, 4 ) = +rotpos(0);//(transform_.block<1,3>(0,0) * position)(0);
//
//		jac( 0, 5 ) = +rotpos(1);//(transform_.block<1,3>(1,0) * position)(0);
//		jac( 1, 5 ) = -rotpos(0);//(transform_.block<1,3>(0,0) * position)(0);
//
//		return jac;

	}


//	Eigen::Matrix< double, 1, 3 > diff_normals_jac_for_angvel( const Eigen::Vector3d& normal1, const Eigen::Vector3d& normal2 ) const {
//
//		Eigen::Matrix< double, 1, 3 > jac = Eigen::Matrix< double, 1, 3 >::Zero();
//
//		double diff_n = normal1.dot( currentRotation * normal2 );
//
////		std::cout << delta_t_ << " " << sqrt( 1.0 - diff_n ) << "\n";
//
//		Eigen::Matrix3d rotSkewedNormal = Eigen::Matrix3d::Zero();
//		rotSkewedNormal(0,1) = -normal2(2);
//		rotSkewedNormal(0,2) = +normal2(1);
//		rotSkewedNormal(1,0) = +normal2(2);
//		rotSkewedNormal(1,1) = -normal2(0);
//		rotSkewedNormal(2,0) = -normal2(1);
//		rotSkewedNormal(2,1) = +normal2(0);
//
//		jac = normal1.transpose() * lastRotation * rotSkewedNormal;
//
////		jac(0,0) = normal1.dot( lastRotation * Eigen::Vector3d( 0, -normal2(2), normal2(1) ) );
////		jac(0,1) = normal1.dot( lastRotation * Eigen::Vector3d( normal2(2), -normal2(0), 0 ) );
////		jac(0,2) = normal1.dot( lastRotation * Eigen::Vector3d( -normal2(1), normal2(0), 0 ) );
//
////		if( diff_n < 0.001 )
////			diff_n = 0.001;
//
//		jac *= -delta_t_ / sqrt( 1.0 - diff_n );
//
//		return jac;
//
//
//	}

	double normalCovFromPoseCov( const Eigen::Vector3d& normal1, const Eigen::Vector3d& normal2, const Eigen::Matrix3d& poseCov ) const {

		Eigen::Matrix< double, 1, 3 > jac = Eigen::Matrix< double, 1, 3 >::Zero();

		double diff_n = normal1.dot( currentRotation * normal2 );

		jac(0,0) = normal1.dot( lastRotation * Eigen::Vector3d( 0, -normal2(2), normal2(1) ) );
		jac(0,1) = normal1.dot( lastRotation * Eigen::Vector3d( normal2(2), -normal2(0), 0 ) );
		jac(0,2) = normal1.dot( lastRotation * Eigen::Vector3d( -normal2(1), normal2(0), 0 ) );

//		Eigen::Matrix3d rotSkewedNormal = Eigen::Matrix3d::Zero();
//		rotSkewedNormal(0,1) = -normal2(2);
//		rotSkewedNormal(0,2) = +normal2(1);
//		rotSkewedNormal(1,0) = +normal2(2);
//		rotSkewedNormal(1,1) = -normal2(0);
//		rotSkewedNormal(2,0) = -normal2(1);
//		rotSkewedNormal(2,1) = +normal2(0);
//
//		jac = normal1.transpose() * lastRotation * rotSkewedNormal;

		if( diff_n > 0.999 )
			diff_n = 0.999;

//		jac *= -delta_t_ / sqrt( 1.0 - diff_n );

		return delta_t_*delta_t_ / (1.0 - diff_n*diff_n) * (jac * poseCov * jac.transpose()).eval()(0);

	}



	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*nodes_)[i]);
	}


	void operator()( MultiResolutionSurfelRegistration::NodeLogLikelihood& node ) const {

		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = node.n_;

		double sumLogLikelihood = 0.0;

		Eigen::Vector4d npos = n->getPosition().cast<double>();
		npos(3) = 1;
		Eigen::Vector4d npos_match_src = targetToSourceTransform * npos;

		// for match log likelihood: query in volume to check the neighborhood for the best matching (discretization issues)
		std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* > nodes;
		nodes.reserve(50);
		const double searchRadius = 2.0 * n->resolution();
		Eigen::Vector4f minPosition, maxPosition;
		minPosition[0] = npos_match_src(0) - searchRadius;
		minPosition[1] = npos_match_src(1) - searchRadius;
		minPosition[2] = npos_match_src(2) - searchRadius;
		maxPosition[0] = npos_match_src(0) + searchRadius;
		maxPosition[1] = npos_match_src(1) + searchRadius;
		maxPosition[2] = npos_match_src(2) + searchRadius;
		source_->octree_->getAllNodesInVolumeOnDepth( nodes, minPosition, maxPosition, n->depth_, false );


		const Eigen::Matrix3d& cov_add = cov_add_[n->depth_];


		// only consider model surfels that are visible from the scene viewpoint under the given transformation

		for( unsigned int i = 4; i <= 4; i++ ) {
//		for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

			node.surfelassocs_[i].loglikelihood_ = 0.0;
			node.surfelassocs_[i].n_dst_ = NULL;
			node.surfelassocs_[i].dst_ = NULL;

			MultiResolutionSurfelMap::Surfel* targetSurfel = &n->value_.surfels_[i];

			if( targetSurfel->num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
				continue;
			}

			// transform surfel mean with current transform and find corresponding node in source for current resolution
			// find corresponding surfel in node via the transformed view direction of the surfel

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = targetSurfel->mean_.block<3,1>(0,0);
			pos(3,0) = 1.f;

			Eigen::Vector4d dir;
			dir.block<3,1>(0,0) = targetSurfel->initial_view_dir_;
			dir(3,0) = 0.f; // pure rotation

			Eigen::Vector4d pos_match_src = targetToSourceTransform * pos;
			Eigen::Vector4d dir_match_src = targetToSourceTransform * dir;

			// precalculate log likelihood when surfel is not matched in the scene
			Eigen::Matrix3d cov2 = targetSurfel->cov_.block<3,3>(0,0);
			cov2 += cov_add;

			Eigen::Matrix3d cov2_RT = cov2 * currentRotationT;
			Eigen::Matrix3d cov2_rotated = (currentRotation * cov2_RT).eval();

			if( add_pose_cov_ ) {
				const Eigen::Vector3d pos3 = pos.block<3,1>(0,0);
				const Eigen::Matrix< double, 3, 6 > jac = diff_jac_for_pose( pos3 );
				cov2_rotated += jac * pcov_ * jac.transpose();
			}

			double mahaldist = -13.5;
			if( targetSurfel->seenThrough_ )
				mahaldist = -24.0;

			double nomatch_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov2_rotated.determinant() ) + mahaldist;

			if( params_.match_likelihood_use_color_ ) {
				nomatch_loglikelihood += -0.5 * log( 8.0 * M_PI * M_PI * M_PI * (targetSurfel->cov_.block<3,3>(3,3) + 0.0001 * Eigen::Matrix3d::Identity()).determinant() ) + mahaldist;
			}

			if( params_.match_likelihood_use_normals_ ) {
				if( targetSurfel->seenThrough_ )
					nomatch_loglikelihood += normalMinLogLikelihoodSeenThrough;
				else
					nomatch_loglikelihood += normalMinLogLikelihood;
			}

			if( boost::math::isinf(nomatch_loglikelihood) || boost::math::isnan(nomatch_loglikelihood) )
				continue;

			double bestSurfelLogLikelihood = nomatch_loglikelihood;

			node.surfelassocs_[i].loglikelihood_ = nomatch_loglikelihood;


			if( !targetSurfel->seenThrough_ ) {

				for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >::iterator it = nodes.begin(); it != nodes.end(); ++it ) {

					spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n_src = *it;

					// find best matching surfel for the view direction in the scene map
					MultiResolutionSurfelMap::Surfel* bestMatchSurfel = NULL;
					double bestMatchDist = -1.f;
					for( unsigned int k = 0; k < MultiResolutionSurfelMap::NodeValue::num_surfels_; k++ ) {

						const double dist = dir_match_src.block<3,1>(0,0).dot( n_src->value_.surfels_[k].initial_view_dir_ );
						if( dist > bestMatchDist ) {
							bestMatchSurfel = &n_src->value_.surfels_[k];
							bestMatchDist = dist;
						}
					}


					// do only associate on the same resolution
					// no match? use maximum distance log likelihood for this surfel
					if( bestMatchSurfel->num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
						continue;
					}


					Eigen::Vector3d diff_pos = bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0);


					Eigen::Matrix3d cov1 = bestMatchSurfel->cov_.block<3,3>(0,0);
					cov1 += cov_add;

					Eigen::Matrix3d cov = cov1 + cov2_rotated;
					Eigen::Matrix3d invcov = cov.inverse().eval();

					double exponent = -0.5 * diff_pos.dot(invcov * diff_pos);
					if( exponent < -6.0 )
						exponent = -13.5;
					double loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov.determinant() ) + exponent;



					Eigen::Vector3d diff_col = bestMatchSurfel->mean_.block<3,1>(3,0) - targetSurfel->mean_.block<3,1>(3,0);
					if( fabs(diff_col(0)) < params_.luminance_damp_diff_ )
						diff_col(0) = 0;
					if( fabs(diff_col(1)) < params_.color_damp_diff_ )
						diff_col(1) = 0;
					if( fabs(diff_col(2)) < params_.color_damp_diff_ )
						diff_col(2) = 0;

					if( diff_col(0) < 0 )
						diff_col(0) += params_.luminance_damp_diff_;
					if( diff_col(1) < 0 )
						diff_col(1) += params_.color_damp_diff_;
					if( diff_col(2) < 0 )
						diff_col(2) += params_.color_damp_diff_;

					if( diff_col(0) > 0 )
						diff_col(0) -= params_.luminance_damp_diff_;
					if( diff_col(1) > 0 )
						diff_col(1) -= params_.color_damp_diff_;
					if( diff_col(2) > 0 )
						diff_col(2) -= params_.color_damp_diff_;

					if( params_.match_likelihood_use_color_ ) {
						const Eigen::Matrix3d cov_cc = bestMatchSurfel->cov_.block<3,3>(3,3) + targetSurfel->cov_.block<3,3>(3,3) + 0.0001 * Eigen::Matrix3d::Identity();
						const Eigen::Matrix3d invcov_cc = cov_cc.inverse();
						double color_exponent = -0.5 * diff_col.dot( (invcov_cc * diff_col ).eval() );
						if( color_exponent < -6.0 )
							color_exponent = -13.5;

						double color_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov_cc.determinant() ) + color_exponent;
						loglikelihood += color_loglikelihood;
					}

					if( params_.match_likelihood_use_normals_ ) {

						// test: also consider normal orientation in the likelihood!!
						Eigen::Vector4d normal_src;
						normal_src.block<3,1>(0,0) = targetSurfel->normal_;
						normal_src(3,0) = 0.0;
						normal_src = (targetToSourceTransform * normal_src).eval();


						double normalStdMod = normalStd;

						if( add_pose_cov_ ) {
		//					Eigen::Matrix< double, 1, 3 > jac_normals = diff_normals_jac_for_angvel( bestMatchSurfel->normal_, normal_src.block<3,1>(0,0) );
		//					normalStdMod += (jac_normals * pcov_.block<3,3>(3,3) * jac_normals.transpose())(0);
							normalStdMod += normalCovFromPoseCov( bestMatchSurfel->normal_, normal_src.block<3,1>(0,0), pcov_.block<3,3>(3,3) );

		//					std::cout << "normal std added: " << normalStdMod-normalStd << "\n";
						}

						double normalError = acos( normal_src.block<3,1>(0,0).dot( bestMatchSurfel->normal_ ) );
						double normalExponent = -0.5 * normalError * normalError / ( normalStdMod*normalStdMod );
						if( normalExponent < -2.0 )
							normalExponent = -4.5;
						double normalLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStdMod ) + normalExponent;

						loglikelihood += normalLogLikelihood;

					}


					if( boost::math::isinf(nomatch_loglikelihood) || boost::math::isnan( exponent ) ) {
						continue;
					}
					if( boost::math::isinf(nomatch_loglikelihood) || boost::math::isnan(loglikelihood) )
						continue;

					if( loglikelihood > bestSurfelLogLikelihood ) {
						bestSurfelLogLikelihood = loglikelihood;
						node.surfelassocs_[i].loglikelihood_ = loglikelihood;
						node.surfelassocs_[i].n_dst_ = n_src;
						node.surfelassocs_[i].dst_ = bestMatchSurfel;
					}

				}

			}

			sumLogLikelihood += bestSurfelLogLikelihood;
		}

		node.loglikelihood_ = sumLogLikelihood;

	}


	MultiResolutionSurfelRegistration::NodeLogLikelihoodList* nodes_;
	MultiResolutionSurfelRegistration::Params params_;
	MultiResolutionSurfelMap* source_;
	MultiResolutionSurfelMap* target_;
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
	MatchLogLikelihoodKnownAssociationsFunctor( MultiResolutionSurfelRegistration::NodeLogLikelihoodList* associations, MultiResolutionSurfelRegistration::Params params, MultiResolutionSurfelMap* source, MultiResolutionSurfelMap* target, const Eigen::Matrix4d& transform, const Eigen::Matrix4d& lastTransform, const Eigen::Matrix< double, 6, 6 >& pcov, double delta_t, bool addPoseCov ) {
		nodes_ = associations;
		params_ = params;
		source_ = source;
		target_ = target;
		transform_ = transform;
		add_pose_cov_ = addPoseCov;
		pcov_ = pcov;
		delta_t_ = delta_t;
		this->lastTransform = lastTransform;
		lastRotation = lastTransform.block<3,3>(0,0);

		normalStd = 0.125*M_PI;
		normalMinLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) - 4.5;
		normalMinLogLikelihoodSeenThrough = -0.5 * log( 2.0 * M_PI * normalStd ) - 8.0;

		// be careful, the inversion of model and scene surfel comes from the parameters,
		// but due to the propagation of the pose uncertainty, we need to evaluate the
		// likelihood in the right direction, i.e. we need to transform the scene surfel!
		targetToSourceTransform = transform_;
		currentRotation = Eigen::Matrix3d( targetToSourceTransform.block<3,3>(0,0) );
		currentRotationT = currentRotation.transpose();
		currentTranslation = Eigen::Vector3d( targetToSourceTransform.block<3,1>(0,3) );

		precomputeCovAdd( target );

	}

	~MatchLogLikelihoodKnownAssociationsFunctor() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*nodes_)[i]);
	}


	void operator()( MultiResolutionSurfelRegistration::NodeLogLikelihood& node ) const {

		double sumLogLikelihood = 0.0;
		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = node.n_;

		const Eigen::Matrix3d& cov_add = cov_add_[n->depth_];

		for( unsigned int i = 4; i <= 4; i++ ) {
//		for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

			if( node.surfelassocs_[i].dst_ ) {

				MultiResolutionSurfelMap::Surfel* targetSurfel = &n->value_.surfels_[i];

				Eigen::Vector4d pos;
				pos.block<3,1>(0,0) = targetSurfel->mean_.block<3,1>(0,0);
				pos(3,0) = 1.f;

				Eigen::Vector4d pos_match_src = targetToSourceTransform * pos;

				Eigen::Matrix3d cov2 = targetSurfel->cov_.block<3,3>(0,0);
				cov2 += cov_add;

				Eigen::Matrix3d cov2_RT = cov2 * currentRotationT;
				Eigen::Matrix3d cov2_rotated = (currentRotation * cov2_RT).eval();

				if( add_pose_cov_ ) {
					const Eigen::Vector3d pos3 = pos.block<3,1>(0,0);
					const Eigen::Matrix< double, 3, 6 > jac = diff_jac_for_pose( pos3 );
					cov2_rotated += jac * pcov_ * jac.transpose();
				}

				MultiResolutionSurfelMap::Surfel* sceneSurfel = node.surfelassocs_[i].dst_;

				Eigen::Vector3d diff_pos = sceneSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0);

				Eigen::Matrix3d cov1 = sceneSurfel->cov_.block<3,3>(0,0);
				cov1 += cov_add;

				Eigen::Matrix3d cov = cov1 + cov2_rotated;
				Eigen::Matrix3d invcov = cov.inverse().eval();

				double exponent = -0.5 * diff_pos.dot(invcov * diff_pos);
				if( exponent < -6.0 )
					exponent = -13.5;
				double loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov.determinant() ) + exponent;


				Eigen::Vector3d diff_col = sceneSurfel->mean_.block<3,1>(3,0) - targetSurfel->mean_.block<3,1>(3,0);
				if( fabs(diff_col(0)) < params_.luminance_damp_diff_ )
					diff_col(0) = 0;
				if( fabs(diff_col(1)) < params_.color_damp_diff_ )
					diff_col(1) = 0;
				if( fabs(diff_col(2)) < params_.color_damp_diff_ )
					diff_col(2) = 0;

				if( diff_col(0) < 0 )
					diff_col(0) += params_.luminance_damp_diff_;
				if( diff_col(1) < 0 )
					diff_col(1) += params_.color_damp_diff_;
				if( diff_col(2) < 0 )
					diff_col(2) += params_.color_damp_diff_;

				if( diff_col(0) > 0 )
					diff_col(0) -= params_.luminance_damp_diff_;
				if( diff_col(1) > 0 )
					diff_col(1) -= params_.color_damp_diff_;
				if( diff_col(2) > 0 )
					diff_col(2) -= params_.color_damp_diff_;

				if( params_.match_likelihood_use_color_ ) {
					const Eigen::Matrix3d cov_cc = sceneSurfel->cov_.block<3,3>(3,3) + targetSurfel->cov_.block<3,3>(3,3) + 0.0001 * Eigen::Matrix3d::Identity();
					const Eigen::Matrix3d invcov_cc = cov_cc.inverse();
					double color_exponent = -0.5 * diff_col.dot( (invcov_cc * diff_col ).eval() );
					if( color_exponent < -6.0 )
						color_exponent = -13.5;
					double color_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov_cc.determinant() ) + color_exponent;
					loglikelihood += color_loglikelihood;
				}

				if( params_.match_likelihood_use_normals_ ) {

					// test: also consider normal orientation in the likelihood!!
					Eigen::Vector4d normal_src;
					normal_src.block<3,1>(0,0) = targetSurfel->normal_;
					normal_src(3,0) = 0.0;
					normal_src = (targetToSourceTransform * normal_src).eval();


					double normalStdMod = normalStd;

					if( add_pose_cov_ ) {
	//					Eigen::Matrix< double, 1, 3 > jac_normals = diff_normals_jac_for_angvel( bestMatchSurfel->normal_, normal_src.block<3,1>(0,0) );
	//					normalStdMod += (jac_normals * pcov_.block<3,3>(3,3) * jac_normals.transpose())(0);
						normalStdMod += normalCovFromPoseCov( sceneSurfel->normal_, normal_src.block<3,1>(0,0), pcov_.block<3,3>(3,3) );

	//					std::cout << "normal std added: " << normalStdMod-normalStd << "\n";
					}

					double normalError = acos( normal_src.block<3,1>(0,0).dot( sceneSurfel->normal_ ) );
					double normalExponent = -0.5 * normalError * normalError / ( normalStdMod*normalStdMod );
					if( normalExponent < -2.0 )
						normalExponent = -4.5;
					double normalLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStdMod ) + normalExponent;
					loglikelihood += normalLogLikelihood;
				}


				node.surfelassocs_[i].loglikelihood_ = 0.0;

				if( boost::math::isnan( exponent ) ) {
					continue;
				}
				if( boost::math::isnan(loglikelihood) )
					continue;

				node.surfelassocs_[i].loglikelihood_ = loglikelihood;

			}

			sumLogLikelihood += node.surfelassocs_[i].loglikelihood_;

		}

		node.loglikelihood_ = sumLogLikelihood;

	}


};



// assumes model is in the target, and scene in source has a node image
class ShootThroughFunctor {
public:

	ShootThroughFunctor( MultiResolutionSurfelRegistration::NodeLogLikelihoodList* nodes, MultiResolutionSurfelRegistration::Params params, MultiResolutionSurfelMap* source, MultiResolutionSurfelMap* target, const Eigen::Matrix4d& transform ) {
		nodes_ = nodes;
		params_ = params;
		source_ = source;
		target_ = target;
		transform_ = transform;
		transform_inv_ = transform.inverse();
		camera_pos_ = transform_inv_.block<3,1>(0,3);

	}

	~ShootThroughFunctor() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*nodes_)[i]);
	}


	void operator()( MultiResolutionSurfelRegistration::NodeLogLikelihood& node ) const {

		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = node.n_;

		Eigen::Vector3d viewDirection = n->getCenterPosition().cast<double>().block<3,1>(0,0) - camera_pos_;
		const MultiResolutionSurfelMap::Surfel* targetSurfel = n->value_.getSurfel( viewDirection );

//		for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

//			// targetSurfel should be a surfel in the model
//			MultiResolutionSurfelMap::Surfel* targetSurfel = &n->value_.surfels_[i];

			if( targetSurfel->num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
				return;
			}

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = targetSurfel->mean_.block<3,1>(0,0);
			pos(3,0) = 1.f;
			Eigen::Vector4d pos_match_src = transform_ * pos;

			bool seenThrough = pointSeenThrough( pos_match_src.cast<float>(), *source_, 2.0*n->resolution(), true);

//		}
	}

	MultiResolutionSurfelRegistration::NodeLogLikelihoodList* nodes_;
	MultiResolutionSurfelRegistration::Params params_;
	MultiResolutionSurfelMap* source_;
	MultiResolutionSurfelMap* target_;
	Eigen::Matrix4d transform_, transform_inv_;
	Eigen::Vector3d camera_pos_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

};



double MultiResolutionSurfelRegistration::matchLogLikelihood( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, const Eigen::Matrix4d& lastTransform, const Eigen::Matrix< double, 6, 6 >& pcov, double delta_t, bool addPoseCov ) {

	target.clearSeenThroughFlag();

	// check for scene occlusions by the model
	{
		auto sourceSamplingMap = algorithm::downsampleVectorOcTree(*source.octree_, false, source.octree_->max_depth_);

		int maxDepth = source.octree_->max_depth_;

		int countNodes = 0;
		for( int d = maxDepth; d >= 0; d-- ) {
			countNodes += sourceSamplingMap[d].size();
		}

		NodeLogLikelihoodList nodes;
		nodes.reserve( countNodes );

		for( int d = params_.model_visibility_max_depth_; d >= 0; d-- ) {

			for( unsigned int i = 0; i < sourceSamplingMap[d].size(); i++ ) {
				spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = sourceSamplingMap[d][i];
				nodes.push_back( NodeLogLikelihood( n ) );
			}

		}

		ShootThroughFunctor stf( &nodes, params_, &target, &source, transform.inverse() );

		if( params_.parallel_ )
			tbb::parallel_for_each( nodes.begin(), nodes.end(), stf );
		else
			std::for_each( nodes.begin(), nodes.end(), stf );

	}


	targetSamplingMap_ = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	int maxDepth = target.octree_->max_depth_;

	int countNodes = 0;
	for( int d = maxDepth; d >= 0; d-- ) {
		countNodes += targetSamplingMap_[d].size();
	}

	NodeLogLikelihoodList nodes;
	nodes.reserve( countNodes );

	for( int d = maxDepth; d >= 0; d-- ) {

		for( unsigned int i = 0; i < targetSamplingMap_[d].size(); i++ )
//			if( targetSamplingMap_[d][i]->type_ != spatialaggregate::OCTREE_BRANCHING_NODE )
				nodes.push_back( NodeLogLikelihood( targetSamplingMap_[d][i] ) );

	}


	MatchLogLikelihoodFunctor mlf( &nodes, params_, &source, &target, transform, lastTransform, pcov, delta_t, addPoseCov );

	if( params_.parallel_ )
		tbb::parallel_for_each( nodes.begin(), nodes.end(), mlf );
	else
		std::for_each( nodes.begin(), nodes.end(), mlf );

	double sumLogLikelihood = 0.0;
	for( unsigned int i = 0; i < nodes.size(); i++ ) {
		sumLogLikelihood += nodes[i].loglikelihood_;
	}

	lastNodeLogLikelihoodList_ = nodes;

	return sumLogLikelihood;

}


double MultiResolutionSurfelRegistration::matchLogLikelihoodKnownAssociations( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, NodeLogLikelihoodList& associations, const Eigen::Matrix4d& lastTransform, const Eigen::Matrix< double, 6, 6 >& pcov, double delta_t, bool addPoseCov ) {

	MatchLogLikelihoodKnownAssociationsFunctor mlf( &associations, params_, &source, &target, transform, lastTransform, pcov, delta_t, addPoseCov );

	if( params_.parallel_ )
		tbb::parallel_for_each( associations.begin(), associations.end(), mlf );
	else
		std::for_each( associations.begin(), associations.end(), mlf );

	double sumLogLikelihood = 0.0;
	for( unsigned int i = 0; i < associations.size(); i++ ) {
		sumLogLikelihood += associations[i].loglikelihood_;
	}


	return sumLogLikelihood;

}

//
//MultiResolutionSurfelRegistration::NodeLogLikelihoodList MultiResolutionSurfelRegistration::precalculateNomatchLogLikelihoodKnownAssociations( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, SurfelAssociationList& surfelAssociations, const Eigen::Matrix4d& lastTransform, const Eigen::Matrix< double, 6, 6 >& pcov, double delta_t, bool addPoseCov ) {
//
//	double sumLogLikelihood = 0.0;
//
//	targetSamplingMap_ = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);
//
//	int maxDepth = target.octree_->max_depth_;
//
//	int countNodes = 0;
//	for( int d = maxDepth; d >= 0; d-- ) {
//		countNodes += targetSamplingMap_[d].size();
//	}
//
//	NodeLogLikelihoodList nodes;
//	nodes.reserve( countNodes );
//
//	for( int d = maxDepth; d >= 0; d-- ) {
//
//		for( unsigned int i = 0; i < targetSamplingMap_[d].size(); i++ ) {
//			nodes.push_back( NodeLogLikelihood( targetSamplingMap_[d][i] ) );
//		}
//
//	}
//
//
//	NoMatchLogLikelihoodKnownAssociationsFunctor nmlf( &nodes, params_, &source, &target, transform, lastTransform, pcov, delta_t, addPoseCov );
//
//	if( params_.parallel_ )
//		tbb::parallel_for_each( nodes.begin(), nodes.end(), nmlf );
//	else
//		std::for_each( nodes.begin(), nodes.end(), nmlf );
//
//	return nodes;
//
//}
//
//
//
//void MultiResolutionSurfelRegistration::nomatchLogLikelihoodKnownAssociationsResetAssocs( const MultiResolutionSurfelRegistration::NodeLogLikelihoodList& nodes ) {
//
//	for( unsigned int i = 0; i < nodes.size(); i++ ) {
//		nodes[i].n_->value_.associated_ = 0;
//	}
//
//}
//
//
//double MultiResolutionSurfelRegistration::nomatchLogLikelihoodKnownAssociationsPreCalc( const MultiResolutionSurfelRegistration::NodeLogLikelihoodList& nodes ) {
//
//	double sumLogLikelihood = 0.0;
//
//	for( unsigned int i = 0; i < nodes.size(); i++ ) {
//		if( nodes[i].n_->value_.associated_ != 2 )
//			sumLogLikelihood += nodes[i].loglikelihood_;
//	}
//
//	return sumLogLikelihood;
//
//}
//
//
//double MultiResolutionSurfelRegistration::nomatchLogLikelihoodKnownAssociations( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, SurfelAssociationList& surfelAssociations, const Eigen::Matrix4d& lastTransform, const Eigen::Matrix< double, 6, 6 >& pcov, double delta_t, bool addPoseCov ) {
//
//	double sumLogLikelihood = 0.0;
//
//	targetSamplingMap_ = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);
//
//	int maxDepth = target.octree_->max_depth_;
//
//	int countNodes = 0;
//	for( int d = maxDepth; d >= 0; d-- ) {
//		countNodes += targetSamplingMap_[d].size();
//	}
//
//	NodeLogLikelihoodList nodes;
//	nodes.reserve( countNodes );
//
//	for( int d = maxDepth; d >= 0; d-- ) {
//
//		for( unsigned int i = 0; i < targetSamplingMap_[d].size(); i++ ) {
//
//			// MatchLogLikelihoodKnownAssociationsFunctor sets associated flag to 2!
//			if( targetSamplingMap_[d][i]->value_.associated_ != 2 )
//				nodes.push_back( NodeLogLikelihood( targetSamplingMap_[d][i] ) );
//		}
//
//	}
//
//	NoMatchLogLikelihoodKnownAssociationsFunctor nmlf( &nodes, params_, &source, &target, transform, lastTransform, pcov, delta_t, addPoseCov );
//
//	if( params_.parallel_ )
//		tbb::parallel_for_each( nodes.begin(), nodes.end(), nmlf );
//	else
//		std::for_each( nodes.begin(), nodes.end(), nmlf );
//
//	for( unsigned int i = 0; i < nodes.size(); i++ ) {
//		sumLogLikelihood += nodes[i].loglikelihood_;
//	}
//
//
//	return sumLogLikelihood;
//
//}


class SelfMatchLogLikelihoodFunctor {
public:
	SelfMatchLogLikelihoodFunctor( MultiResolutionSurfelRegistration::NodeLogLikelihoodList* nodes, MultiResolutionSurfelRegistration::Params params, MultiResolutionSurfelMap* target ) {
		nodes_ = nodes;
		params_ = params;
		target_ = target;

		normalStd = 0.125*M_PI;
		normalMinLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) - 8.0;
	}

	~SelfMatchLogLikelihoodFunctor() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*nodes_)[i]);
	}


	void operator()( MultiResolutionSurfelRegistration::NodeLogLikelihood& node ) const {

		spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = node.n_;

		double sumLogLikelihood = 0.0;

		const float processResolution = n->resolution();

		Eigen::Matrix3d cov_add;
		cov_add.setZero();
		if( params_.add_smooth_pos_covariance_ ) {
			cov_add.setIdentity();
			cov_add *= params_.smooth_surface_cov_factor_ * processResolution*processResolution;
		}


		// only consider model surfels that are visible from the scene viewpoint under the given transformation

		for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

			MultiResolutionSurfelMap::Surfel* modelSurfel = &n->value_.surfels_[i];

			if( modelSurfel->num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
				continue;
			}

			// precalculate log likelihood when surfel is not matched in the scene
			Eigen::Matrix3d cov2 = modelSurfel->cov_.block<3,3>(0,0);
			cov2 += cov_add;

			double nomatch_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov2.determinant() ) - 24.0;

			if( params_.match_likelihood_use_color_ ) {
				nomatch_loglikelihood += -0.5 * log( 8.0 * M_PI * M_PI * M_PI * modelSurfel->cov_.block<3,3>(3,3).determinant() ) - 24.0;
			}

			nomatch_loglikelihood += normalMinLogLikelihood;

			if( boost::math::isinf(nomatch_loglikelihood) || boost::math::isnan(nomatch_loglikelihood) )
				continue;

			double bestSurfelLogLikelihood = nomatch_loglikelihood;

			Eigen::Matrix3d cov = 2.0*cov2;
//			Eigen::Matrix3d invcov = cov.inverse().eval();

			double loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov.determinant() );

			if( params_.match_likelihood_use_color_ ) {
				const Eigen::Matrix3d cov_cc = 2.0 * modelSurfel->cov_.block<3,3>(3,3) + 0.01 * Eigen::Matrix3d::Identity();
				double color_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov_cc.determinant() );
				loglikelihood += color_loglikelihood;
			}

			double normalLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd );


			if( boost::math::isinf(nomatch_loglikelihood) || boost::math::isnan(loglikelihood) )
				continue;

			bestSurfelLogLikelihood = std::max( bestSurfelLogLikelihood, loglikelihood + normalLogLikelihood );


			sumLogLikelihood += bestSurfelLogLikelihood;

		}

		node.loglikelihood_ = sumLogLikelihood;

	}


	MultiResolutionSurfelRegistration::NodeLogLikelihoodList* nodes_;
	MultiResolutionSurfelRegistration::Params params_;
	MultiResolutionSurfelMap* target_;

	double normalStd;
	double normalMinLogLikelihood;

};


double MultiResolutionSurfelRegistration::selfMatchLogLikelihood( MultiResolutionSurfelMap& target ) {

	targetSamplingMap_ = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	int maxDepth = target.octree_->max_depth_;

	int countNodes = 0;
	for( int d = maxDepth; d >= 0; d-- ) {
		countNodes += targetSamplingMap_[d].size();
	}

	NodeLogLikelihoodList nodes;
	nodes.reserve( countNodes );

	for( int d = maxDepth; d >= 0; d-- ) {

		for( unsigned int i = 0; i < targetSamplingMap_[d].size(); i++ )
			nodes.push_back( NodeLogLikelihood( targetSamplingMap_[d][i] ) );

	}


	SelfMatchLogLikelihoodFunctor mlf( &nodes, params_, &target );

	if( params_.parallel_ )
		tbb::parallel_for_each( nodes.begin(), nodes.end(), mlf );
	else
		std::for_each( nodes.begin(), nodes.end(), mlf );

	double sumLogLikelihood = 0.0;
	for( unsigned int i = 0; i < nodes.size(); i++ ) {
		sumLogLikelihood += nodes[i].loglikelihood_;
	}

	return sumLogLikelihood;

}


bool MultiResolutionSurfelRegistration::estimatePoseCovariance( Eigen::Matrix< double, 6, 6 >& poseCov, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution ) {

	target.clearAssociations();

	float minResolution = std::min( startResolution, stopResolution );
	float maxResolution = std::max( startResolution, stopResolution );

	algorithm::OcTreeSamplingVectorMap<float, MultiResolutionSurfelMap::NodeValue> targetSamplingMap = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	double sumWeight = 0.0;

	Eigen::Quaterniond q( transform.block<3,3>(0,0) );

	const double tx = transform(0,3);
	const double ty = transform(1,3);
	const double tz = transform(2,3);
	const double qx = q.x();
	const double qy = q.y();
	const double qz = q.z();
	const double qw = q.w();


	MultiResolutionSurfelRegistration::SurfelAssociationList surfelAssociations;
	associateMapsBreadthFirstParallel( surfelAssociations, source, target, targetSamplingMap, transform, 0.99f*minResolution, 1.01f*maxResolution, 2.f, 2.f*maxResolution, false );

	MultiResolutionSurfelRegistration::Params params = params_;
//	params.interpolation_cov_factor_ = 1.0;
	GradientFunctor gf( &surfelAssociations, params, tx, ty, tz, qx, qy, qz, qw, false, true, true, true );

	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	Eigen::Matrix< double, 6, 6 > d2f, JSzJ;
	d2f.setZero();
	JSzJ.setZero();

	for( MultiResolutionSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;

//		d2f += it->d2f;
//		JSzJ += it->weight * it->JSzJ;
		d2f += it->weight * it->d2f;
		JSzJ += it->weight * it->weight * it->JSzJ;
		sumWeight += it->weight;

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
	else {
		d2f /= sumWeight;
		JSzJ /= sumWeight * sumWeight;
	}

	poseCov.setZero();

	if( fabsf(d2f.determinant()) < 1e-8 ) {
		poseCov.setIdentity();
		return false;
	}

	poseCov = d2f.inverse() * JSzJ * d2f.inverse();

	if( params_.use_prior_pose_ ) {
		poseCov = (poseCov.inverse().eval() + params_.prior_pose_invcov_).inverse().eval();
	}

	return true;


}


bool MultiResolutionSurfelRegistration::estimatePoseCovarianceLM( Eigen::Matrix< double, 6, 6 >& poseCov, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, SurfelAssociationList* surfelAssociationsArg, bool knownAssociations ) {

	MultiResolutionSurfelRegistration::SurfelAssociationList surfelAssociations;

	if( !knownAssociations ) {
		target.clearAssociations();

		float minResolution = std::min( startResolution, stopResolution );
		float maxResolution = std::max( startResolution, stopResolution );

		algorithm::OcTreeSamplingVectorMap<float, MultiResolutionSurfelMap::NodeValue> targetSamplingMap = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

		associateMapsBreadthFirstParallel( surfelAssociations, source, target, targetSamplingMap, transform, 0.99f*minResolution, 1.01f*maxResolution, 2.f, 2.f*maxResolution, false );

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

	MultiResolutionSurfelRegistration::Params params = params_;
	params.interpolation_cov_factor_ = 1.0;
	params.add_smooth_pos_covariance_ = false;
	GradientFunctorLM gf( &surfelAssociations, params, tx, ty, tz, qx, qy, qz, qw, true, true );

	if( params_.parallel_ )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	Eigen::Matrix< double, 6, 6 > d2f, JSzJ;
	d2f.setZero();
	JSzJ.setZero();

	for( MultiResolutionSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;

		d2f += it->dh_dx.transpose() * it->W * it->dh_dx;
		JSzJ += it->JSzJ;
		sumWeight += 1.0;

//		d2f += it->weight * it->dh_dx.transpose() * it->W * it->dh_dx;
//		JSzJ += it->weight * it->weight * it->JSzJ;
//		sumWeight += it->weight;

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
	else {
//		d2f /= sumWeight;
//		JSzJ /= sumWeight * sumWeight;
	}

	poseCov.setZero();

	if( fabsf(d2f.determinant()) < 1e-8 ) {
		poseCov.setIdentity();
		return false;
	}

	poseCov = d2f.inverse() * JSzJ * d2f.inverse();

	if( params_.use_prior_pose_ ) {
		poseCov = (poseCov.inverse().eval() + params_.prior_pose_invcov_).inverse().eval();
	}

	return true;


}



void MultiResolutionSurfelRegistration::improvedProposalMatchLogLikelihoodKnownAssociations( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
                                                                                             Geometry::PoseAndVelocity & pose, Geometry::PoseAndVelocity & poseOut, double & likelihood, NodeLogLikelihoodList& associations, bool add_pos_cov,
                                                                                             pcl::PointCloud<pcl::PointXYZRGB>* modelCloud, pcl::PointCloud<pcl::PointXYZRGB>* sceneCloud, const int minDepth, const int maxDepth ) {

    Eigen::Matrix<double, 6, 6> pcov = pose.delta_t_ * pose.poseWithCovariance_.covariance_;
    Eigen::Matrix4d meanTransform ( pose.poseWithCovariance_.mean_.asMatrix4d() );
    Eigen::Matrix4d lastTransform = pose.prevPose_.asMatrix4d();

    double logLikelihood = 0.0;

    logLikelihood = matchLogLikelihoodKnownAssociations( source, target, meanTransform, associations, lastTransform, pcov, pose.delta_t_, add_pos_cov );

    assert( logLikelihood != 0.0 );
    likelihood  = logLikelihood;
}




void MultiResolutionSurfelRegistration::improvedProposalMatchLogLikelihood( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
                                                                            Geometry::PoseAndVelocity & pose, Geometry::PoseAndVelocity & poseOut, double& likelihood, const int numRegistrationStepsCoarse, const int numRegistrationStepsFine, bool add_pose_cov,
                                                                            pcl::PointCloud<pcl::PointXYZRGB>* modelCloud, pcl::PointCloud<pcl::PointXYZRGB>* sceneCloud, const int minDepth, const int maxDepth ) {

    Eigen::Matrix<double, 6, 6> pcov = pose.delta_t_ * pose.poseWithCovariance_.covariance_;
    Eigen::Matrix4d transform ( pose.poseWithCovariance_.mean_.asMatrix4d() );
    Eigen::Matrix4d meanTransform ( pose.poseWithCovariance_.mean_.asMatrix4d() );
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;

    double logLikelihood = 0.0;

    SurfelAssociationList surfelAssociations;
    source_ = &source;
    target_ = &target;
    params_.startResolution_ = target.min_resolution_;
    params_.stopResolution_ = 16.f * target.min_resolution_;
    correspondences_source_points_ = corrSrc;
    correspondences_target_points_ = corrTgt;
    targetSamplingMap_ = target_->samplingMap_;
    bool success = estimateTransformationLevenbergMarquardt( transform, numRegistrationStepsCoarse, &surfelAssociations, false, false );

//    bool useFeatures = params_.use_features_;
//    params_.use_features_ = false;
    success = estimateTransformationLevenbergMarquardt( transform, numRegistrationStepsFine, &surfelAssociations, false, true );
//    params_.use_features_ = useFeatures;

    logLikelihood = matchLogLikelihood( source, target, meanTransform, pose.prevPose_.asMatrix4d(), pcov, pose.delta_t_, add_pose_cov );

    if ( success ) {

        poseOut.pose_ = Geometry::Pose( transform );

        Eigen::Matrix< double, 6, 6 > regPoseCov = Eigen::Matrix< double, 6, 6 >::Identity();
        estimatePoseCovarianceLM( regPoseCov, source, target, transform, params_.startResolution_, params_.stopResolution_, &surfelAssociations, true );

//        regPoseCov *= 0.001;

        poseOut.poseWithCovariance_ = Geometry::PoseWithCovariance( poseOut.pose_, regPoseCov );
        poseOut.velocity_ = pose.velocity_;
        poseOut.prevPose_ = pose.prevPose_;

    } else {
        poseOut = pose;
    }

    assert( logLikelihood != 0.0 );
    likelihood  = logLikelihood;

}



void MultiResolutionSurfelRegistration::getAssociations( SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
                                                         const Geometry::Pose & pose ) {

    Eigen::Matrix4d transform = pose.asMatrix4d();

    bool useFeatures = false;//params_.use_features_;
    float minResolution = std::min( source.min_resolution_, target.min_resolution_ );
    float maxResolution = 16.0 * minResolution;

    float searchDistFactor = 2.f;
    float maxSearchDist = 2.f*maxResolution;

    target.clearAssociations();
    surfelAssociations.clear();

    associateMapsBreadthFirstParallel( surfelAssociations, source, target, target.samplingMap_, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, useFeatures );

}



bool MultiResolutionSurfelRegistration::registerPose( SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
                                                                       Geometry::PoseWithCovariance & pose, Geometry::PoseWithCovariance & poseOut,
                                                                       pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrSrc,
                                                                       pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrTgt,
                                                                       const int numRegistrationStepsCoarse, const int numRegistrationStepsFine,
                                                                       bool regularizeRegistration, const Eigen::Matrix6d& registrationPriorPoseCov ) {

    Eigen::Matrix4d transform = pose.mean_.asMatrix4d();

    SurfelAssociationList associations = surfelAssociations;

    bool useFeatures = params_.use_features_;
    float minResolution = std::min( source.min_resolution_, target.min_resolution_ );
    float maxResolution = 16.0 * minResolution;

    float searchDistFactor = 2.f;
    float maxSearchDist = 2.f*maxResolution;

    source_ = &source;
    target_ = &target;
    params_.startResolution_ = maxResolution;
    params_.stopResolution_ = minResolution;
    correspondences_source_points_ = corrSrc;
    correspondences_target_points_ = corrTgt;
    targetSamplingMap_ = target_->samplingMap_;

    if( regularizeRegistration ) {
		setPriorPose( true, pose.mean_.asVector().block<6,1>(0,0), registrationPriorPoseCov );
    }
    else
        setPriorPoseEnabled( false );

    params_.prior_pose_invcov_ /= params_.interpolation_cov_factor_;
    estimateTransformationLevenbergMarquardt( transform, numRegistrationStepsCoarse, &associations, false, false );
    params_.prior_pose_invcov_ *= params_.interpolation_cov_factor_;

//    setPriorPoseEnabled( false );

//    params_.use_features_ = false;
    bool success = estimateTransformationLevenbergMarquardt( transform, numRegistrationStepsFine, &associations, true, true );
//    params_.use_features_ = useFeatures;


    if ( success ) {
        poseOut.mean_ = Geometry::Pose( transform );
        Eigen::Matrix6d covariance;
        bool cov_success = estimatePoseCovarianceLM( covariance, source, target, transform, maxResolution, minResolution, &associations, true );
        if ( cov_success ) {
            poseOut.covariance_ = covariance;
            surfelAssociations = associations;
        } else {
            LOG_STREAM("Estimate pose covariance failed");
            poseOut.covariance_ = pose.covariance_;
        }
    } else {
        poseOut = pose;
        LOG_STREAM( "Registration failed!" );
    }

    return success;
}


bool MultiResolutionSurfelRegistration::registerPoseKnownAssociations( SurfelAssociationList& surfelAssociations, MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
                                                                       Geometry::PoseWithCovariance & pose, Geometry::PoseWithCovariance & poseOut,
                                                                       pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrSrc,
                                                                       pcl::PointCloud< pcl::PointXYZRGB >::Ptr & corrTgt,
                                                                       const int numRegistrationStepsCoarse, const int numRegistrationStepsFine,
                                                                       bool regularizeRegistration, const Eigen::Matrix6d& registrationPriorPoseCov ) {

    Eigen::Matrix4d transform = pose.mean_.asMatrix4d();

    SurfelAssociationList associations = surfelAssociations;

    bool useFeatures = params_.use_features_;
    float minResolution = std::min( source.min_resolution_, target.min_resolution_ );
    float maxResolution = 16.0 * minResolution;

    float searchDistFactor = 2.f;
    float maxSearchDist = 2.f*maxResolution;

    source_ = &source;
    target_ = &target;
    params_.startResolution_ = maxResolution;
    params_.stopResolution_ = minResolution;
    correspondences_source_points_ = corrSrc;
    correspondences_target_points_ = corrTgt;
    targetSamplingMap_ = target_->samplingMap_;

    if( regularizeRegistration ) {
		setPriorPose( true, pose.mean_.asVector().block<6,1>(0,0), registrationPriorPoseCov );
    }
    else
        setPriorPoseEnabled( false );

    params_.prior_pose_invcov_ /= params_.interpolation_cov_factor_;
    bool success = estimateTransformationLevenbergMarquardt( transform, numRegistrationStepsCoarse, &associations, true, false );
    params_.prior_pose_invcov_ *= params_.interpolation_cov_factor_;

//    setPriorPoseEnabled( false );

//    params_.use_features_ = false;
    estimateTransformationLevenbergMarquardt( transform, numRegistrationStepsFine, &associations, true, true );
//    params_.use_features_ = useFeatures;


    if ( success ) {
        poseOut.mean_ = Geometry::Pose( transform );
        Eigen::Matrix6d covariance;
        bool cov_success = estimatePoseCovarianceLM( covariance, source, target, transform, maxResolution, minResolution, &associations, true );
        if ( cov_success ) {
            poseOut.covariance_ = covariance;
            surfelAssociations = associations;
        } else {
            LOG_STREAM("Estimate pose covariance failed");
            poseOut.covariance_ = pose.covariance_;
        }
    } else {
        poseOut = pose;
        LOG_STREAM( "Registration failed!" );
    }

    return success;
}

