/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 01.07.2014
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

#ifndef SURFEL_H_
#define SURFEL_H_

#include <Eigen/Core>
#include <Eigen/Eigen>

#include <vector>
#include <set>

#include <mrsmap/map/shapetexture_feature.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/vector_average.h>


//#define MIN_SURFEL_POINTS 10.0
#define MAX_SURFEL_POINTS 10000.0 //stop at this point count, since the sums may get numerically unstable



namespace mrsmap {

	// TODO (Jan): move optional fields to vectors indexed by surfel idx.


	template< unsigned int MinPoints > struct GSurfel_evaluator;
	template< unsigned int MinPoints > struct GSurfel_unevaluator;

	template< unsigned int MinPoints >
	class GSurfel {
	public:
		GSurfel() {
			clear();
		}

		~GSurfel() {}

		inline void clear() {

			num_points_ = 0.0;
			mean_.setZero();
			cov_.setZero();

			up_to_date_ = false;
			applyUpdate_ = true;
			unevaluated_ = false;

			assocWeight_ = 1.f;

			idx_ = -1;

			reference_pose_set = false;

			seenThrough_ = false;

		}


		inline GSurfel& operator+=(const GSurfel& rhs) {

			if( !applyUpdate_ )
				return *this;

			if( rhs.num_points_ > 0 && num_points_ < MAX_SURFEL_POINTS ) {

				// numerically stable one-pass update scheme
				if( num_points_ <= std::numeric_limits<double>::epsilon() ) {
					cov_ = rhs.cov_;
					mean_ = rhs.mean_;
					num_points_ = rhs.num_points_;
				}
				else {
					const Eigen::Matrix< double, 6, 1 > deltaS = rhs.num_points_ * mean_ - num_points_ * rhs.mean_;
					cov_ += rhs.cov_ + 1.0 / (num_points_ * rhs.num_points_ * (rhs.num_points_ + num_points_)) * deltaS * deltaS.transpose();
					mean_ += rhs.mean_;
					num_points_ += rhs.num_points_;
				}

				first_view_dir_ = rhs.first_view_dir_;
				up_to_date_ = false;
			}

			return *this;
		}


		inline void add( const Eigen::Matrix< double, 6, 1 >& point ) {

			if( !applyUpdate_ )
				return;

			// numerically stable one-pass update scheme
			if( num_points_ < std::numeric_limits<double>::epsilon() ) {
				mean_ += point;
				num_points_ += 1.0;
				up_to_date_ = false;
			}
			else if( num_points_ < MAX_SURFEL_POINTS ) {
				const Eigen::Matrix< double, 6, 1 > deltaS = (mean_ - num_points_ * point);
				cov_ += 1.0 / (num_points_ * (num_points_ + 1.0)) * deltaS * deltaS.transpose();
				mean_ += point;
				num_points_ += 1.0;
				up_to_date_ = false;
			}
		}


		inline void evaluateNormal() {

			Eigen::Matrix< double, 3, 1> eigen_values_;
			Eigen::Matrix< double, 3, 3> eigen_vectors_;

			// eigen vectors are stored in the columns
			pcl::eigen33(Eigen::Matrix3d(cov_.block<3,3>(0,0)), eigen_vectors_, eigen_values_);

			normal_ = eigen_vectors_.col(0);
			if( normal_.dot( first_view_dir_ ) > 0.0 )
				normal_ *= -1.0;

		}

		inline void evaluate();
		inline void unevaluate();


		// transforms from local surfel frame to map frame
		inline void updateReferencePose() {
			Eigen::Vector3d pos = mean_.block<3,1>(0,0);
			Eigen::AngleAxisd refRot( -acos ( normal_.dot( Eigen::Vector3d::UnitX () ) ),
									normal_.cross( Eigen::Vector3d::UnitX () ).normalized () );

			reference_pose_.block<3,1>(0,0) = pos;
			Eigen::Quaterniond q( refRot );
			reference_pose_(3,0) = q.x();
			reference_pose_(4,0) = q.y();
			reference_pose_(5,0) = q.z();
			reference_pose_(6,0) = q.w();

			reference_pose_set = true;
		}

	  Eigen::Matrix< double, 3, 1 > initial_view_dir_, first_view_dir_;

	  double num_points_;
	  Eigen::Matrix< double, 6, 1 > mean_;
	  Eigen::Matrix< double, 3, 1 > normal_;
	  Eigen::Matrix< double, 6, 6 > cov_;
	  bool up_to_date_, applyUpdate_;
	  bool unevaluated_;

	  // TODO (Jan): move to outside vector on idx_
	  float assocDist_;
	  float assocWeight_;

	  bool seenThrough_; // TODO (Jan) pull this out to a vector

	  int idx_;
	  int ref_idx_;

	  Eigen::Matrix< double, 7, 1 > reference_pose_;
	  bool reference_pose_set;

	  ShapeTextureFeature simple_shape_texture_features_; // TODO (Jan): move to outside vector on idx_
	  ShapeTextureFeature agglomerated_shape_texture_features_;

	  static constexpr double min_points_ = MinPoints;

	public:
	  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	};


	template< unsigned int MinPoints >
	struct GSurfel_evaluator {
		void operator()( GSurfel< MinPoints >* self ) {

			if( self->num_points_ >= self->min_points_ ) {

				const double inv_num = 1.0 / self->num_points_;
				self->mean_ *= inv_num;
				self->cov_ /= (self->num_points_-1.0);


				// enforce symmetry..
				self->cov_(1,0) = self->cov_(0,1);
				self->cov_(2,0) = self->cov_(0,2);
				self->cov_(3,0) = self->cov_(0,3);
				self->cov_(4,0) = self->cov_(0,4);
				self->cov_(5,0) = self->cov_(0,5);
				self->cov_(2,1) = self->cov_(1,2);
				self->cov_(2,3) = self->cov_(3,2);
				self->cov_(2,4) = self->cov_(4,2);
				self->cov_(2,5) = self->cov_(5,2);
				self->cov_(3,1) = self->cov_(1,3);
				self->cov_(3,4) = self->cov_(4,3);
				self->cov_(3,5) = self->cov_(5,3);
				self->cov_(4,1) = self->cov_(1,4);
				self->cov_(4,5) = self->cov_(5,4);
				self->cov_(5,1) = self->cov_(1,5);

				const Eigen::Matrix<double, 6, 6> cov = self->cov_;
				double det = cov.block<3,3>(0,0).determinant();

				if( det <= std::numeric_limits<double>::epsilon() ) {

					self->mean_.setZero();
					self->cov_.setZero();

					self->num_points_ = 0;

					self->clear();
				}

			}


			self->up_to_date_ = true;
			self->unevaluated_ = false;

		}

	};

	template< unsigned int MinPoints >
	struct GSurfel_unevaluator {
		void operator()( GSurfel< MinPoints >* self ) {
			if( self->num_points_ > 0.0 ) {

				self->mean_ *= self->num_points_;
				self->cov_ *= (self->num_points_-1.0);

				self->unevaluated_ = true;

			}
		}
	};


	template<>
	struct GSurfel_evaluator< 1 > {
		void operator()( GSurfel< 1 >* self ) {

			if( self->num_points_ >= self->min_points_ ) {

				const double inv_num = 1.0 / self->num_points_;
				self->mean_ *= inv_num;
				self->cov_ *= inv_num;


				// enforce symmetry..
				self->cov_(1,0) = self->cov_(0,1);
				self->cov_(2,0) = self->cov_(0,2);
				self->cov_(3,0) = self->cov_(0,3);
				self->cov_(4,0) = self->cov_(0,4);
				self->cov_(5,0) = self->cov_(0,5);
				self->cov_(2,1) = self->cov_(1,2);
				self->cov_(2,3) = self->cov_(3,2);
				self->cov_(2,4) = self->cov_(4,2);
				self->cov_(2,5) = self->cov_(5,2);
				self->cov_(3,1) = self->cov_(1,3);
				self->cov_(3,4) = self->cov_(4,3);
				self->cov_(3,5) = self->cov_(5,3);
				self->cov_(4,1) = self->cov_(1,4);
				self->cov_(4,5) = self->cov_(5,4);
				self->cov_(5,1) = self->cov_(1,5);

				double det = self->cov_.block<3,3>(0,0).determinant();

				if( det <= std::numeric_limits<double>::epsilon() ) {

					self->cov_.block<3,3>(0,0) += 1e-9 * Eigen::Matrix3d::Identity();

				}

			}


			self->up_to_date_ = true;
			self->unevaluated_ = false;

		}

	};


	template<>
	struct GSurfel_unevaluator< 1 > {
		void operator()( GSurfel< 1 >* self ) {
			if( self->num_points_ > 0.0 ) {

				self->mean_ *= self->num_points_;
				self->cov_ *= self->num_points_;

				self->unevaluated_ = true;

			}
		}
	};



	template< unsigned int MinPoints >
	void GSurfel< MinPoints >::evaluate() {
		GSurfel_evaluator< MinPoints >()( this );
	}

	template< unsigned int MinPoints >
	void GSurfel< MinPoints >::unevaluate() {
		GSurfel_unevaluator< MinPoints >()( this );
	}


};


#endif /* SURFEL_H_ */

