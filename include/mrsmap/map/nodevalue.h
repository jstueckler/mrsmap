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

#ifndef NODE_VALUE_H_
#define NODE_VALUE_H_

#include <Eigen/Core>
#include <Eigen/Eigen>

#include <vector>
#include <set>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/functional/hash.hpp>
//#include <boost/unordered_map.hpp>
#include <unordered_map>
#include <functional>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/vector_average.h>

#include <octreelib/spatialaggregate/octree.h>
#include <octreelib/algorithm/downsample.h>

#include <mrsmap/map/surfel.h>
#include <mrsmap/map/shapetexture_feature.h>

#include <gsl/gsl_rng.h>

#include <pcl/common/time.h>

#define uchar flann_uchar
#include <flann/flann.h>
#undef uchar //Prevent ambiguous symbol error when OpenCV defines uchar

#include <opencv2/opencv.hpp>

#include <tbb/tbb.h>
#include <tbb/concurrent_queue.h>



namespace mrsmap {

	template< typename TSurfel, unsigned int NumSurfels > struct GNodeValue_initializer;

	template< typename TSurfel, unsigned int NumSurfels >
	class GNodeValue {
	public:
		GNodeValue();
		GNodeValue( unsigned int v );
		~GNodeValue() {}

		typedef TSurfel Surfel;

		void initialize();

		inline GNodeValue& operator+=(const GNodeValue& rhs) {

			// merge surfels
			for( unsigned int i = 0; i < NumSurfels; i++ ) {

				TSurfel& surfel = surfels_[i];

				if( surfel.applyUpdate_ ) {
					if( surfel.up_to_date_ )
						surfel.clear();

					surfel += rhs.surfels_[i];
				}

			}

			return *this;
		}


		inline TSurfel* getSurfel( const Eigen::Vector3d& viewDirection ) {

			TSurfel* bestMatchSurfel = NULL;
			double bestMatchDist = -1.;

			for( unsigned int i = 0; i < NumSurfels; i++ ) {
				const double dist = viewDirection.dot( surfels_[i].initial_view_dir_ );
				if( dist > bestMatchDist ) {
					bestMatchSurfel = &surfels_[i];
					bestMatchDist = dist;
				}
			}

			return bestMatchSurfel;
		}


		inline void addSurfel( const Eigen::Vector3d& viewDirection, const TSurfel& surfel ) {

			// find best matching surfel for the view direction
			TSurfel* bestMatchSurfel = getSurfel( viewDirection );

			if( bestMatchSurfel->applyUpdate_ ) {
				if( bestMatchSurfel->up_to_date_ )
					bestMatchSurfel->clear();

				*bestMatchSurfel += surfel;
			}

		}


		inline void evaluateNormals( spatialaggregate::OcTreeNode<float, GNodeValue>* node ) {
			for( unsigned int i = 0; i < NumSurfels; i++ ) {
				if( !surfels_[i].up_to_date_ || surfels_[i].unevaluated_ ) {

					TSurfel surfel = surfels_[i];
					for( unsigned int n = 0; n < 27; n++ ) {
						if(node->neighbors_[n] && node->neighbors_[n] != node ) {
							surfel += node->neighbors_[n]->value_.surfels_[i];
						}
					}

					surfel.first_view_dir_ = surfels_[i].first_view_dir_;
					surfel.evaluate();
					surfel.evaluateNormal();

					surfels_[i].normal_ = surfel.normal_;

				}
			}
		}


		inline void evaluateSurfels() {
			for( unsigned int i = 0; i < NumSurfels; i++ ) {
				if( !surfels_[i].up_to_date_ || surfels_[i].unevaluated_ ) {
					surfels_[i].evaluate();
				}
			}
		}

		inline void unevaluateSurfels() {
			for( unsigned int i = 0; i < NumSurfels; i++ ) {
				if( surfels_[i].up_to_date_ ) {
					surfels_[i].unevaluate();
				}
			}
		}



		TSurfel surfels_[ NumSurfels ];

		int idx_;

		bool border_;

		static const unsigned int num_surfels_ = NumSurfels;


		// TODO: move these out to a vector on idx_ where it is really needed
		char associated_; // -1: disabled, 0: not associated, 1: associated, 2: not associated but neighbor of associated node
		spatialaggregate::OcTreeNode<float, GNodeValue>* association_;
		char assocSurfelIdx_, assocSurfelDstIdx_;
		float assocWeight_;

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};


	template< typename TSurfel, unsigned int NumSurfels >
	struct GNodeValue_initializer {
		void operator()( GNodeValue< TSurfel, NumSurfels >* self ) {

			self->idx_ = -1;
			self->associated_ = 0;
			self->assocWeight_ = 1.f;
			self->border_ = false;

			for( unsigned int i = 0; i < NumSurfels; i++ ) {
				self->surfels_[i].initial_view_dir_ = self->surfels_[i].first_view_dir_ = Eigen::Vector3d::UnitZ();
			}

		}

	};


	template< typename TSurfel >
	struct GNodeValue_initializer< TSurfel, 6 > {
		void operator()( GNodeValue< TSurfel, 6 >* self ) {

			self->idx_ = -1;
			self->associated_ = 0;
			self->assocWeight_ = 1.f;
			self->border_ = false;

			self->surfels_[0].initial_view_dir_ = Eigen::Vector3d( 1., 0., 0. );
			self->surfels_[1].initial_view_dir_ = Eigen::Vector3d( -1., 0., 0. );
			self->surfels_[2].initial_view_dir_ = Eigen::Vector3d( 0., 1., 0. );
			self->surfels_[3].initial_view_dir_ = Eigen::Vector3d( 0., -1., 0. );
			self->surfels_[4].initial_view_dir_ = Eigen::Vector3d( 0., 0., 1. );
			self->surfels_[5].initial_view_dir_ = Eigen::Vector3d( 0., 0., -1. );

			for( unsigned int i = 0; i < 6; i++ )
				self->surfels_[i].first_view_dir_ = self->surfels_[i].initial_view_dir_;

		}

	};

	template< typename TSurfel, unsigned int NumSurfels >
	GNodeValue< TSurfel, NumSurfels >::GNodeValue() {
		GNodeValue_initializer< TSurfel, NumSurfels >()( this );
	}

	template< typename TSurfel, unsigned int NumSurfels >
	GNodeValue< TSurfel, NumSurfels >::GNodeValue( unsigned int v ) {
		GNodeValue_initializer< TSurfel, NumSurfels >()( this );
	}

	template< typename TSurfel, unsigned int NumSurfels >
	void GNodeValue< TSurfel, NumSurfels >::initialize() {
		GNodeValue_initializer< TSurfel, NumSurfels >()( this );
	}




};

#endif /* NODE_VALUE_H_ */

