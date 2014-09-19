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

#ifndef SURFEL_PAIR_H_
#define SURFEL_PAIR_H_

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

#include <mrsmap/map/surfel.h>

#include <opencv2/opencv.hpp>



namespace mrsmap {

	struct SurfelPairKey {
		SurfelPairKey( ) : shape1_(0 ), shape2_(0 ),
			shape3_(0 ), shape4_(0 ), color1_(0 ), color2_(0 ) {}

		SurfelPairKey( uint64 shape1, uint64 shape2, uint64 shape3, uint64 shape4,
					   uint64 color1, uint64 color2, uint64 color3 )
			: shape1_(shape1 ), shape2_(shape2 ),
				shape3_(shape3 ), shape4_(shape4 ), color1_(color1 ), color2_(color2 ),
				color3_(color3 ) {}

		bool operator==( const SurfelPairKey & other ) const {
			return shape1_ == other.shape1_ && shape2_ == other.shape2_ &&
					shape3_ == other.shape3_ && shape4_ == other.shape4_ &&
					color1_ == other.color1_ && color2_ == other.color2_ &&
					color3_ == other.color3_;
		}

		size_t hash() const {
			std::size_t seed = 0;
			boost::hash_combine(seed, shape1_);
			boost::hash_combine(seed, shape2_);
			boost::hash_combine(seed, shape3_);
			boost::hash_combine(seed, shape4_);
			boost::hash_combine(seed, color1_);
			boost::hash_combine(seed, color2_);
			boost::hash_combine(seed, color3_);
			return seed;
		}

		uint64 shape1_;
		uint64 shape2_;
		uint64 shape3_;
		uint64 shape4_;
		uint64 color1_;
		uint64 color2_;
		uint64 color3_;
	};



	/**
	 * Signature for a surfel pair
	 */
	class SurfelPairSignature {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		SurfelPairSignature() {
		}

		~SurfelPairSignature() {
		}

		// distance, dot products of n1 to d, n2 to d, and n1 to n2
		Eigen::Matrix< double, 4, 1 > shape_signature_;

		// distance in L, alpha, beta color space
		Eigen::Matrix< double, 3, 1 > color_signature_;

		// hash key
		inline SurfelPairKey getKey( const float maxDist, const float bin_dist, const float bin_angle, const bool use_color = true ) {

				const int bins_chrom = 5;
				const int bins_lum = 3;
				const int bins_angle = (int) 360 / bin_angle;

				uint64 bin_s1 = (int)( shape_signature_(0) / bin_dist );
				uint64 bin_s2 = std::max( 0, std::min( bins_angle-1, (int)(shape_signature_(1) * ((double)bins_angle)) ) );
				uint64 bin_s3 = std::max( 0, std::min( bins_angle-1, (int)(shape_signature_(2) * ((double)bins_angle)) ) );
				uint64 bin_s4 = std::max( 0, std::min( bins_angle-1, (int)(shape_signature_(3) * ((double)bins_angle)) ) );

				uint64 bin_c1 = 0;
				uint64 bin_c2 = 0;
				uint64 bin_c3 = 0;

				if ( use_color ) {
					bin_c1 = std::max( 0, std::min( bins_lum-1, (int)(color_signature_(0) * ((double)bins_lum)) ) );
					bin_c2 = std::max( 0, std::min( bins_chrom-1, (int)(color_signature_(1) * ((double)bins_chrom)) ) );
					bin_c3 = std::max( 0, std::min( bins_chrom-1, (int)(color_signature_(2) * ((double)bins_chrom)) ) );
				}

				key_ = SurfelPairKey( bin_s1, bin_s2, bin_s3, bin_s4,
									  bin_c1, bin_c2, bin_c3 );

			return key_;

		}

		float alpha_;
		SurfelPairKey key_;

	};


	/**
	 * Signature for a surfel pair
	 */
	template< unsigned int MinPoints >
	class GSurfelPair {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		GSurfelPair( GSurfel< MinPoints >* src, GSurfel< MinPoints >* dst, const SurfelPairSignature& signature, float weight = 0.f )
			: src_(src), dst_(dst), signature_( signature ),   weight_( weight ) {

		}

		~GSurfelPair() {
		}

		GSurfel< MinPoints >* src_;
		GSurfel< MinPoints >* dst_;

		SurfelPairSignature signature_;

		float weight_;
	};


};


namespace std {
	template<>
	class hash< mrsmap::SurfelPairKey >
		: public std::unary_function< mrsmap::SurfelPairKey, size_t >
	{
	public:
		size_t operator()( const mrsmap::SurfelPairKey& k ) const
		{
			return k.hash();
		}
	};
};


#endif /* SURFEL_PAIR_H_ */

