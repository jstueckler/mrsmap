/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 05.12.2012
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

#ifndef SHAPETEXTURE_FEATURE_H_
#define SHAPETEXTURE_FEATURE_H_

#include <Eigen/Core>
#include <Eigen/Eigen>


#define NUM_SHAPE_BINS 3
#define NUM_TEXTURE_BINS 3

#define LUMINANCE_BIN_THRESHOLD 0.1
#define COLOR_BIN_THRESHOLD 0.05

#define SHAPE_TEXTURE_TABLE_SIZE 10000



namespace mrsmap {

	class ShapeTextureFeature {
	public:

		ShapeTextureFeature() {
			initialize();
		}

		~ShapeTextureFeature() {
		}

		void initialize() {
			shape_.setZero();
			texture_.setZero();
			num_points_ = 0.f;
		}


		void add( const Eigen::Matrix< double, 6, 1 >& p_src, const Eigen::Matrix< double, 6, 1 >& p_dst, const Eigen::Vector3d& n_src, const Eigen::Vector3d& n_dst, float weight );

		void add( const ShapeTextureFeature& feature, float weight ) {
			shape_ += weight*feature.shape_;
			texture_ += weight*feature.texture_;
			num_points_ += weight*feature.num_points_;
		}

		inline float textureDistance( const ShapeTextureFeature& feature ) const {
			return (texture_ - feature.texture_).squaredNorm();
		}

		inline float shapeDistance( const ShapeTextureFeature& feature ) const {
			return (shape_ - feature.shape_).squaredNorm();
		}

		inline float distance( const ShapeTextureFeature& feature ) const {
			return (shape_ - feature.shape_).squaredNorm() + (texture_ - feature.texture_).squaredNorm();
		}


		EIGEN_ALIGN16 Eigen::Matrix< float, 3, NUM_SHAPE_BINS > shape_;
		EIGEN_ALIGN16 Eigen::Matrix< float, 3, NUM_TEXTURE_BINS > texture_;
		float num_points_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};


	class ShapeTextureTable {
	public:

		ShapeTextureTable() { initialize(); }
		~ShapeTextureTable() {}

		Eigen::Matrix< float, 1, NUM_SHAPE_BINS > shape_value_table_[3][ SHAPE_TEXTURE_TABLE_SIZE ];
		Eigen::Matrix< float, 1, NUM_TEXTURE_BINS > texture_value_table_[3][ SHAPE_TEXTURE_TABLE_SIZE ];

		void initialize();

		static ShapeTextureTable* instance() {
			if( !instance_ )
				instance_ = new ShapeTextureTable();
			return instance_;
		}

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	protected:
		static ShapeTextureTable* instance_;

	};

};



#endif /* SHAPETEXTURE_FEATURE_H_ */

