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

#include <mrsmap/map/shapetexture_feature.h>

#include <mrsmap/map/surfel.h>
#include <boost/math/special_functions/round.hpp>

using namespace mrsmap;

ShapeTextureTable* ShapeTextureTable::instance_ = NULL;

void ShapeTextureTable::initialize() {

	// shape features in [-1,1]
	const float inv_size = 1.f / ((float)SHAPE_TEXTURE_TABLE_SIZE);
	for( unsigned int i = 0; i < SHAPE_TEXTURE_TABLE_SIZE; i++ ) {

		float s = std::min( (NUM_SHAPE_BINS-1.0), std::max( 0., (NUM_SHAPE_BINS-1.0) * ((float)i) * inv_size ) );
		const float ds = s - floor(s);

		unsigned int fs = std::max( 0, std::min( NUM_SHAPE_BINS-1, (int)floor(s) ) );
		unsigned int cs = std::max( 0, std::min( NUM_SHAPE_BINS-1, (int)ceil(s) ) );

		shape_value_table_[0][i].setZero();
		shape_value_table_[1][i].setZero();
		shape_value_table_[2][i].setZero();

		shape_value_table_[0][i]( fs ) = 1.f-ds;
		shape_value_table_[0][i]( cs ) = ds;
		shape_value_table_[1][i]( fs ) = 1.f-ds;
		shape_value_table_[1][i]( cs ) = ds;
		shape_value_table_[2][i]( fs ) = 1.f-ds;
		shape_value_table_[2][i]( cs ) = ds;


		float v = 2.f*((float)i) * inv_size - 1.f;

		float lowl = 0;
		float ctrl = 0;
		float uppl = 0;

		float lowc = 0;
		float ctrc = 0;
		float uppc = 0;

		if( v >= 0 ) {
			if( v >= LUMINANCE_BIN_THRESHOLD )
				uppl = 1.f;
			else {
				uppl = v / LUMINANCE_BIN_THRESHOLD;
				ctrl = 1.f - uppl;
			}

			if( v >= COLOR_BIN_THRESHOLD )
				uppc = 1.f;
			else {
				uppc = v / COLOR_BIN_THRESHOLD;
				ctrc = 1.f - uppc;
			}
		}
		else {

			if( -v >= LUMINANCE_BIN_THRESHOLD )
				lowl = 1.f;
			else {
				lowl = -v / LUMINANCE_BIN_THRESHOLD;
				ctrl = 1.f - lowl;
			}

			if( -v >= COLOR_BIN_THRESHOLD )
				lowc = 1.f;
			else {
				lowc = -v / COLOR_BIN_THRESHOLD;
				ctrc = 1.f - lowc;
			}

		}

		texture_value_table_[0][i].setZero();
		texture_value_table_[1][i].setZero();
		texture_value_table_[2][i].setZero();

		texture_value_table_[0][i]( 0 ) = lowl;
		texture_value_table_[0][i]( 1 ) = ctrl;
		texture_value_table_[0][i]( 2 ) = uppl;

		texture_value_table_[1][i]( 0 ) = lowc;
		texture_value_table_[1][i]( 1 ) = ctrc;
		texture_value_table_[1][i]( 2 ) = uppc;

		texture_value_table_[2][i]( 0 ) = lowc;
		texture_value_table_[2][i]( 1 ) = ctrc;
		texture_value_table_[2][i]( 2 ) = uppc;

	}

}




void ShapeTextureFeature::add( const Eigen::Matrix< double, 6, 1 >& p_src, const Eigen::Matrix< double, 6, 1 >& p_dst, const Eigen::Vector3d& n_src, const Eigen::Vector3d& n_dst, float weight ) {
	// surflet pair relation as in "model globally match locally"
	const Eigen::Vector3d& p1 = p_src.block<3,1>(0,0);
	const Eigen::Vector3d& p2 = p_dst.block<3,1>(0,0);
	const Eigen::Vector3d& n1 = n_src;
	const Eigen::Vector3d& n2 = n_dst;

	Eigen::Vector3d d = p2-p1;
	d.normalize();

	const int s1 = std::min( (SHAPE_TEXTURE_TABLE_SIZE-1), std::max( 0, (int)boost::math::round((SHAPE_TEXTURE_TABLE_SIZE-1.0) * 0.5 * (n1.dot( d )+1.0) ) ) );
	const int s2 = std::min( (SHAPE_TEXTURE_TABLE_SIZE-1), std::max( 0, (int)boost::math::round((SHAPE_TEXTURE_TABLE_SIZE-1.0) * 0.5 * (n2.dot( d )+1.0) ) ) );
	const int s3 = std::min( (SHAPE_TEXTURE_TABLE_SIZE-1), std::max( 0, (int)boost::math::round((SHAPE_TEXTURE_TABLE_SIZE-1.0) * 0.5 * (n1.dot( n2 )+1.0) ) ) );

	shape_.block<1,NUM_SHAPE_BINS>(0,0) += weight * ShapeTextureTable::instance()->shape_value_table_[0][s1];
	shape_.block<1,NUM_SHAPE_BINS>(1,0) += weight * ShapeTextureTable::instance()->shape_value_table_[1][s2];
	shape_.block<1,NUM_SHAPE_BINS>(2,0) += weight * ShapeTextureTable::instance()->shape_value_table_[2][s3];


	const int c1 = std::min( (SHAPE_TEXTURE_TABLE_SIZE-1), std::max( 0, (int)boost::math::round((SHAPE_TEXTURE_TABLE_SIZE-1.0) * 0.5 * ((p_dst(3,0) - p_src(3,0))+1.0) ) ) );
	const int c2 = std::min( (SHAPE_TEXTURE_TABLE_SIZE-1), std::max( 0, (int)boost::math::round((SHAPE_TEXTURE_TABLE_SIZE-1.0) * 0.25 * ((p_dst(4,0) - p_src(4,0))+2.0) ) ) );
	const int c3 = std::min( (SHAPE_TEXTURE_TABLE_SIZE-1), std::max( 0, (int)boost::math::round((SHAPE_TEXTURE_TABLE_SIZE-1.0) * 0.25 * ((p_dst(5,0) - p_src(5,0))+2.0) ) ) );

	texture_.block<1,NUM_TEXTURE_BINS>(0,0) += weight * ShapeTextureTable::instance()->texture_value_table_[0][c1];
	texture_.block<1,NUM_TEXTURE_BINS>(1,0) += weight * ShapeTextureTable::instance()->texture_value_table_[1][c2];
	texture_.block<1,NUM_TEXTURE_BINS>(2,0) += weight * ShapeTextureTable::instance()->texture_value_table_[2][c3];

	num_points_ += weight;
}
