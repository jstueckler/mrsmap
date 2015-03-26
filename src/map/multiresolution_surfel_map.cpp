/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 02.05.2011
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

#include "mrsmap/map/multiresolution_surfel_map.h"

#include <mrsmap/utilities/utilities.h>

#include "octreelib/feature/normalestimation.h"
#include "octreelib/algorithm/downsample.h"

#include <mrsmap/utilities/eigen_extensions.h>

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#include <ostream>
#include <fstream>


using namespace mrsmap;

#define MAX_VIEWDIR_DIST cos( 0.25 * M_PI + 0.125*M_PI )

// for surfel pairs
#define MAX_ANGLE_DIFF 0.98f
#define MIN_LUMINANCE_DIFF 0.1f
#define MIN_COLOR_DIFF 0.05f

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>

#define LOG_OUTPUT 1

#define LOG_STREAM(args) \
    if ( LOG_OUTPUT ) std::cout << std::setprecision (15) << args << std::endl;


gsl_rng* MultiResolutionSurfelMap::r = NULL;


MultiResolutionSurfelMap::Params::Params() {

	dist_dependency = 0.01f;

	depthNoiseFactor = 1.f;
	pixelNoise = 3.f;
//	depthNoiseAssocFactor = 4.f;
//	pixelNoiseAssocFactor = 4.f;
	usePointFeatures = false;
	debugPointFeatures = false;
	GridCols = 8;
	GridRows = 6;
	numPointFeatures = 4000;
	GridCellMax = 25;

    surfelPairFeatureBinAngle_ = 10;
    surfelPairFeatureBinDist_ = 0.05;
    surfelPairSamplingRate_ = 1.f;
    surfelPairMaxDist_ = 1.f;
    surfelPairMaxDistResFactor_ = 8.f;
    surfelPairMinDepth_ = 1;
    surfelPairMaxDepth_ = 16;
    surfelPairUseColor_ = true;

    parallel_ = true;

}



MultiResolutionSurfelMap::ImagePreAllocator::ImagePreAllocator()
: imageNodeAllocator_( 20000 ) {
	imgKeys = NULL;
	valueMap = NULL;
	node_image_ = NULL;
	width = height = 0;
}


MultiResolutionSurfelMap::ImagePreAllocator::~ImagePreAllocator() {
	if( imgKeys )
		delete[] imgKeys;

	if( valueMap )
		delete[] valueMap;
}


void MultiResolutionSurfelMap::ImagePreAllocator::prepare( unsigned int w, unsigned int h, bool buildNodeImage ) {

	typedef NodeValue* NodeValuePtr;
	typedef spatialaggregate::OcTreeNode< float, NodeValue >* NodePtr;

	if( !valueMap || height != h || width != w ) {

		if( imgKeys )
			delete[] imgKeys;
		imgKeys = new uint64_t[ w*h ];

		if( valueMap )
			delete[] valueMap;

		valueMap = new NodeValuePtr[ w*h ];

		if( node_image_ )
			delete[] node_image_;

		if( buildNodeImage )
			node_image_ = new NodePtr[w*h];

		infoList.resize( w*h );

		width = w;
		height = h;

	}

	memset( &imgKeys[0], 0LL, w*h * sizeof( uint64_t ) );
	memset( &valueMap[0], 0, w*h * sizeof( NodeValuePtr ) );
	if( buildNodeImage )
		memset( &node_image_[0], 0, w*h * sizeof( NodePtr ) );
	imageNodeAllocator_.reset();

	parallelInfoList.clear();

}


MultiResolutionSurfelMap::MultiResolutionSurfelMap( float minResolution, float maxRange) {
	construct( minResolution, maxRange, boost::make_shared< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > >() );
}

MultiResolutionSurfelMap::MultiResolutionSurfelMap( float minResolution, float maxRange, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > > allocator ) {
	construct( minResolution, maxRange, allocator );
}

void MultiResolutionSurfelMap::construct( float minResolution, float maxRange, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > > allocator ) {

	min_resolution_ = minResolution;
	max_range_ = maxRange;

	last_pair_surfel_idx_ = 0;

	reference_pose_.setIdentity();

	Eigen::Matrix< float, 4, 1 > center( 0.f, 0.f, 0.f, 0.f );
	allocator_ = allocator;
	octree_ = boost::shared_ptr< spatialaggregate::OcTree< float, NodeValue > >( new spatialaggregate::OcTree< float, NodeValue >( center, minResolution, maxRange, allocator ) );

	if ( !r ) {
		const gsl_rng_type* T = gsl_rng_default;
		gsl_rng_env_setup();
		r = gsl_rng_alloc( T );
	}


}

MultiResolutionSurfelMap::~MultiResolutionSurfelMap() {
}


void MultiResolutionSurfelMap::extents( Eigen::Matrix< double, 3, 1 >& mean, Eigen::Matrix< double, 3, 3 >& cov ) {

	std::list< spatialaggregate::OcTreeNode< float, NodeValue >* > nodes;
	octree_->root_->getAllLeaves( nodes );

	Eigen::Matrix< double, 3, 1 > sum;
	Eigen::Matrix< double, 3, 3 > sumSquares;
	double numPoints = 0;
	sum.setZero();
	sumSquares.setZero();

	for( std::list< spatialaggregate::OcTreeNode< float, NodeValue >* >::iterator it = nodes.begin(); it != nodes.end(); ++it ) {

		  NodeValue& v = (*it)->value_;

		  for( int i = 0; i < NodeValue::num_surfels_; i++ ) {

				Eigen::Vector3d mean_s = v.surfels_[i].mean_.block<3,1>(0,0);
				double num_points_s = v.surfels_[i].num_points_;

				sum += num_points_s * mean_s;
				sumSquares += num_points_s * (v.surfels_[i].cov_.block<3,3>(0,0) + mean_s * mean_s.transpose());
				numPoints += num_points_s;

		  }

	}

	if( numPoints > 0 ) {

		  const double inv_num = 1.0 / numPoints;
		  mean = sum * inv_num;
		  cov = inv_num * sumSquares - mean * mean.transpose();

	}

}

void MultiResolutionSurfelMap::addPoints( const boost::shared_ptr< const pcl::PointCloud< pcl::PointXYZRGB > >& cloud, const boost::shared_ptr< const std::vector< int > >& indices, bool smoothViewDir ) {
	addPoints( *cloud, *indices, smoothViewDir );
}

void MultiResolutionSurfelMap::addPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices, bool smoothViewDir ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f*sqrtf(3.f);
	const double max_dist = MAX_VIEWDIR_DIST;


	// go through the point cloud and add point information to map
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( boost::math::isnan( x ) || boost::math::isinf( x ) )
			continue;

		if ( boost::math::isnan( y ) || boost::math::isinf( y ) )
			continue;

		if ( boost::math::isnan( z ) || boost::math::isinf( z ) )
			continue;

		float rgbf = p.rgb;
		unsigned int rgb = * ( reinterpret_cast< unsigned int* > ( &rgbf ) );
		unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
		unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
		unsigned int b = ( rgb & 0x000000FF );

		// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
		float rf = inv_255*r, gf = inv_255*g, bf = inv_255*b;


		// RGB to L-alpha-beta:
		float L = 0.5f * ( std::max( std::max( rf, gf ), bf ) + std::min( std::min( rf, gf ), bf ) );
		float alpha = 0.5f * ( 2.f*rf - gf - bf );
		float beta = sqrt305 * (gf-bf);

		Eigen::Matrix< double, 6, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;
		pos( 3 ) = L;
		pos( 4 ) = alpha;
		pos( 5 ) = beta;


		Eigen::Vector3d viewDirection = pos.block< 3, 1 > ( 0, 0 ) - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		double viewDistanceInv = 1.0 / viewDistance;
		viewDirection *= viewDistanceInv;

		double distanceWeight = 1.0;

		Surfel surfel;
		surfel.add( pos );
//		surfel.add( distanceWeight * pos, ( distanceWeight * pos ) * pos.transpose(), distanceWeight );
		surfel.first_view_dir_ = viewDirection;

		NodeValue value;

		if( !smoothViewDir ) {
			Surfel* surfel = value.getSurfel( viewDirection );
			surfel->add( pos );
		}
		else {
			// add surfel to view directions within an angular interval
			for( unsigned int k = 0; k < NodeValue::num_surfels_; k++ ) {
				const double dist = viewDirection.dot( value.surfels_[k].initial_view_dir_ );
				if( dist > max_dist ) {
					value.surfels_[k] += surfel;
				}
			}
		}




		// max resolution depends on depth: the farer, the bigger the minimumVolumeSize
		// see: http://www.ros.org/wiki/openni_kinect/kinect_accuracy
		// i roughly used the 90% percentile function for a single kinect
		int depth = ceil( octree_->depthForVolumeSize( std::max( (float) min_resolution_, (float) ( 2.f * params_.dist_dependency * viewDistance * viewDistance ) ) ) );

		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->addPoint( p.getVector4fMap(), value, depth );


	}

}


void MultiResolutionSurfelMap::addImage( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, bool smoothViewDir, bool buildNodeImage ) {


	imageAllocator_->prepare( cloud.width, cloud.height, buildNodeImage );
	int imageAggListIdx = 0;

	int idx = 0;
	const unsigned int width4 = 4*cloud.width;
	uint64_t* imgPtr = &imageAllocator_->imgKeys[0];
	NodeValue** mapPtr = &imageAllocator_->valueMap[0];

	const NodeValue initValue;

	Eigen::Vector4d sensorOrigin = cloud.sensor_origin_.cast<double>();
	const double sox = sensorOrigin(0);
	const double soy = sensorOrigin(1);
	const double soz = sensorOrigin(2);

	Eigen::Matrix4f sensorTransform = Eigen::Matrix4f::Identity();
	sensorTransform.block<4,1>(0,3) = cloud.sensor_origin_;
	sensorTransform.block<3,3>(0,0) = cloud.sensor_orientation_.matrix();


	const float inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f*sqrtf(3.f);
	const double max_dist = MAX_VIEWDIR_DIST;

	stopwatch_.reset();

	const float minpx = octree_->min_position_(0);
	const float minpy = octree_->min_position_(1);
	const float minpz = octree_->min_position_(2);

	const float pnx = octree_->position_normalizer_(0);
	const float pny = octree_->position_normalizer_(1);
	const float pnz = octree_->position_normalizer_(2);

	const int maxdepth = octree_->max_depth_;

	const int w = cloud.width;
	const int wm1 = w-1;
	const int wp1 = w+1;
	const int h = cloud.height;

//	std::cout << octree_->max_depth_ << "\n";

	unsigned char depth = maxdepth;
	float minvolsize = octree_->minVolumeSizeForDepth( maxdepth );
	float maxvolsize = octree_->maxVolumeSizeForDepth( maxdepth );

	Eigen::Matrix< double, 6, 1 > pos;
	Eigen::Matrix< double, 1, 6 > posT;

	const float DIST_DEPENDENCY = params_.dist_dependency;

	for( int y = 0; y < h; y++ ) {

		uint64_t keyleft = 0;

		for( int x = 0; x < w; x++ ) {

			const pcl::PointXYZRGB& p = cloud.points[idx++];

			if( boost::math::isnan( p.x ) ) {
				mapPtr++;
				imgPtr++;
				continue;
			}

			Eigen::Vector3d viewDirection( p.x - sox, p.y - soy, p.z - soz );
			const double viewDistance = viewDirection.norm();

			const float distdep = (2. * DIST_DEPENDENCY * viewDistance*viewDistance);

			const unsigned int kx_ = (p.x - minpx) * pnx + 0.5;
			const unsigned int ky_ = (p.y - minpy) * pny + 0.5;
			const unsigned int kz_ = (p.z - minpz) * pnz + 0.5;



			// try to avoid the log
			if( distdep < minvolsize || distdep > maxvolsize ) {

				depth = octree_->depthForVolumeSize( (double)distdep ) + 0.5f;

				if( depth >= maxdepth ) {
					depth = maxdepth;
				}

				minvolsize = octree_->minVolumeSizeForDepth( depth );
				maxvolsize = octree_->maxVolumeSizeForDepth( depth );

			}


			const unsigned int x_ = (kx_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int y_ = (ky_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int z_ = (kz_ >> (MAX_REPRESENTABLE_DEPTH-depth));

			uint64_t imgkey = (((uint64_t)x_ & 0xFFFFLL) << 48) | (((uint64_t)y_ & 0xFFFFLL) << 32) | (((uint64_t)z_ & 0xFFFFLL) << 16) | (uint64_t)depth;

			// check pixel above
			if( y > 0 ) {

				if( imgkey == *(imgPtr-w) )
					*mapPtr = *(mapPtr-w);
				else {

					if( imgkey == *(imgPtr-wp1) ) {
						*mapPtr = *(mapPtr-wp1);
					}
					else {

						// check pixel right
						if( x < wm1 ) {

							if( imgkey == *(imgPtr-wm1) ) {
								*mapPtr = *(mapPtr-wm1);
							}

						}

					}

				}

			}

			// check pixel before
			if( !*mapPtr && imgkey == keyleft ) {
				*mapPtr = *(mapPtr-1);
			}


			const double viewDistanceInv = 1.0 / viewDistance;
			viewDirection *= viewDistanceInv;

			if( !*mapPtr ) {
				// create new node value
				*mapPtr = imageAllocator_->imageNodeAllocator_.allocate();
				memcpy( (*mapPtr)->surfels_, initValue.surfels_, sizeof(initValue.surfels_) );
				for( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {
					(*mapPtr)->surfels_[i].first_view_dir_ = viewDirection;
				}

				ImagePreAllocator::Info& info = imageAllocator_->infoList[imageAggListIdx];
				info.value = *mapPtr;
				info.key.x_ = kx_;
				info.key.y_ = ky_;
				info.key.z_ = kz_;
				info.depth = depth;

				imageAggListIdx++;

			}



			// add point to surfel
			const float rgbf = p.rgb;
			const unsigned int rgb = * ( reinterpret_cast< const unsigned int* > ( &rgbf ) );
			const unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
			const unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
			const unsigned int b = ( rgb & 0x000000FF );

			// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
			const float rf = inv_255*(float)r;
			const float gf = inv_255*(float)g;
			const float bf = inv_255*(float)b;

			float maxch = rf;
			if( bf > maxch )
				maxch = bf;
			if( gf > maxch )
				maxch = gf;

			float minch = rf;
			if( bf < minch )
				minch = bf;
			if( gf < minch )
				minch = gf;

			const float L = 0.5f * ( maxch + minch );
			const float alpha = 0.5f * ( 2.f*rf - gf - bf );
			const float beta = sqrt305 * (gf-bf);


			pos( 0 ) = posT( 0 ) = p.x;
			pos( 1 ) = posT( 1 ) = p.y;
			pos( 2 ) = posT( 2 ) = p.z;
			pos( 3 ) = posT( 3 ) = L;
			pos( 4 ) = posT( 4 ) = alpha;
			pos( 5 ) = posT( 5 ) = beta;


			const Eigen::Matrix< double, 6, 6 > ppT = pos * posT;

			if( !smoothViewDir ) {
				Surfel* surfel = (*mapPtr)->getSurfel( viewDirection );
				surfel->add( pos );
			}
			else {
				// add surfel to view directions within an angular interval
				for( unsigned int k = 0; k < NodeValue::num_surfels_; k++ ) {
					const double dist = viewDirection.dot( (*mapPtr)->surfels_[k].initial_view_dir_ );
					if( dist > max_dist ) {
						(*mapPtr)->surfels_[k].add( pos );
					}
				}
			}

			*imgPtr++ = keyleft = imgkey;
			mapPtr++;

		}
	}

	double delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

//	std::cout << "aggregation took " << delta_t << "\n";

    stopwatch_.reset();

    for( unsigned int i = 0; i < imageAggListIdx; i++ ) {

    	const ImagePreAllocator::Info& info = imageAllocator_->infoList[i];
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_->addPoint( info.key, *info.value, info.depth );
		info.value->association_ = n;

    }

	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

//	std::cout << "tree construction took " << delta_t << "\n";

	imageAllocator_->node_set_.clear();

	if( buildNodeImage ) {

		NodeValue** mapPtr = &imageAllocator_->valueMap[0];
		unsigned int idx = 0;

		NodeValue* lastNodeValue = NULL;

		for( int y = 0; y < h; y++ ) {

			for( int x = 0; x < w; x++ ) {

				if( *mapPtr ) {
					imageAllocator_->node_image_[idx++] = (*mapPtr)->association_;
					if( *mapPtr != lastNodeValue ) {
						imageAllocator_->node_set_.insert( (*mapPtr)->association_ );
					}
				}
				else
					imageAllocator_->node_image_[idx++] = NULL;

				lastNodeValue = *mapPtr;
				mapPtr++;

			}
		}

	}

}


void MultiResolutionSurfelMap::addDisplacementImage( const pcl::PointCloud< pcl::PointXYZRGB >& cloud_pos, const pcl::PointCloud< pcl::PointXYZRGB >& cloud_disp, bool smoothViewDir, bool buildNodeImage ) {


	imageAllocator_->prepare( cloud_pos.width, cloud_pos.height, buildNodeImage );
	int imageAggListIdx = 0;

	int idx = 0;
	const unsigned int width4 = 4*cloud_pos.width;
	uint64_t* imgPtr = &imageAllocator_->imgKeys[0];
	NodeValue** mapPtr = &imageAllocator_->valueMap[0];

	const NodeValue initValue;

	Eigen::Vector4d sensorOrigin = cloud_pos.sensor_origin_.cast<double>();
	const double sox = sensorOrigin(0);
	const double soy = sensorOrigin(1);
	const double soz = sensorOrigin(2);

	Eigen::Matrix4f sensorTransform = Eigen::Matrix4f::Identity();
	sensorTransform.block<4,1>(0,3) = cloud_pos.sensor_origin_;
	sensorTransform.block<3,3>(0,0) = cloud_pos.sensor_orientation_.matrix();


	const float inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f*sqrtf(3.f);
	const double max_dist = MAX_VIEWDIR_DIST;

	stopwatch_.reset();

	const float minpx = octree_->min_position_(0);
	const float minpy = octree_->min_position_(1);
	const float minpz = octree_->min_position_(2);

	const float pnx = octree_->position_normalizer_(0);
	const float pny = octree_->position_normalizer_(1);
	const float pnz = octree_->position_normalizer_(2);

	const int maxdepth = octree_->max_depth_;

	const int w = cloud_pos.width;
	const int wm1 = w-1;
	const int wp1 = w+1;
	const int h = cloud_pos.height;

	unsigned char depth = maxdepth;
	float minvolsize = octree_->minVolumeSizeForDepth( maxdepth );
	float maxvolsize = octree_->maxVolumeSizeForDepth( maxdepth );

	Eigen::Matrix< double, 6, 1 > pos;
	Eigen::Matrix< double, 1, 6 > posT;

	const float DIST_DEPENDENCY = params_.dist_dependency;

	for( int y = 0; y < h; y++ ) {

		uint64_t keyleft = 0;

		for( int x = 0; x < w; x++ ) {

			const pcl::PointXYZRGB& p_disp = cloud_disp.points[idx];
			const pcl::PointXYZRGB& p = cloud_pos.points[idx++];

			if( boost::math::isnan( p.x ) ) {
				mapPtr++;
				imgPtr++;
				continue;
			}

			Eigen::Vector3d viewDirection( p.x - sox, p.y - soy, p.z - soz );
			const double viewDistance = viewDirection.norm();

			const float distdep = (2. * DIST_DEPENDENCY * viewDistance*viewDistance);

			const unsigned int kx_ = (p.x - minpx) * pnx + 0.5;
			const unsigned int ky_ = (p.y - minpy) * pny + 0.5;
			const unsigned int kz_ = (p.z - minpz) * pnz + 0.5;



			// try to avoid the log
			if( distdep < minvolsize || distdep > maxvolsize ) {

				depth = octree_->depthForVolumeSize( (double)distdep ) + 0.5f;

				if( depth >= maxdepth ) {
					depth = maxdepth;
				}

				minvolsize = octree_->minVolumeSizeForDepth( depth );
				maxvolsize = octree_->maxVolumeSizeForDepth( depth );

			}


			const unsigned int x_ = (kx_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int y_ = (ky_ >> (MAX_REPRESENTABLE_DEPTH-depth));
			const unsigned int z_ = (kz_ >> (MAX_REPRESENTABLE_DEPTH-depth));

			uint64_t imgkey = (((uint64_t)x_ & 0xFFFFLL) << 48) | (((uint64_t)y_ & 0xFFFFLL) << 32) | (((uint64_t)z_ & 0xFFFFLL) << 16) | (uint64_t)depth;

			// check pixel above
			if( y > 0 ) {

				if( imgkey == *(imgPtr-w) )
					*mapPtr = *(mapPtr-w);
				else {

					if( imgkey == *(imgPtr-wp1) ) {
						*mapPtr = *(mapPtr-wp1);
					}
					else {

						// check pixel right
						if( x < wm1 ) {

							if( imgkey == *(imgPtr-wm1) ) {
								*mapPtr = *(mapPtr-wm1);
							}

						}

					}

				}

			}

			// check pixel before
			if( !*mapPtr && imgkey == keyleft ) {
				*mapPtr = *(mapPtr-1);
			}


			const double viewDistanceInv = 1.0 / viewDistance;
			viewDirection *= viewDistanceInv;

			if( !*mapPtr ) {
				// create new node value
				*mapPtr = imageAllocator_->imageNodeAllocator_.allocate();
				memcpy( (*mapPtr)->surfels_, initValue.surfels_, sizeof(initValue.surfels_) );
				for( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {
					(*mapPtr)->surfels_[i].first_view_dir_ = viewDirection;
				}

				ImagePreAllocator::Info& info = imageAllocator_->infoList[imageAggListIdx];
				info.value = *mapPtr;
				info.key.x_ = kx_;
				info.key.y_ = ky_;
				info.key.z_ = kz_;
				info.depth = depth;

				imageAggListIdx++;

			}



			// add point to surfel

			pos( 0 ) = posT( 0 ) = p.x;
			pos( 1 ) = posT( 1 ) = p.y;
			pos( 2 ) = posT( 2 ) = p.z;
			pos( 3 ) = posT( 3 ) = p_disp.x;
			pos( 4 ) = posT( 4 ) = p_disp.y;
			pos( 5 ) = posT( 5 ) = p_disp.z;


			const Eigen::Matrix< double, 6, 6 > ppT = pos * posT;

			if( !smoothViewDir ) {
				Surfel* surfel = (*mapPtr)->getSurfel( viewDirection );
//				surfel->add( pos, ppT, 1.0 );
				surfel->add( pos );
			}
			else {
				// add surfel to view directions within an angular interval
				for( unsigned int k = 0; k < NodeValue::num_surfels_; k++ ) {
					const double dist = viewDirection.dot( (*mapPtr)->surfels_[k].initial_view_dir_ );
					if( dist > max_dist ) {
//						(*mapPtr)->surfels_[k].add( pos, ppT, 1.0 );
						(*mapPtr)->surfels_[k].add( pos );
					}
				}
			}

			*imgPtr++ = keyleft = imgkey;
			mapPtr++;

		}
	}

	double delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

//	std::cout << "aggregation took " << delta_t << "\n";

    stopwatch_.reset();

    for( unsigned int i = 0; i < imageAggListIdx; i++ ) {

    	const ImagePreAllocator::Info& info = imageAllocator_->infoList[i];
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_->addPoint( info.key, *info.value, info.depth );
		info.value->association_ = n;

    }

	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;

//	std::cout << "tree construction took " << delta_t << "\n";

	imageAllocator_->node_set_.clear();

	if( buildNodeImage ) {

		NodeValue** mapPtr = &imageAllocator_->valueMap[0];
		unsigned int idx = 0;

		NodeValue* lastNodeValue = NULL;

		for( int y = 0; y < h; y++ ) {

			for( int x = 0; x < w; x++ ) {

				if( *mapPtr ) {
					imageAllocator_->node_image_[idx++] = (*mapPtr)->association_;
					if( *mapPtr != lastNodeValue ) {
						imageAllocator_->node_set_.insert( (*mapPtr)->association_ );
					}
				}
				else
					imageAllocator_->node_image_[idx++] = NULL;

				lastNodeValue = *mapPtr;
				mapPtr++;

			}
		}

	}

}




struct KeypointComparator {
	bool operator() ( unsigned int i, unsigned int j ) {
		return (*keypoints_)[i].response > (*keypoints_)[j].response;
	}

	std::vector< cv::KeyPoint >* keypoints_;
};


// requires cloud in sensor frame
void MultiResolutionSurfelMap::addImagePointFeatures( const cv::Mat& img, const pcl::PointCloud< pcl::PointXYZRGB >& cloud ) {

	img_rgb_ = img;

	lsh_index_.reset();
	features_.clear();
	descriptors_ = cv::Mat();

	const float DIST_DEPENDENCY = params_.dist_dependency;

	const float pixelNoise = params_.pixelNoise;
	const float pixelNoise2 = pixelNoise*pixelNoise;
	const float depthNoiseScale2 = params_.depthNoiseFactor * DIST_DEPENDENCY * DIST_DEPENDENCY;

//	const float pixelNoiseAssoc = params_.pixelNoiseAssoc;//20.f;
//	const float pixelNoiseAssoc2 = pixelNoiseAssoc*pixelNoiseAssoc;
//	const float depthNoiseAssocScale2 = params_.depthNoiseAssocFactor * DIST_DEPENDENCY * DIST_DEPENDENCY; //4*4

	const float imgScaleFactor = (float)img.cols / 640.f;
	const int imageSearchRadius = 50 * imgScaleFactor;
	const int imageSearchRadius2 = imageSearchRadius*imageSearchRadius;
	const int descriptorDissimilarityThreshold = 30;
	const int descriptorSimilarityThreshold = 60;

	const int height = img.rows;
	const int width = img.cols;
	const int depthWindowSize = 2;

	float inv_focallength = 1.f / 525.f / imgScaleFactor;
	const float centerX = 0.5f * width * imgScaleFactor - 0.5f;
	const float centerY = 0.5f * height * imgScaleFactor - 0.5f;

	Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
	transform.block<3,3>(0,0) = Eigen::Matrix3d( cloud.sensor_orientation_.cast<double>() );
	transform.block<4,1>(0,3) = cloud.sensor_origin_.cast<double>();

//	std::cout << transform << "\n";

	Eigen::Matrix4d jac = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d rot = Eigen::Matrix4d::Identity();
	rot.block<3,3>(0,0) = transform.block<3,3>(0,0);



    stopwatch_.reset();

	double delta_t;

	// extract ORB features (OpenCV 2.4.6)

	//CV_WRAP explicit ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31,
    //int firstLevel = 0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31 );

#if CV_VERSION_EPOCH > 2
	//cv::ORB is abstract
	cv::Ptr<cv::ORB> orb = cv::ORB::create( params_.numPointFeatures, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31 );
#else
	cv::Ptr<cv::ORB> orb = new cv::ORB( params_.numPointFeatures, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31 );
#endif

	const Eigen::Vector3d so = transform.block<3,1>(0,3);
	const Eigen::Quaterniond sori( transform.block<3,3>(0,0) );

	static std::vector< cv::KeyPoint > last_keypoints;
	static cv::Mat last_descriptors;
	static cv::Mat last_img;

	// in bytes
	unsigned int descriptorSize = orb->descriptorSize();

	std::vector< cv::KeyPoint > detectedKeypoints, keypoints;
	detectedKeypoints.reserve( params_.numPointFeatures );

	orb->detect( img, detectedKeypoints, cv::Mat() );

	std::cout << "detect: " << stopwatch_.getTimeSeconds() * 1000.0 << "\n";
	stopwatch_.reset();

	// bin detections in grid and restrict to specific number of detections per bin
	const size_t MaxFeaturesPerGridCell = params_.GridCellMax;
	const unsigned int rows = params_.GridRows;
	const unsigned int cols = params_.GridCols;
	const unsigned int colWidth = width / cols;
	const unsigned int rowHeight = height / rows;

	std::vector<std::vector<std::vector< unsigned int >>> keypointGrid;
	keypointGrid.reserve(params_.GridRows);

	for( unsigned int y = 0; y < params_.GridRows; y++ ) {
		keypointGrid[y].reserve(params_.GridCols);
		for( unsigned int x = 0; x < params_.GridCols; x++ )
			keypointGrid[y][x].reserve( 100 );

	}
	for( unsigned int i = 0; i < detectedKeypoints.size(); i++ ) {
		unsigned int gridx = std::max( 0.f, std::min( (float)params_.GridCols, detectedKeypoints[i].pt.x / colWidth ) );
		unsigned int gridy = std::max( 0.f, std::min( (float)params_.GridRows, detectedKeypoints[i].pt.y / rowHeight ) );
		keypointGrid[gridy][gridx].push_back( i );
	}


	KeypointComparator keypointComparator;
	keypointComparator.keypoints_ = &detectedKeypoints;

	std::vector< cv::KeyPoint > gridSampledKeypoints;
	gridSampledKeypoints.reserve( params_.numPointFeatures );
	for( unsigned int y = 0; y < params_.GridRows; y++ ) {
		for( unsigned int x = 0; x < params_.GridCols; x++ ) {
			// sort by response in descending order
			std::sort( keypointGrid[y][x].begin(), keypointGrid[y][x].end(), keypointComparator );
			// keep only specific number of strongest features
			keypointGrid[y][x].resize( std::min( keypointGrid[y][x].size(), (size_t) params_.GridCellMax ) );
			// add to new keypoint list
			for( unsigned int i = 0; i < keypointGrid[y][x].size(); i++ )
				gridSampledKeypoints.push_back( detectedKeypoints[keypointGrid[y][x][i]] );
		}
	}

	detectedKeypoints = gridSampledKeypoints;

	std::cout << "gridsample: " << stopwatch_.getTimeSeconds() * 1000.0 << "\n";
	stopwatch_.reset();

	cv::Mat detectedDescriptors;
	orb->compute( img, detectedKeypoints, detectedDescriptors );

	std::cout << "extract: " << stopwatch_.getTimeSeconds() * 1000.0 << "\n";




//	std::vector< std::vector< std::vector< doubleSort > > > imgGrid( rows );
//	for( int y = 0; y < rows; y++ )
//		imgGrid[ y ].resize( cols );
//
//	//add keypoints to imgGrid cell vectors
//	unsigned int NDetectedKeyPoints = detectedKeypoints.size();
//	for (unsigned int i = 0; i<NDetectedKeyPoints; i++)
//	{
//		cv::KeyPoint* kp = &(detectedKeypoints[i]);
//		doubleSort ds = { kp , i };
//
//		unsigned int colIndex = kp->pt.x / colWidth;
//		unsigned int rowIndex = kp->pt.y / rowHeight;
//		imgGrid[rowIndex][colIndex].push_back( ds );
//	}
//
//	// sort KeyPoints in grid cells
//	for (unsigned int row = 0; row < rows; row++)
//		for (unsigned int col = 0; col < cols; col++)
//			std::sort( imgGrid[row][col].begin(), imgGrid[row][col].end(), compareKeypoints );
//
//	// renew detectedKeypoints
//	cv::Mat detectedDescriptorsNew( detectedDescriptors.rows, detectedDescriptors.cols, CV_8UC1, 0.f );
//	std::vector< cv::KeyPoint > detectedKeypointsNew;
//	for (unsigned int row = 0; row < rows; row++)
//		for (unsigned int col = 0; col < cols; col++)
//		{
//			for (unsigned int keyNr = 0; keyNr < std::min(imgGrid[row][col].size(), MaxFeaturesPerGridCell ) ; keyNr++ )
//			{
//				int index = imgGrid[row][col][keyNr].index;
//				detectedKeypointsNew.push_back( detectedKeypoints[index] );
//
//				// Eintrag für Eintrag kopieren (geht nicht anders?)
//				for ( int descriptorElement = 0; descriptorElement < descriptorSize; descriptorElement++ )
//					detectedDescriptorsNew.data[ (detectedKeypointsNew.size()-1) * descriptorSize + descriptorElement ] = detectedDescriptors.data[ index * descriptorSize + descriptorElement ];
//			}
//		}
//	detectedDescriptors = detectedDescriptorsNew;
//	detectedKeypoints = detectedKeypointsNew;



	if( detectedKeypoints.size() < 3 ) {
		keypoints.clear();
	}
	else {

		cv::Mat tmpDescriptors = cv::Mat( features_.size() + detectedDescriptors.rows, detectedDescriptors.cols, CV_8UC1 );
		if( features_.size() > 0 )
			memcpy( &tmpDescriptors.data[ 0 ], &descriptors_.data[ 0 ], features_.size() * descriptorSize * sizeof( unsigned char ) );

		// build search index on image coordinates
		cv::Mat imageCoordinates = cv::Mat( (int)detectedKeypoints.size(), 2, CV_32FC1 ,0.f );
		for( size_t i = 0; i < detectedKeypoints.size(); i++ ) {
			imageCoordinates.at<float>( i, 0 ) = detectedKeypoints[i].pt.x;
			imageCoordinates.at<float>( i, 1 ) = detectedKeypoints[i].pt.y;
		}
		flann::Matrix< float > indexImageCoordinates( (float*)imageCoordinates.data, detectedKeypoints.size(), 2 );
		flann::Index< flann::L2_Simple< float > > image_index( indexImageCoordinates, flann::KDTreeSingleIndexParams() );
		image_index.buildIndex();

		std::vector< std::vector< int > > foundImageIndices;
		std::vector< std::vector< float > > foundImageDists;
		image_index.radiusSearch( indexImageCoordinates, foundImageIndices, foundImageDists, imageSearchRadius2, flann::SearchParams( 32, 0, false ) );

		flann::HammingPopcnt< unsigned char > hammingDist;
		keypoints.clear();
		keypoints.reserve( detectedKeypoints.size() );
		for( size_t i = 0; i < detectedKeypoints.size(); i++ ) {

			unsigned char* descriptor_i = &detectedDescriptors.data[ i * descriptorSize ];

			bool foundSimilarFeature = false;
			for( size_t j = 0; j < foundImageIndices[i].size(); j++ ) {

				size_t k = foundImageIndices[i][j];

				if( i == k )
					continue;

				if( k >= 0 && k < detectedKeypoints.size() ) {
					// compare descriptors.. results not sorted by descriptor similarity!
					const unsigned char* descriptor_k = &detectedDescriptors.data[ k * descriptorSize ];
//					if( hammingDist( descriptor_i, descriptor_k, descriptorSize ) < descriptorSimilarityThreshold ) {
					if( detectedKeypoints[i].response < detectedKeypoints[k].response && hammingDist( descriptor_i, descriptor_k, descriptorSize ) < descriptorDissimilarityThreshold ) {
//					if( detectedKeypoints[i].octave == detectedKeypoints[k].octave && hammingDist( descriptor_i, descriptor_k, descriptorSize ) < descriptorSimilarityThreshold ) {
						foundSimilarFeature = true;
						break;
					}
				}

			}

			if( !foundSimilarFeature ) {

				memcpy( &tmpDescriptors.data[ (features_.size() + keypoints.size()) * descriptorSize ], descriptor_i, descriptorSize * sizeof( unsigned char ) );
				keypoints.push_back( detectedKeypoints[ i ] );

			}
		}

		descriptors_ = tmpDescriptors.rowRange( 0, features_.size() + keypoints.size() ).clone();
	}

	std::cout << "sorted out " << detectedKeypoints.size()-keypoints.size() << " / " << detectedKeypoints.size() << " features for concurrent similarity in image position and descriptor\n";
	std::cout << "map has " << features_.size() + keypoints.size() << " features\n";




	// add features to the map
	size_t startFeatureIdx = features_.size();
	features_.resize( features_.size() + keypoints.size() );
	for( unsigned int i = 0; i < keypoints.size(); i++ ) {

		PointFeature& f = features_[startFeatureIdx+i];
		const cv::KeyPoint& kp = keypoints[i];

		// set inverse depth parametrization from point cloud
		bool hasDepth = false;
		double z = std::numeric_limits<double>::quiet_NaN();
		int minx = std::max( 0, (int)kp.pt.x - depthWindowSize );
		int maxx = std::min( width-1, (int)kp.pt.x + depthWindowSize );
		int miny = std::max( 0, (int)kp.pt.y - depthWindowSize );
		int maxy = std::min( height-1, (int)kp.pt.y + depthWindowSize );
		double sum_z = 0;
		double sum2_z = 0;
		double num_z = 0;
		for( int y = miny; y <= maxy; y++ ) {
			for( int x = minx; x <= maxx; x++ ) {

				int idx = y*width+x;
				const pcl::PointXYZRGB& p = cloud.points[idx];
				if( boost::math::isnan( p.x ) ) {
					continue;
				}

				Eigen::Vector4d pos( p.x, p.y, p.z, 1.0 );
				pos = (transform.inverse() * pos).eval();

				sum_z += pos(2);
				sum2_z += pos(2)*pos(2);
				num_z += 1.f;

				if( boost::math::isnan( z ) )
					z = pos(2);
				else
					z = std::min( z, pos(2) );
				hasDepth = true;

			}
		}

		// found depth?

		f.has_depth_ = hasDepth;

		if( hasDepth ) {

			float xi = inv_focallength * (kp.pt.x-centerX);
			float yi = inv_focallength * (kp.pt.y-centerY);

			f.image_pos_(0) = xi;
			f.image_pos_(1) = yi;

			f.pos_(0) = xi * z;
			f.pos_(1) = yi * z;
			f.pos_(2) = z;
			f.pos_(3) = 1.0;

            jac(0,0) = inv_focallength * z;
            jac(0,2) = xi;
            jac(1,1) = inv_focallength * z;
            jac(1,2) = yi;

			f.invzpos_(0) = kp.pt.x;
			f.invzpos_(1) = kp.pt.y;
			f.invzpos_(2) = 1.0 / z;
			f.invzpos_(3) = 1.0;


			// depth variance depends on depth..
			// propagate variance in depth to variance in inverse depth
			const double z4 = z*z*z*z;
			const double z_cov_emp = sum2_z / num_z - sum_z*sum_z / (num_z*num_z);

			f.cov_.setIdentity();
			f.cov_(0,0) = pixelNoise2; // in pixel^2
			f.cov_(1,1) = pixelNoise2; // in pixel^2
			f.cov_(2,2) = (depthNoiseScale2 * z4 + z_cov_emp);
            f.cov_ = (jac * f.cov_ * jac.transpose()).eval();

//			f.assoc_cov_.setIdentity();
//			f.assoc_cov_(0,0) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_cov_(1,1) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_cov_(2,2) = params_.depthNoiseAssocFactor * (depthNoiseScale2 * z4 + z_cov_emp);
//            f.assoc_cov_ = (jac * f.assoc_cov_ * jac.transpose()).eval();

			f.image_cov_.setIdentity();
			f.image_cov_ *= inv_focallength*inv_focallength*pixelNoise2;

//			f.image_assoc_cov_.setIdentity();
//			f.image_assoc_cov_ *= inv_focallength*inv_focallength*params_.pixelNoiseAssocFactor * pixelNoise2;



			f.invzcov_.setIdentity();
			f.invzcov_(0,0) = pixelNoise2; // in pixel^2
			f.invzcov_(1,1) = pixelNoise2; // in pixel^2
			f.invzcov_(2,2) = (depthNoiseScale2 * z4 + z_cov_emp) / z4;

//			f.assoc_invzcov_.setIdentity();
//			f.assoc_invzcov_(0,0) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_invzcov_(1,1) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_invzcov_(2,2) = params_.depthNoiseAssocFactor * (depthNoiseScale2 * z4 + z_cov_emp) / z4;



//			// add feature to corresponding surfels (at all resolutions)
//			spatialaggregate::OcTreeKey< float, NodeValue > poskey = octree_->getKey( mapPos(0), mapPos(1), mapPos(2) );
//
//			spatialaggregate::OcTreeNode< float, NodeValue >* node = octree_->root_;
//			while( node ) {
//
//				node->value_.addFeature( viewDirection, f );
//
//				node = node->children_[node->getOctant( poskey )];
//			}


		}
		else {

			// init to unknown depth
			z = 10.f;
			double z_cov_emp = z*z*0.25f;

			float xi = inv_focallength * (kp.pt.x-centerX);
			float yi = inv_focallength * (kp.pt.y-centerY);

			f.image_pos_(0) = xi;
			f.image_pos_(1) = yi;

			f.pos_(0) = xi * z;
			f.pos_(1) = yi * z;
			f.pos_(2) = z;
			f.pos_(3) = 1.0;

            jac(0,0) = inv_focallength * z;
            jac(0,2) = xi;
            jac(1,1) = inv_focallength * z;
            jac(1,2) = yi;


			f.invzpos_(0) = kp.pt.x;
			f.invzpos_(1) = kp.pt.y;
			f.invzpos_(2) = 1.0 / z;
			f.invzpos_(3) = 1.0;



			// covariance depends on depth..
			// propagate variance in depth to variance in inverse depth
			// add uncertainty from depth estimate in local image region
			const double z4 = z*z*z*z;

			f.cov_.setIdentity();
			f.cov_(0,0) = pixelNoise2; // in pixel^2
			f.cov_(1,1) = pixelNoise2; // in pixel^2
			f.cov_(2,2) = depthNoiseScale2 * z4 + z_cov_emp;
            f.cov_ = (jac * f.cov_ * jac.transpose()).eval();

//			f.assoc_cov_.setIdentity();
//			f.assoc_cov_(0,0) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_cov_(1,1) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_cov_(2,2) = params_.depthNoiseAssocFactor * (depthNoiseScale2 * z4 + z_cov_emp);
//            f.assoc_cov_ = (jac * f.assoc_cov_ * jac.transpose()).eval();

			f.image_cov_.setIdentity();
			f.image_cov_ *= inv_focallength*inv_focallength*pixelNoise2;

			f.image_assoc_cov_.setIdentity();
			f.image_assoc_cov_ *= inv_focallength*inv_focallength*params_.pixelNoiseAssocFactor * pixelNoise2;


			f.invzcov_.setIdentity();
			f.invzcov_(0,0) = pixelNoise2; // in pixel^2
			f.invzcov_(1,1) = pixelNoise2; // in pixel^2
			f.invzcov_(2,2) = depthNoiseScale2;


//			f.assoc_invzcov_.setIdentity();
//			f.assoc_invzcov_(0,0) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_invzcov_(1,1) = params_.pixelNoiseAssocFactor * pixelNoise2;
//			f.assoc_invzcov_(2,2) = params_.depthNoiseAssocFactor * (depthNoiseScale2 * z4 + z_cov_emp) / z4;


			// features without depth can only be added to the root node, they are valid for all nodes and will be associated afterwards
//			octree_->root_->value_.addFeature( viewDirection, f );

		}

		f.pos_ = (transform * f.pos_).eval();
		f.cov_ = (rot * f.cov_ * rot.transpose()).eval();
//		f.assoc_cov_ = (rot * f.assoc_cov_ * rot.transpose()).eval();

		f.invzinvcov_ = f.invzcov_.inverse();

		f.origin_ = so;
		f.orientation_ = sori;



	}

	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;
    std::cerr << "feature extraction took " << delta_t << "ms.\n";


	// build LSH search index for this image using LSH implementation in FLANN 1.7.1
    stopwatch_.reset();
    flann::Matrix< unsigned char > indexDescriptors( descriptors_.data, keypoints.size(), orb->descriptorSize() );
	lsh_index_ = boost::shared_ptr< flann::Index< flann::HammingPopcnt< unsigned char > > >( new flann::Index< flann::HammingPopcnt< unsigned char > >( indexDescriptors, flann::LshIndexParams( 2, 20, 2 ) ) );
	lsh_index_->buildIndex();
	delta_t = stopwatch_.getTimeSeconds() * 1000.0f;
    std::cerr << "lsh search index construction took " << delta_t << "ms.\n";




    if( params_.debugPointFeatures ) {
		cv::Mat outimg;
		cv::drawKeypoints( img, keypoints, outimg );
		cv::imshow( "keypoints", outimg );
		cv::waitKey(1);
    }




}



void MultiResolutionSurfelMap::getImage( cv::Mat& img, const Eigen::Vector3d& viewPosition ) {

	int h = imageAllocator_->height;
	int w = imageAllocator_->width;

	img = cv::Mat( h, w, CV_8UC3, 0.f );

	spatialaggregate::OcTreeNode< float, NodeValue >** nodeImgPtr = &imageAllocator_->node_image_[0];

	cv::Vec3b v;
	for( int y = 0; y < h; y++ ) {

		for( int x = 0; x < w; x++ ) {

			if( *nodeImgPtr ) {

				float rf = 0, gf = 0, bf = 0;
				Eigen::Vector3d viewDirection = (*nodeImgPtr)->getPosition().block<3,1>(0,0).cast<double>() - viewPosition;
				viewDirection.normalize();

				Surfel* surfel = (*nodeImgPtr)->value_.getSurfel( viewDirection );
				Eigen::Matrix< double, 6, 1 > vec = (*nodeImgPtr)->value_.getSurfel( viewDirection )->mean_;

				const float L = vec( 3 );
				const float alpha = vec( 4 );
				const float beta = vec( 5 );

				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				v[0] = b;
				v[1] = g;
				v[2] = r;

			}
			else {

				v[0] = 0;
				v[1] = 0;
				v[2] = 0;

			}

			img.at< cv::Vec3b >(y,x) = v;

			nodeImgPtr++;

		}
	}

}


inline bool MultiResolutionSurfelMap::splitCriterion( spatialaggregate::OcTreeNode< float, NodeValue >* oldLeaf, spatialaggregate::OcTreeNode< float, NodeValue >* newLeaf ) {

	return true;

}

void MultiResolutionSurfelMap::findImageBorderPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, std::vector< int >& indices ) {

	// determine first image points from the borders that are not nan

	// horizontally
	for ( unsigned int y = 0; y < cloud.height; y++ ) {

		for ( unsigned int x = 0; x < cloud.width; x++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if ( boost::math::isnan( px ) || boost::math::isinf( px ) )
				continue;

			if ( boost::math::isnan( py ) || boost::math::isinf( py ) )
				continue;

			if ( boost::math::isnan( pz ) || boost::math::isinf( pz ) )
				continue;

			indices.push_back( idx );
			break;

		}

		for ( int x = cloud.width - 1; x >= 0; x-- ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if ( boost::math::isnan( px ) || boost::math::isinf( px ) )
				continue;

			if ( boost::math::isnan( py ) || boost::math::isinf( py ) )
				continue;

			if ( boost::math::isnan( pz ) || boost::math::isinf( pz ) )
				continue;

			indices.push_back( idx );
			break;

		}

	}

	// vertically
	for ( unsigned int x = 0; x < cloud.width; x++ ) {

		for ( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if ( boost::math::isnan( px ) || boost::math::isinf( px ) )
				continue;

			if ( boost::math::isnan( py ) || boost::math::isinf( py ) )
				continue;

			if ( boost::math::isnan( pz ) || boost::math::isinf( pz ) )
				continue;

			indices.push_back( idx );
			break;

		}

		for ( int y = cloud.height - 1; y >= 0; y-- ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if ( boost::math::isnan( px ) || boost::math::isinf( px ) )
				continue;

			if ( boost::math::isnan( py ) || boost::math::isinf( py ) )
				continue;

			if ( boost::math::isnan( pz ) || boost::math::isinf( pz ) )
				continue;

			indices.push_back( idx );
			break;

		}

	}

}


void MultiResolutionSurfelMap::findVirtualBorderPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, std::vector< int >& indices ) {

	// detect background points at depth jumps
	// determine first image points from the borders that are not nan => use 0 depth beyond borders

	const float depthJumpRatio = 0.9f*0.9f;
	const float invDepthJumpRatio = 1.f/depthJumpRatio;

	indices.reserve( cloud.points.size() );

	// horizontally
	int idx = -1;
	for ( unsigned int y = 0; y < cloud.height; y++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int x = 0; x < cloud.width; x++ ) {

			idx++;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( boost::math::isnan( px ) ) {
				continue;
			}

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

		if( lastIdx >= 0 )
			indices.push_back( lastIdx );

	}


	// vertically
	for ( unsigned int x = 0; x < cloud.width; x++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( boost::math::isnan( px ) )
				continue;

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

		if( lastIdx >= 0 )
			indices.push_back( lastIdx );

	}

}


void MultiResolutionSurfelMap::findForegroundBorderPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, std::vector< int >& indices ) {

	// detect foreground points at depth jumps
	// determine first image points from the borders that are not nan => use 0 depth beyond borders

	const float depthJumpRatio = 0.9f*0.9f;
	const float invDepthJumpRatio = 1.f/depthJumpRatio;

	indices.clear();
	indices.reserve( cloud.points.size() );

	// horizontally
	int idx = -1;
	for ( unsigned int y = 0; y < cloud.height; y++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int x = 0; x < cloud.width; x++ ) {

			idx++;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( boost::math::isnan( px ) ) {
				continue;
			}

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}


	// vertically
	for ( unsigned int x = 0; x < cloud.width; x++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( boost::math::isnan( px ) )
				continue;

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}

}


void MultiResolutionSurfelMap::findContourPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, std::vector< int >& indices ) {

	// detect foreground points at depth jumps
	// determine first image points from the borders that are not nan => use 0 depth beyond borders

	const float depthJumpRatio = 0.95f*0.95f;
	const float invDepthJumpRatio = 1.f/depthJumpRatio;

	indices.clear();
	indices.reserve( cloud.points.size() );

	// horizontally
	int idx = -1;
	for ( unsigned int y = 0; y < cloud.height; y++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int x = 0; x < cloud.width; x++ ) {

			idx++;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( boost::math::isnan( px ) ) {
				continue;
			}

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}


	// vertically
	for ( unsigned int x = 0; x < cloud.width; x++ ) {

		float lastDepth2 = 0.0;
		int lastIdx = -1;

		for ( unsigned int y = 0; y < cloud.height; y++ ) {

			int idx = y * cloud.width + x;

			// if not nan, push back and break
			const pcl::PointXYZRGB& p = cloud.points[idx];
			const float px = p.x;
			const float py = p.y;
			const float pz = p.z;

			if( boost::math::isnan( px ) )
				continue;

			// check for depth jumps
			float depth2 = px*px+py*py+pz*pz;
			float ratio = lastDepth2 / depth2;
			if( ratio < depthJumpRatio ) {
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
				indices.push_back( idx );
			}
			if( ratio > invDepthJumpRatio ) {
				indices.push_back( idx );
				if( lastIdx != -1 )
					indices.push_back( lastIdx );
			}

			lastIdx = idx;

			lastDepth2 = depth2;

		}

	}

}


void MultiResolutionSurfelMap::clearAtPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and remove surfels
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( boost::math::isnan( x ) || boost::math::isinf( x ) )
			continue;

		if ( boost::math::isnan( y ) || boost::math::isinf( y ) )
			continue;

		if ( boost::math::isnan( z ) || boost::math::isinf( z ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		// traverse tree and clear all surfels that include this points
		spatialaggregate::OcTreeKey< float, NodeValue > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_;
		while ( n ) {


			for( unsigned int k = 0; k < NodeValue::num_surfels_; k++ ) {
				const double dist = viewDirection.dot( n->value_.surfels_[k].initial_view_dir_ );
				if( dist > max_dist ) {
					n->value_.surfels_[k].clear();
				}
			}

			n = n->children_[n->getOctant( position )];
		}

	}

}

void MultiResolutionSurfelMap::markNoUpdateAtPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and remove surfels
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( boost::math::isnan( x ) || boost::math::isinf( x ) )
			continue;

		if ( boost::math::isnan( y ) || boost::math::isinf( y ) )
			continue;

		if ( boost::math::isnan( z ) || boost::math::isinf( z ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		// traverse tree and clear all surfels that include this points
		spatialaggregate::OcTreeKey< float, NodeValue > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_;
		while ( n ) {

			for( unsigned int k = 0; k < NodeValue::num_surfels_; k++ ) {
				const double dist = viewDirection.dot( n->value_.surfels_[k].initial_view_dir_ );
				if( dist > max_dist ) {
					n->value_.surfels_[k].applyUpdate_ = false;
				}
			}

			n = n->children_[n->getOctant( position )];

		}

	}

}


void MultiResolutionSurfelMap::markBorderAtPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( boost::math::isnan( x ) || boost::math::isinf( x ) )
			continue;

		if ( boost::math::isnan( y ) || boost::math::isinf( y ) )
			continue;

		if ( boost::math::isnan( z ) || boost::math::isinf( z ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		spatialaggregate::OcTreeKey< float, NodeValue > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_;
		while ( n ) {

			n->value_.border_ = true;

			n = n->children_[n->getOctant( position )];

		}

	}

}


struct MarkBorderInfo {
	Eigen::Vector3d viewpoint;
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


void markBorderFromViewpointFunction( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* current, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* next, void* data ) {

	MarkBorderInfo* info = (MarkBorderInfo*) data;

//	current->value_.border_ = false;

	for( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {

		const MultiResolutionSurfelMap::Surfel& surfel = current->value_.surfels_[i];

		if( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		Eigen::Vector3d viewDirection = info->viewpoint - surfel.mean_.block<3,1>(0,0);
		viewDirection.normalize();

		double cangle = viewDirection.dot( surfel.normal_ );

		if( cangle < 0.0 ) {
			current->value_.border_ = true;
//			for( unsigned int n = 0; n < 27; n++ )
//				if( current->neighbors_[n] )
//					current->neighbors_[n]->value_.border_ = true;
		}

	}

}


void MultiResolutionSurfelMap::markBorderFromViewpoint( const Eigen::Vector3d& viewpoint ) {

	MarkBorderInfo info;
	info.viewpoint = viewpoint;

	clearBorderFlag();
	octree_->root_->sweepDown( &info, &markBorderFromViewpointFunction );

}


inline void MultiResolutionSurfelMap::clearBorderFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	current->value_.border_ = false;

}

void MultiResolutionSurfelMap::clearBorderFlag() {

	octree_->root_->sweepDown( NULL, &clearBorderFlagFunction );

}


void MultiResolutionSurfelMap::clearUpdateSurfelsAtPoints( const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices ) {

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	const double max_dist = MAX_VIEWDIR_DIST;

	// go through the point cloud and remove surfels
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( boost::math::isnan( x ) )
			continue;

		Eigen::Matrix< double, 3, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;

		Eigen::Vector3d viewDirection = pos - sensorOrigin;
		const double viewDistance = viewDirection.norm();

		if ( viewDistance < 1e-10 )
			continue;

		viewDirection = viewDirection / viewDistance;

		// traverse tree and clear all surfels that include this points
		spatialaggregate::OcTreeKey< float, NodeValue > position = octree_->getKey( p.getVector4fMap() );
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_;
		while ( n ) {

			for( unsigned int k = 0; k < NodeValue::num_surfels_; k++ ) {
				const double dist = viewDirection.dot( n->value_.surfels_[k].initial_view_dir_ );
				if( dist > max_dist ) {
					if( !n->value_.surfels_[k].up_to_date_ ) {
						n->value_.surfels_[k].clear();
					}
				}
			}

			n = n->children_[n->getOctant( position )];

		}

	}

}



void MultiResolutionSurfelMap::markUpdateAllSurfels() {

	octree_->root_->sweepDown( NULL, &markUpdateAllSurfelsFunction );

}

inline void MultiResolutionSurfelMap::markUpdateAllSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ )
		current->value_.surfels_[i].applyUpdate_ = true;

}




void MultiResolutionSurfelMap::evaluateSurfels() {

	octree_->root_->sweepDown( NULL, &evaluateNormalsFunction );
	octree_->root_->sweepDown( NULL, &evaluateSurfelsFunction );

}


void MultiResolutionSurfelMap::unevaluateSurfels() {

	octree_->root_->sweepDown( NULL, &unevaluateSurfelsFunction );

}


void MultiResolutionSurfelMap::setApplyUpdate( bool v ) {

	octree_->root_->sweepDown( &v, &setApplyUpdateFunction );

}

void MultiResolutionSurfelMap::setUpToDate( bool v ) {

	octree_->root_->sweepDown( &v, &setUpToDateFunction );

}


void MultiResolutionSurfelMap::clearUnstableSurfels() {

	octree_->root_->sweepDown( NULL, &clearUnstableSurfelsFunction );

}




void MultiResolutionSurfelMap::clearAssociatedFlag() {

	octree_->root_->sweepDown( NULL, &clearAssociatedFlagFunction );

}


void MultiResolutionSurfelMap::clearSeenThroughFlag() {

	octree_->root_->sweepDown( NULL, &clearSeenThroughFlagFunction );

}


void MultiResolutionSurfelMap::distributeAssociatedFlag() {

	octree_->root_->sweepDown( NULL, &distributeAssociatedFlagFunction );

}

void MultiResolutionSurfelMap::clearAssociations() {

	octree_->root_->sweepDown( NULL, &clearAssociationsFunction );

}

inline void MultiResolutionSurfelMap::clearAssociationsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	if( current->value_.associated_ != -1 )
		current->value_.associated_ = 1;
	current->value_.association_ = NULL;

	for( unsigned int i = 0; i < NodeValue::num_surfels_; i++ )
		current->value_.surfels_[i].seenThrough_ = false;
}


inline void MultiResolutionSurfelMap::clearSeenThroughFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	for( unsigned int i = 0; i < NodeValue::num_surfels_; i++ )
		current->value_.surfels_[i].seenThrough_ = false;
}


bool MultiResolutionSurfelMap::pointInForeground( const Eigen::Vector3f& position, const cv::Mat& image_depth, const cv::Point2f imagePoint, float scale, float jumpThreshold ) {

	float queryDepth = position.norm();

	int scale05 = ceil( 0.5f * scale );

	cv::Rect r;
	r.x = (int) floor( imagePoint.x - scale05 );
	r.y = (int) floor( imagePoint.y - scale05 );
	r.width = 2 * scale05;
	r.height = 2 * scale05;

	if ( r.x < 0 ) {
		r.width += r.x;
		r.x = 0;
	}

	if ( r.y < 0 ) {
		r.height += r.y;
		r.y = 0;
	}

	if ( r.x + r.width > image_depth.cols )
		r.width = image_depth.cols - r.x;

	if ( r.y + r.height > image_depth.rows )
		r.height = image_depth.rows - r.y;

	cv::Mat patch = image_depth( r );

	// find correponding point for query point in image
	float bestDist = 1e10f;
	int bestX = -1, bestY = -1;
	for ( int y = 0; y < patch.rows; y++ ) {
		for ( int x = 0; x < patch.cols; x++ ) {
			const float depth = patch.at< float > ( y, x );
			if ( !boost::math::isnan( depth ) ) {
				float dist = fabsf( queryDepth - depth );
				if ( dist < bestDist ) {
					bestDist = dist;
					bestX = x;
					bestY = y;
				}
			}

		}
	}

	// find depth jumps to the foreground in horizontal, vertical, and diagonal directions
	//	cv::Mat img_show = image_depth.clone();

	for ( int dy = -1; dy <= 1; dy++ ) {
		for ( int dx = -1; dx <= 1; dx++ ) {

			if ( dx == 0 && dy == 0 )
				continue;

			float trackedDepth = queryDepth;
			for ( int y = bestY + dy, x = bestX + dx; y >= 0 && y < patch.rows && x >= 0 && x < patch.cols; y += dy, x += dx ) {

				const float depth = patch.at< float > ( y, x );
				//				img_show.at<float>(r.y+y,r.x+x) = 0.f;
				if ( !boost::math::isnan( depth ) ) {

					if ( trackedDepth - depth > jumpThreshold ) {
						return false;
					}

					trackedDepth = depth;

				}

			}

		}
	}

	return true;
}


void MultiResolutionSurfelMap::buildShapeTextureFeatures() {

	octree_->root_->sweepDown( NULL, &buildSimpleShapeTextureFeatureFunction );
	octree_->root_->sweepDown( NULL, &buildAgglomeratedShapeTextureFeatureFunction );

}

inline void MultiResolutionSurfelMap::buildSimpleShapeTextureFeatureFunction( spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data ) {

	for( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {

		current->value_.surfels_[i].simple_shape_texture_features_.initialize();

		if( current->value_.surfels_[i].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		current->value_.surfels_[i].simple_shape_texture_features_.add( current->value_.surfels_[i].mean_, current->value_.surfels_[i].mean_, current->value_.surfels_[i].normal_, current->value_.surfels_[i].normal_, current->value_.surfels_[i].num_points_ );
//		current->value_.surfels_[i].simple_shape_texture_features_.add( &current->value_.surfels_[i], &current->value_.surfels_[i], current->value_.surfels_[i].num_points_ );

		for( unsigned int n = 0; n < 27; n++ ) {

			if( n == 13 ) // pointer to this node
				continue;

			if( current->neighbors_[n] ) {

				if( current->neighbors_[n]->value_.surfels_[i].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
					continue;

				current->value_.surfels_[i].simple_shape_texture_features_.add( current->value_.surfels_[i].mean_, current->neighbors_[n]->value_.surfels_[i].mean_, current->value_.surfels_[i].normal_, current->neighbors_[n]->value_.surfels_[i].normal_, current->neighbors_[n]->value_.surfels_[i].num_points_ );
//				current->value_.surfels_[i].simple_shape_texture_features_.add( &current->value_.surfels_[i], &current->neighbors_[n]->value_.surfels_[i], current->neighbors_[n]->value_.surfels_[i].num_points_ );

			}

		}

	}

}


inline void MultiResolutionSurfelMap::buildAgglomeratedShapeTextureFeatureFunction( spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data ) {

	const float neighborFactor = 0.1f;

	for( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {

		current->value_.surfels_[i].agglomerated_shape_texture_features_ = current->value_.surfels_[i].simple_shape_texture_features_;

		if( current->value_.surfels_[i].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		for( unsigned int n = 0; n < 27; n++ ) {

			if( n == 13 ) // pointer to this node
				continue;

			if( current->neighbors_[n] ) {

				if( current->neighbors_[n]->value_.surfels_[i].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
					continue;
				}

				current->value_.surfels_[i].agglomerated_shape_texture_features_.add( current->neighbors_[n]->value_.surfels_[i].simple_shape_texture_features_, current->neighbors_[n]->value_.surfels_[i].simple_shape_texture_features_.num_points_ * neighborFactor );

			}

		}

		if( current->value_.surfels_[i].agglomerated_shape_texture_features_.num_points_ > 0.5f ) {
			float inv_num = 1.f / current->value_.surfels_[i].agglomerated_shape_texture_features_.num_points_;
			current->value_.surfels_[i].agglomerated_shape_texture_features_.shape_ *= inv_num;
			current->value_.surfels_[i].agglomerated_shape_texture_features_.texture_ *= inv_num;
		}

		current->value_.surfels_[i].agglomerated_shape_texture_features_.num_points_ = 1.f;

	}

}

void MultiResolutionSurfelMap::clearAssociationDist() {
	octree_->root_->sweepDown( NULL, &clearAssociationDistFunction );
}



inline void MultiResolutionSurfelMap::clearAssociationDistFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data) {
	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {
		current->value_.surfels_[i].assocDist_ = std::numeric_limits<float>::max();
	}
}


inline void MultiResolutionSurfelMap::setApplyUpdateFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	bool v = *((bool*) data);
	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {
		if( current->value_.surfels_[i].num_points_ >= MultiResolutionSurfelMap::Surfel::min_points_ ) {
			current->value_.surfels_[i].applyUpdate_ = v;
		}
	}
}

inline void MultiResolutionSurfelMap::setUpToDateFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	bool v = *((bool*) data);
	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {
		current->value_.surfels_[i].up_to_date_ = v;
	}
}


inline void MultiResolutionSurfelMap::clearUnstableSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {
		if( current->value_.surfels_[i].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ ) {
			// reinitialize
			current->value_.surfels_[i].up_to_date_ = false;
			current->value_.surfels_[i].mean_.setZero();
			current->value_.surfels_[i].cov_.setZero();
			current->value_.surfels_[i].num_points_ = 0;
//			current->value_.surfels_[i].became_robust_ = false;
			current->value_.surfels_[i].applyUpdate_ = true;
		}
	}
}






inline void MultiResolutionSurfelMap::evaluateNormalsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	current->value_.evaluateNormals( current );
}

inline void MultiResolutionSurfelMap::evaluateSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	current->value_.evaluateSurfels();
}


inline void MultiResolutionSurfelMap::unevaluateSurfelsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	current->value_.unevaluateSurfels();
}


inline void MultiResolutionSurfelMap::clearAssociatedFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {
	if( current->value_.associated_ != -1 )
		current->value_.associated_ = 1;
}


inline void MultiResolutionSurfelMap::distributeAssociatedFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	for( unsigned int n = 0; n < 27; n++ ) {

		if( current->neighbors_[n] && current->neighbors_[n]->value_.associated_ == 0 ) {
			current->neighbors_[n]->value_.associated_ = 2;
		}

	}

}



std::vector< unsigned int > MultiResolutionSurfelMap::findInliers( const std::vector< unsigned int >& indices, const pcl::PointCloud<pcl::PointXYZRGB>& cloud, int maxDepth ) {

	std::vector< unsigned int > inliers;
	inliers.reserve( indices.size() );

	const float max_mahal_dist = 12.59f;

	const double inv_255 = 1.0 / 255.0;
	const float sqrt305 = 0.5f*sqrtf(3.f);

	Eigen::Vector3d sensorOrigin;
	for ( int i = 0; i < 3; i++ )
		sensorOrigin( i ) = cloud.sensor_origin_( i );

	// project each point into map and find inliers
	// go through the point cloud and add point information to map
	for ( unsigned int i = 0; i < indices.size(); i++ ) {

		const pcl::PointXYZRGB& p = cloud.points[indices[i]];
		const float x = p.x;
		const float y = p.y;
		const float z = p.z;

		if ( boost::math::isnan( x ) )
			continue;

		float rgbf = p.rgb;
		unsigned int rgb = * ( reinterpret_cast< unsigned int* > ( &rgbf ) );
		unsigned int r = ( ( rgb & 0x00FF0000 ) >> 16 );
		unsigned int g = ( ( rgb & 0x0000FF00 ) >> 8 );
		unsigned int b = ( rgb & 0x000000FF );

		// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
		float rf = inv_255*r, gf = inv_255*g, bf = inv_255*b;

		// RGB to L-alpha-beta:
		// normalize RGB to [0,1]
		// M := max( R, G, B )
		// m := min( R, G, B )
		// L := 0.5 ( M + m )
		// alpha := 0.5 ( 2R - G - B )
		// beta := 0.5 sqrt(3) ( G - B )
		float L = 0.5f * ( std::max( std::max( rf, gf ), bf ) + std::min( std::min( rf, gf ), bf ) );
		float alpha = 0.5f * ( 2.f*rf - gf - bf );
		float beta = sqrt305 * (gf-bf);


		Eigen::Matrix< double, 6, 1 > pos;
		pos( 0 ) = p.x;
		pos( 1 ) = p.y;
		pos( 2 ) = p.z;
		pos( 3 ) = L;
		pos( 4 ) = alpha;
		pos( 5 ) = beta;


		Eigen::Vector3d viewDirection = pos.block< 3, 1 > ( 0, 0 ) - sensorOrigin;
		viewDirection.normalize();

		Eigen::Vector4f pos4f = pos.block<4,1>(0,0).cast<float>();

		// lookup node for point
		spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_->findRepresentative( pos4f, maxDepth );

		Surfel* surfel = n->value_.getSurfel( viewDirection );
		if( surfel->num_points_ > MultiResolutionSurfelMap::Surfel::min_points_ ) {

			// inlier? check mahalanobis distance
			Eigen::Matrix< double, 6, 6 > invcov = surfel->cov_.inverse();
			Eigen::Matrix< double, 6, 1 > diff = surfel->mean_.block<6,1>(0,0) - pos;

			if( diff.dot( invcov * diff ) < max_mahal_dist ) {
				inliers.push_back( i );
			}

		}


	}

	return inliers;
}




struct Visualize3DColorDistributionInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	int viewDir, depth;
	bool random;
};


void MultiResolutionSurfelMap::visualize3DColorDistribution( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool random ) {

	Visualize3DColorDistributionInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.random = random;

	octree_->root_->sweepDown( &info, &visualize3DColorDistributionFunction );

	cloudPtr->width = cloudPtr->points.size();
	cloudPtr->height = 1;

}

inline void MultiResolutionSurfelMap::visualize3DColorDistributionFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	Visualize3DColorDistributionInfo* info = (Visualize3DColorDistributionInfo*) data;

	if( (info->depth == -1 && current->type_ == spatialaggregate::OCTREE_BRANCHING_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;

//	if( current->depth_ < 10 )
//		return;

//	std::cout << current->resolution() << "\n";

	Eigen::Matrix< float, 4, 1 > minPos = current->getMinPosition();
	Eigen::Matrix< float, 4, 1 > maxPos = current->getMaxPosition();


	// generate markers for histogram surfels
	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		if( info->random ) {

			// samples N points from the normal distribution in mean and cov...
			unsigned int N = 100;

			// cholesky decomposition
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::LLT< Eigen::Matrix< double, 6, 6 > > chol = cov.llt();


			for ( unsigned int j = 0; j < N; j++ ) {

				Eigen::Matrix< double, 6, 1 > vec;
				for ( unsigned int k = 0; k < 6; k++ )
					vec( k ) = gsl_ran_gaussian( r, 1.0 );

				vec( 3 ) = vec( 4 ) = vec( 5 ) = 0.0;

				vec = ( chol.matrixL() * vec ).eval();

				vec = ( surfel.mean_ + vec ).eval();

				pcl::PointXYZRGB p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				if( !current->inRegion( spatialaggregate::OcTreeKey< float, NodeValue >( vec(0), vec(1), vec(2), current->tree_ ) ) )
					continue;

				const float L = vec( 3 );
				const float alpha = vec( 4 );
				const float beta = vec( 5 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );


				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

				info->cloudPtr->points.push_back( p );

			}

		}
		else {

			// PCA projection
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::Matrix< double, 3, 3 > cov3_inv = cov.block<3,3>(0,0).inverse();

			Eigen::Matrix< double, 3, 1> eigen_values_;
			Eigen::Matrix< double, 3, 3> eigen_vectors_;

			pcl::eigen33(Eigen::Matrix3d(cov.block<3,3>(0,0)), eigen_vectors_, eigen_values_);

			eigen_values_(0) = 0.0;
			eigen_values_(1) = sqrt( eigen_values_(1) );
			eigen_values_(2) = sqrt( eigen_values_(2) );

			Eigen::Matrix< double, 3, 3 > L = eigen_vectors_ * eigen_values_.asDiagonal();

			std::vector< Eigen::Matrix< double, 3, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 3, 1 > > > vecs;

			Eigen::Matrix< double, 3, 1 > v;

//			// sample a dense 2d grid
//			const int steps = 12;
//			const double inv_steps = 1.0 / (double)steps;
//			for( int y = -steps; y <= steps; y++ ) {
//				for( int x = -steps; x <= steps; x++ ) {
//					v(0) = 0.0; v(1) = 2.0 * (double)x * inv_steps; v(2) = 2.0 * (double)y * inv_steps;
//					vecs.push_back( v );
//				}
//			}

			v(0) =  0.0; v(1) =  -1.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  -1.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  -1.0; v(2) =  1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  1.0;
			vecs.push_back( v );


//			v(0) =  0.0; v(1) =  -0.5; v(2) =  -0.5;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  -0.5; v(2) =  0.0;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  -0.5; v(2) =  0.5;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.0; v(2) =  -0.5;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.0; v(2) =  0.0;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.0; v(2) =  0.5;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.5; v(2) =  -0.5;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.5; v(2) =  0.0;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.5; v(2) =  0.5;
//			vecs.push_back( v );
//
//
//			v(0) =  0.0; v(1) =  -0.8; v(2) =  -0.8;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  -0.8; v(2) =  0.0;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  -0.8; v(2) =  0.8;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.0; v(2) =  -0.8;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.0; v(2) =  0.0;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.0; v(2) =  0.8;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.8; v(2) =  -0.8;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.8; v(2) =  0.0;
//			vecs.push_back( v );
//
//			v(0) =  0.0; v(1) =  0.8; v(2) =  0.8;
//			vecs.push_back( v );






			for( unsigned int k = 0; k < vecs.size(); k++ ) {

				Eigen::Matrix< double, 3, 1 > vec = 1.1*vecs[k];

				vec = ( L * vec ).eval();

				// check if position is likely
				if( -0.5 * vec.dot( cov3_inv * vec ) < -2.5  )
					continue;


				vec = ( surfel.mean_.block<3,1>(0,0) + vec ).eval();


				pcl::PointXYZRGB p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				// get color mean conditioned on position
				Eigen::Matrix< double, 3, 1 > cvec = surfel.mean_.block<3,1>(3,0) + cov.block<3,3>(3,0) * cov3_inv * ( vec - surfel.mean_.block<3,1>(0,0) );

				const float L = cvec( 0 );
				const float alpha = cvec( 1 );
				const float beta = cvec( 2 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );


				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

				info->cloudPtr->points.push_back( p );

			}

//			for( unsigned int j = 0; j < 2; j++ ) {
//				for( unsigned int k = 0; k < 2; k++ ) {
//					for( unsigned int l = 0; l < 2; l++ ) {
//
//						spatialaggregate::OcTreeNode< float, NodeValue >* neighbor = current->getNeighbor( j, k, l );
//
//						if( !neighbor )
//							continue;
//
////						if( neighbor->value_.surfels_[i].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
////							continue;
//
//						Surfel surfel;
//						surfel.num_points_ = current->value_.surfels_[i].num_points_ + neighbor->value_.surfels_[i].num_points_;
//						surfel.mean_ = current->value_.surfels_[i].num_points_ * current->value_.surfels_[i].mean_ + neighbor->value_.surfels_[i].num_points_ * neighbor->value_.surfels_[i].mean_;
//
//						surfel.mean_ /= surfel.num_points_;
//
//						pcl::PointXYZRGB p;
//						p.x = surfel.mean_( 0 );
//						p.y = surfel.mean_( 1 );
//						p.z = surfel.mean_( 2 );
//
//						const float L = surfel.mean_( 3 );
//						const float alpha = surfel.mean_( 4 );
//						const float beta = surfel.mean_( 5 );
//
//						float rf = 0, gf = 0, bf = 0;
//						convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );
//
//						int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
//						int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
//						int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );
//
//						int rgb = ( r << 16 ) + ( g << 8 ) + b;
//						p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );
//
//						info->cloudPtr->points.push_back( p );
//
//					}
//				}
//			}

		}
	}

}


struct Visualize3DColorDistributionWithNormalInfo {
	pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr;
	int viewDir, depth;
	bool random;
	int numSamples;
};


void MultiResolutionSurfelMap::visualize3DColorDistributionWithNormals( pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr, int depth, int viewDir, bool random, int numSamples ) {

	Visualize3DColorDistributionWithNormalInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.random = random;
	info.numSamples = numSamples;

	octree_->root_->sweepDown( &info, &visualize3DColorDistributionWithNormalsFunction );

}


void MultiResolutionSurfelMap::visualize3DColorDistributionWithNormalsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	Visualize3DColorDistributionWithNormalInfo* info = (Visualize3DColorDistributionWithNormalInfo*) data;

	if( (info->depth == -1 && current->type_ == spatialaggregate::OCTREE_BRANCHING_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;

	Eigen::Matrix< float, 4, 1 > minPos = current->getMinPosition();
	Eigen::Matrix< float, 4, 1 > maxPos = current->getMaxPosition();


	// generate markers for histogram surfels
	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		if( info->random ) {

			// samples N points from the normal distribution in mean and cov...
			unsigned int N = info->numSamples;

			// cholesky decomposition
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::LLT< Eigen::Matrix< double, 6, 6 > > chol = cov.llt();


			for ( unsigned int j = 0; j < N; j++ ) {

				Eigen::Matrix< double, 6, 1 > vec;
				for ( unsigned int k = 0; k < 6; k++ )
					vec( k ) = gsl_ran_gaussian( r, 1.0 );


				vec( 3 ) = vec( 4 ) = vec( 5 ) = 0.0;

				vec = ( chol.matrixL() * vec ).eval();

				vec = ( surfel.mean_ + vec ).eval();

				pcl::PointXYZRGBNormal p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				const float L = vec( 3 );
				const float alpha = vec( 4 );
				const float beta = vec( 5 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

				p.normal_x = surfel.normal_(0);
				p.normal_y = surfel.normal_(1);
				p.normal_z = surfel.normal_(2);

				info->cloudPtr->points.push_back( p );

			}

		}
		else {

			// PCA projection
			Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
			Eigen::Matrix< double, 3, 3 > cov3_inv = cov.block<3,3>(0,0).inverse();

			Eigen::Matrix< double, 3, 1> eigen_values_;
			Eigen::Matrix< double, 3, 3> eigen_vectors_;

			pcl::eigen33(Eigen::Matrix3d(cov.block<3,3>(0,0)), eigen_vectors_, eigen_values_);

			eigen_values_(0) = 0.0;
			eigen_values_(1) = sqrt( eigen_values_(1) );
			eigen_values_(2) = sqrt( eigen_values_(2) );

			Eigen::Matrix< double, 3, 3 > L = eigen_vectors_ * eigen_values_.asDiagonal();

			std::vector< Eigen::Matrix< double, 3, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 3, 1 > > > vecs;

			Eigen::Matrix< double, 3, 1 > v;

			v(0) =  0.0; v(1) =  -1.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  -1.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  -1.0; v(2) =  1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  0.0; v(2) =  1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  -1.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  0.0;
			vecs.push_back( v );

			v(0) =  0.0; v(1) =  1.0; v(2) =  1.0;
			vecs.push_back( v );

			for( unsigned int k = 0; k < vecs.size(); k++ ) {

				Eigen::Matrix< double, 3, 1 > vec = 1.1*vecs[k];

				vec = ( L * vec ).eval();

				vec = ( surfel.mean_.block<3,1>(0,0) + vec ).eval();


				pcl::PointXYZRGBNormal p;
				p.x = vec( 0 );
				p.y = vec( 1 );
				p.z = vec( 2 );

				// get color mean conditioned on position
				Eigen::Matrix< double, 3, 1 > cvec = surfel.mean_.block<3,1>(3,0) + cov.block<3,3>(3,0) * cov3_inv * ( vec - surfel.mean_.block<3,1>(0,0) );

				const float L = cvec( 0 );
				const float alpha = cvec( 1 );
				const float beta = cvec( 2 );

				float rf = 0, gf = 0, bf = 0;
				convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

				int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
				int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
				int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

				int rgb = ( r << 16 ) + ( g << 8 ) + b;
				p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

				p.normal_x = surfel.normal_(0);
				p.normal_y = surfel.normal_(1);
				p.normal_z = surfel.normal_(2);

				info->cloudPtr->points.push_back( p );

			}

//			for( unsigned int j = 0; j < 2; j++ ) {
//				for( unsigned int k = 0; k < 2; k++ ) {
//					for( unsigned int l = 0; l < 2; l++ ) {
//
//						spatialaggregate::OcTreeNode< float, NodeValue >* neighbor = current->getNeighbor( j, k, l );
//
//						if( !neighbor )
//							continue;
//
////						if( neighbor->value_.surfels_[i].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
////							continue;
//
//						Surfel surfel;
//						surfel.num_points_ = current->value_.surfels_[i].num_points_ + neighbor->value_.surfels_[i].num_points_;
//						surfel.mean_ = current->value_.surfels_[i].num_points_ * current->value_.surfels_[i].mean_ + neighbor->value_.surfels_[i].num_points_ * neighbor->value_.surfels_[i].mean_;
//						surfel.cov_ = (current->value_.surfels_[i].num_points_-1.f) * current->value_.surfels_[i].cov_ + (neighbor->value_.surfels_[i].num_points_-1.f) * neighbor->value_.surfels_[i].cov_;
//
//						surfel.mean_ /= surfel.num_points_;
//						surfel.cov_ /= (surfel.num_points_-1.f);
//
//						surfel.evaluateNormal();
//
//						pcl::PointXYZRGBNormal p;
//						p.x = surfel.mean_( 0 );
//						p.y = surfel.mean_( 1 );
//						p.z = surfel.mean_( 2 );
//
//						const float L = surfel.mean_( 3 );
//						const float alpha = surfel.mean_( 4 );
//						const float beta = surfel.mean_( 5 );
//
//						float rf = 0, gf = 0, bf = 0;
//						convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );
//
//						int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
//						int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
//						int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );
//
//						int rgb = ( r << 16 ) + ( g << 8 ) + b;
//						p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );
//
//						p.normal_x = surfel.normal_(0);
//						p.normal_y = surfel.normal_(1);
//						p.normal_z = surfel.normal_(2);
//
//						info->cloudPtr->points.push_back( p );
//
//					}
//				}
//			}


		}
	}

}



struct Visualize3DColorMeansInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	int viewDir, depth;
};


void MultiResolutionSurfelMap::visualize3DColorMeans( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir ) {

	Visualize3DColorMeansInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;

	octree_->root_->sweepDown( &info, &visualizeMeansFunction );

}

inline void MultiResolutionSurfelMap::visualizeMeansFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	Visualize3DColorMeansInfo* info = (Visualize3DColorMeansInfo*) data;

	if( (info->depth == -1 && current->type_ == spatialaggregate::OCTREE_BRANCHING_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;

	Eigen::Matrix< float, 4, 1 > minPos = current->getMinPosition();
	Eigen::Matrix< float, 4, 1 > maxPos = current->getMaxPosition();


	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		pcl::PointXYZRGB p;
		p.x = surfel.mean_( 0 );
		p.y = surfel.mean_( 1 );
		p.z = surfel.mean_( 2 );

		const float L = surfel.mean_( 3 );
		const float alpha = surfel.mean_( 4 );
		const float beta = surfel.mean_( 5 );

		float rf = 0, gf = 0, bf = 0;
		convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

		int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
		int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
		int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

		int rgb = ( r << 16 ) + ( g << 8 ) + b;
		p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

		info->cloudPtr->points.push_back( p );

	}

}



struct VisualizeContoursInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	Eigen::Matrix4d transform;
	int viewDir, depth;
	bool random;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


void MultiResolutionSurfelMap::visualizeContours( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, const Eigen::Matrix4d& transform, int depth, int viewDir, bool random ) {

	VisualizeContoursInfo info;
	info.transform = transform;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.random = random;

	octree_->root_->sweepDown( &info, &visualizeContoursFunction );

}

inline void MultiResolutionSurfelMap::visualizeContoursFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	VisualizeContoursInfo* info = (VisualizeContoursInfo*) data;

	if( (info->depth == -1 && current->type_ != spatialaggregate::OCTREE_LEAF_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;


	Eigen::Matrix< float, 4, 1 > minPos = current->getMinPosition();
	Eigen::Matrix< float, 4, 1 > maxPos = current->getMaxPosition();


	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		// determine angle between surfel normal and view direction onto surfel
		Eigen::Vector3d viewDirection = surfel.mean_.block<3,1>(0,0) - info->transform.block<3,1>(0,3);
		viewDirection.normalize();

		float cangle = viewDirection.dot( surfel.normal_ );

		// cholesky decomposition
		Eigen::Matrix< double, 6, 6 > cov = surfel.cov_;
		Eigen::LLT< Eigen::Matrix< double, 6, 6 > > chol = cov.llt();

		std::vector< Eigen::Matrix< double, 6, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 6, 1 > > > vecs;

		Eigen::Matrix< double, 6, 1 > v;
		v.setZero();

		vecs.push_back( v );

		v(0) =  1.0; v(1) =  0.0; v(2) =  0.0;
		vecs.push_back( v );

		v(0) = -1.0; v(1) =  0.0; v(2) =  0.0;
		vecs.push_back( v );

		v(0) =  0.0; v(1) =  1.0; v(2) =  0.0;
		vecs.push_back( v );

		v(0) =  0.0; v(1) = -1.0; v(2) =  0.0;
		vecs.push_back( v );

		v(0) =  0.0; v(1) =  0.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  0.0; v(1) =  0.0; v(2) = -1.0;
		vecs.push_back( v );


		v(0) =  1.0; v(1) =  1.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  1.0; v(1) =  1.0; v(2) = -1.0;
		vecs.push_back( v );

		v(0) =  1.0; v(1) = -1.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  1.0; v(1) = -1.0; v(2) = -1.0;
		vecs.push_back( v );

		v(0) =  -1.0; v(1) =  1.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  -1.0; v(1) =  1.0; v(2) = -1.0;
		vecs.push_back( v );

		v(0) =  -1.0; v(1) = -1.0; v(2) =  1.0;
		vecs.push_back( v );

		v(0) =  -1.0; v(1) = -1.0; v(2) = -1.0;
		vecs.push_back( v );

		for( unsigned int k = 0; k < vecs.size(); k++ ) {

			Eigen::Matrix< double, 6, 1 > vec = 1.1*vecs[k];

			vec = ( chol.matrixL() * vec ).eval();

			vec = ( surfel.mean_ + vec ).eval();

			pcl::PointXYZRGB p;
			p.x = vec( 0 );
			p.y = vec( 1 );
			p.z = vec( 2 );

			const float L = vec( 3 );
			const float alpha = vec( 4 );
			const float beta = vec( 5 );

			float rf = 0, gf = 0, bf = 0;

			rf = 1.f;
			gf = fabsf( cangle );
			bf = fabsf( cangle );

//			convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

			int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
			int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
			int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

			int rgb = ( r << 16 ) + ( g << 8 ) + b;
			p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

//			if( fabsf( cangle ) > 0.2 )
//				continue;

			info->cloudPtr->points.push_back( p );

		}

	}

}



struct VisualizeNormalsInfo {
	pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr;
	int viewDir, depth;
};


void MultiResolutionSurfelMap::visualizeNormals( pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr, int depth, int viewDir ) {

	VisualizeNormalsInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;

	octree_->root_->sweepDown( &info, &visualizeNormalsFunction );

}

inline void MultiResolutionSurfelMap::visualizeNormalsFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	VisualizeNormalsInfo* info = (VisualizeNormalsInfo*) data;

	if( (info->depth == -1 && current->type_ != spatialaggregate::OCTREE_LEAF_NODE) )
		return;

	if( info->depth >= 0 && current->depth_ != info->depth )
		return;

	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		const Surfel& surfel = current->value_.surfels_[i];

		if ( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		pcl::PointXYZRGBNormal p;

		p.x = surfel.mean_(0);
		p.y = surfel.mean_(1);
		p.z = surfel.mean_(2);

		const float L = surfel.mean_( 3 );
		const float alpha = surfel.mean_( 4 );
		const float beta = surfel.mean_( 5 );

		float rf = 0, gf = 0, bf = 0;
		convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );

		int r = std::max( 0, std::min( 255, (int) ( 255.0 * rf ) ) );
		int g = std::max( 0, std::min( 255, (int) ( 255.0 * gf ) ) );
		int b = std::max( 0, std::min( 255, (int) ( 255.0 * bf ) ) );

		int rgb = ( r << 16 ) + ( g << 8 ) + b;
		p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

		p.normal_x = surfel.normal_(0);
		p.normal_y = surfel.normal_(1);
		p.normal_z = surfel.normal_(2);

		info->cloudPtr->points.push_back( p );

	}

}



struct VisualizeSimilarityInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	int viewDir, depth;
	spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* referenceNode;
	bool simple;
};


void MultiResolutionSurfelMap::visualizeSimilarity( spatialaggregate::OcTreeNode< float, NodeValue >* referenceNode, pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool simple ) {

	VisualizeSimilarityInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.simple = simple;

	info.referenceNode = referenceNode;

	if( !info.referenceNode )
		return;

	if( info.referenceNode->depth_ != depth )
		return;

	octree_->root_->sweepDown( &info, &visualizeSimilarityFunction );

}

inline void MultiResolutionSurfelMap::visualizeSimilarityFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	const float maxDist = 0.1f;

	VisualizeSimilarityInfo* info = (VisualizeSimilarityInfo*) data;

	if( current->depth_ != info->depth )
		return;

	// generate markers for histogram surfels
	float minDist = std::numeric_limits<float>::max();
	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		Surfel& surfel = info->referenceNode->value_.surfels_[i];

		if( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		Surfel& surfel2 = current->value_.surfels_[i];

		if( surfel2.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		if( info->simple ) {
			ShapeTextureFeature f1 = surfel.simple_shape_texture_features_;
			ShapeTextureFeature f2 = surfel2.simple_shape_texture_features_;
			f1.shape_ /= f1.num_points_;
			f1.texture_ /= f1.num_points_;
			f2.shape_ /= f2.num_points_;
			f2.texture_ /= f2.num_points_;
			minDist = std::min( minDist, f1.distance( f2 ) );
		}
		else
			minDist = std::min( minDist, surfel.agglomerated_shape_texture_features_.distance( surfel2.agglomerated_shape_texture_features_ ) );

	}

	if( minDist == std::numeric_limits<float>::max() )
		return;

	Eigen::Vector4f pos = current->getCenterPosition();

	pcl::PointXYZRGB p;
	p.x = pos( 0 );
	p.y = pos( 1 );
	p.z = pos( 2 );

	int r = std::max( 0, std::min( 255, (int) ( 255.0 * minDist / maxDist ) ) );
	int g = 255 - std::max( 0, std::min( 255, (int) ( 255.0 * minDist / maxDist ) ) );
	int b = 255 - std::max( 0, std::min( 255, (int) ( 255.0 * minDist / maxDist ) ) );

	int rgb = ( r << 16 ) + ( g << 8 ) + b;
	p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

	info->cloudPtr->points.push_back( p );

}



struct VisualizeBordersInfo {
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr;
	int viewDir, depth;
	bool foreground;
};


void MultiResolutionSurfelMap::visualizeBorders( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool foreground ) {

	VisualizeBordersInfo info;
	info.cloudPtr = cloudPtr;
	info.viewDir = viewDir;
	info.depth = depth;
	info.foreground = foreground;

	octree_->root_->sweepDown( &info, &visualizeBordersFunction );

}

inline void MultiResolutionSurfelMap::visualizeBordersFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data ) {

	VisualizeBordersInfo* info = (VisualizeBordersInfo*) data;

	if( current->depth_ != info->depth )
		return;

	for ( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {

		if( info->viewDir >= 0 && info->viewDir != i )
			continue;

		Surfel& surfel = current->value_.surfels_[i];

		if( surfel.num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
			continue;

		Eigen::Vector4f pos = current->getCenterPosition();

		pcl::PointXYZRGB p;
		p.x = pos( 0 );
		p.y = pos( 1 );
		p.z = pos( 2 );

		int r = 255;
		int g = 0;
		int b = 0;

		if( info->foreground && current->value_.border_ ) {
			r = 0; g = 255; b = 255;
		}

		if( !info->foreground && !surfel.applyUpdate_ ) {
			r = 0; g = 255; b = 255;
		}

		int rgb = ( r << 16 ) + ( g << 8 ) + b;
		p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

		info->cloudPtr->points.push_back( p );

	}


}


// s. http://people.cs.vt.edu/~kafura/cs2704/op.overloading2.html
template< typename T, int rows, int cols >
std::ostream& operator<<( std::ostream& os, Eigen::Matrix< T, rows, cols >& m ) {
	for ( unsigned int i = 0; i < rows; i++ ) {
		for ( unsigned int j = 0; j < cols; j++ ) {
			T d = m( i, j );
			os.write( (char*) &d, sizeof(T) );
		}
	}

	return os;
}

template< typename T, int rows, int cols >
std::istream& operator>>( std::istream& os, Eigen::Matrix< T, rows, cols >& m ) {
	for ( unsigned int i = 0; i < rows; i++ ) {
		for ( unsigned int j = 0; j < cols; j++ ) {
			T d;
			os.read( (char*) &d, sizeof(T) );
			m( i, j ) = d;
		}
	}

	return os;
}

std::ostream& operator<<( std::ostream& os, MultiResolutionSurfelMap::NodeValue& v ) {

	for ( int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {
		os << v.surfels_[i].initial_view_dir_;
		os << v.surfels_[i].first_view_dir_;
		os.write( (char*) &v.surfels_[i].num_points_, sizeof(double) );
		os << v.surfels_[i].mean_;
		os << v.surfels_[i].normal_;
		os << v.surfels_[i].cov_;

//		os << v.surfels_[i].agglomerated_shape_texture_features_.shape_;
//		os << v.surfels_[i].agglomerated_shape_texture_features_.texture_;
//		os.write( (char*) &v.surfels_[i].agglomerated_shape_texture_features_.num_points_, sizeof(float) );

		os.write( (char*) &v.surfels_[i].up_to_date_, sizeof(bool) );
		os.write( (char*) &v.surfels_[i].applyUpdate_, sizeof(bool) );


	}

	return os;
}

std::istream& operator>>( std::istream& os, MultiResolutionSurfelMap::NodeValue& v ) {

	for ( int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {
		os >> v.surfels_[i].initial_view_dir_;
		os >> v.surfels_[i].first_view_dir_;
		os.read( (char*) &v.surfels_[i].num_points_, sizeof(double) );
		os >> v.surfels_[i].mean_;
		os >> v.surfels_[i].normal_;
		os >> v.surfels_[i].cov_;
		os.read( (char*) &v.surfels_[i].up_to_date_, sizeof(bool) );
		os.read( (char*) &v.surfels_[i].applyUpdate_, sizeof(bool) );

	}

	return os;
}

std::ostream& operator<<( std::ostream& os, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >& node ) {

	os.write( (char*) &node.depth_, sizeof(int) );
	os.write( (char*) &node.pos_key_.x_, sizeof(uint32_t) );
	os.write( (char*) &node.pos_key_.y_, sizeof(uint32_t) );
	os.write( (char*) &node.pos_key_.z_, sizeof(uint32_t) );
	os.write( (char*) &node.min_key_.x_, sizeof(uint32_t) );
	os.write( (char*) &node.min_key_.y_, sizeof(uint32_t) );
	os.write( (char*) &node.min_key_.z_, sizeof(uint32_t) );
	os.write( (char*) &node.max_key_.x_, sizeof(uint32_t) );
	os.write( (char*) &node.max_key_.y_, sizeof(uint32_t) );
	os.write( (char*) &node.max_key_.z_, sizeof(uint32_t) );
	os.write( (char*) &node.type_, sizeof(spatialaggregate::OcTreeNodeType) );

//	for( unsigned int n = 0; n < 27; n++ )
//		os.write( (char*) &node.neighbors_[n], sizeof(spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >*) );
//
//	for( unsigned int c = 0; c < 8; c++ )
//		os.write( (char*) &node.children_[c], sizeof(spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >*) );
//
//	os.write( (char*) &node.parent_, sizeof(spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >*) );
//
//	os.write( (char*) &node.tree_, sizeof(spatialaggregate::OcTree< float, MultiResolutionSurfelMap::NodeValue >*) );


	os << node.value_;

	return os;
}

std::istream& operator>>( std::istream& os, spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >& node ) {

	os.read( (char*) &node.depth_, sizeof(int) );
	os.read( (char*) &node.pos_key_.x_, sizeof(uint32_t) );
	os.read( (char*) &node.pos_key_.y_, sizeof(uint32_t) );
	os.read( (char*) &node.pos_key_.z_, sizeof(uint32_t) );
	os.read( (char*) &node.min_key_.x_, sizeof(uint32_t) );
	os.read( (char*) &node.min_key_.y_, sizeof(uint32_t) );
	os.read( (char*) &node.min_key_.z_, sizeof(uint32_t) );
	os.read( (char*) &node.max_key_.x_, sizeof(uint32_t) );
	os.read( (char*) &node.max_key_.y_, sizeof(uint32_t) );
	os.read( (char*) &node.max_key_.z_, sizeof(uint32_t) );
	os.read( (char*) &node.type_, sizeof(spatialaggregate::OcTreeNodeType) );
	os >> node.value_;

	return os;

}

void MultiResolutionSurfelMap::save( const std::string& filename ) {

	// create downsampling map for the target
	algorithm::OcTreeSamplingMap< float, NodeValue > samplingMap = algorithm::downsampleOcTree( *octree_, false, octree_->max_depth_ );

	std::ofstream outfile( filename.c_str(), std::ios::out | std::ios::binary );

	// header information
	outfile.write( (char*) &min_resolution_, sizeof(float) );
	outfile.write( (char*) &max_range_, sizeof(float) );

	outfile << reference_pose_;

	for ( int i = 0; i <= octree_->max_depth_; i++ ) {
		int numNodes = samplingMap[i].size();
		outfile.write( (char*) &numNodes, sizeof(int) );

		for ( std::list< spatialaggregate::OcTreeNode< float, NodeValue >* >::iterator it = samplingMap[i].begin(); it != samplingMap[i].end(); ++it ) {
			outfile << *(*it);
		}
	}

}

void MultiResolutionSurfelMap::load( const std::string& filename ) {

	std::ifstream infile( filename.c_str(), std::ios::in | std::ios::binary );

	if ( !infile.is_open() ) {
		std::cout << "could not open file " << filename.c_str() << "\n";
	}

	infile.read( (char*) &min_resolution_, sizeof(float) );
	infile.read( (char*) &max_range_, sizeof(float) );

	infile >> reference_pose_;

	octree_ = boost::shared_ptr< spatialaggregate::OcTree< float, NodeValue > >( new spatialaggregate::OcTree< float, NodeValue >( Eigen::Matrix< float, 4, 1 >( 0.f, 0.f, 0.f, 0.f ), min_resolution_, max_range_ ) );
	octree_->allocator_->deallocateNode( octree_->root_ );
	octree_->root_ = NULL;

	for ( int i = 0; i <= octree_->max_depth_; i++ ) {
		int numNodesOnDepth = 0;
		infile.read( (char*) &numNodesOnDepth, sizeof(int) );

		for ( int j = 0; j < numNodesOnDepth; j++ ) {

			spatialaggregate::OcTreeNode< float, NodeValue >* node = octree_->allocator_->allocateNode();
			octree_->acquire( node );

			infile >> ( *node );

			// insert octree node into the tree
			// start at root and traverse the tree until we find an empty leaf
			spatialaggregate::OcTreeNode< float, NodeValue >* n = octree_->root_;

			if ( !n ) {
				node->parent_ = NULL;
				octree_->root_ = node;
			} else {

				// search for parent
				spatialaggregate::OcTreeNode< float, NodeValue >* n2 = n;
				while ( n2 ) {
					n = n2;
					n2 = n->children_[n->getOctant( node->pos_key_ )];
				}

				// assert that found parent node has the correct depth
				if ( n->depth_ != node->depth_ - 1 || (n->type_ != spatialaggregate::OCTREE_BRANCHING_NODE && n->type_ != spatialaggregate::OCTREE_MAX_DEPTH_BRANCHING_NODE) ) {
					std::cout << "MultiResolutionMap::load(): bad things happen\n";
				} else {
					n->children_[n->getOctant( node->pos_key_ )] = node;
					node->parent_ = n;
				}
			}

		}

	}

}


void MultiResolutionSurfelMap::indexNodesRecursive( spatialaggregate::OcTreeNode< float, NodeValue >* node, int minDepth, int maxDepth, bool includeBranchingNodes ) {

	if( node->depth_ >= minDepth && node->depth_ <= maxDepth && ( includeBranchingNodes || node->type_ != spatialaggregate::OCTREE_BRANCHING_NODE ) ) {

		bool hasValidSurfel = true;
//		bool hasValidSurfel = false;
//		for( unsigned int i = 0; i < NodeValue::num_surfels_; i++ )
//			if( node->value_.surfels_[i].num_points_ >= MultiResolutionSurfelMap::Surfel::min_points_ )
//				hasValidSurfel = true;

		if( hasValidSurfel ) {
			node->value_.idx_ = indexedNodes_.size();
			indexedNodes_.push_back( node );
		}
	}
	else
		node->value_.idx_ = -1;

	for( unsigned int i = 0; i < 8; i ++ ) {
		if( node->children_[i] ) {
			indexNodesRecursive( node->children_[i], minDepth, maxDepth, includeBranchingNodes );
		}
	}

}


void MultiResolutionSurfelMap::indexNodes( int minDepth, int maxDepth, bool includeBranchingNodes ) {

	indexedNodes_.clear();
	indexNodesRecursive( octree_->root_, minDepth, maxDepth, includeBranchingNodes );

}


unsigned int MultiResolutionSurfelMap::numSurfelsRecursive( spatialaggregate::OcTreeNode< float, NodeValue >* node ) {

	unsigned int count = 0;
	for( unsigned int i = 0; i < NodeValue::num_surfels_; i++ ) {
		if( node->value_.surfels_[i].num_points_ > MultiResolutionSurfelMap::Surfel::min_points_ ) {
			count++;
		}
	}

	for( unsigned int i = 0; i < 8; i++ ) {
		if( node->children_[i] )
			count += numSurfelsRecursive( node->children_[i] );
	}

	return count;

}


unsigned int MultiResolutionSurfelMap::numSurfels() {

	return numSurfelsRecursive( octree_->root_ );

}


void MultiResolutionSurfelMap::buildSurfelPairs( ) {
    surfel_pair_list_map_.resize( 17 );
    all_surfel_pairs_.resize( 17 );
    reference_surfels_.resize( 17 );

    int maxDepth = this->octree_->max_depth_;

    float minResolution = this->octree_->volumeSizeForDepth( maxDepth );
    float maxResolution = minResolution * 4.f;

    minResolution *= 0.99f;
    maxResolution *= 1.01f;

    float maxDist = -std::numeric_limits<float>::max();

    for( int d = maxDepth; d >= 0; d-- ) {

        const float processResolution = this->octree_->volumeSizeForDepth( d );

        if( processResolution < minResolution || processResolution > maxResolution ) {
            continue;
        }

        float maxDistAtDepth = 0.f;
        buildSurfelPairsOnDepthParallel( samplingMap_[d], d, maxDistAtDepth );

        maxDist = std::max( maxDist, maxDistAtDepth );
    }

    LOG_STREAM("Object diameter: " << maxDist );

    this->surfelMaxDist_ = maxDist;
}

void MultiResolutionSurfelMap::buildSurfelPairsHashmap() {
    int maxDepth = this->octree_->max_depth_;

    float minResolution = this->octree_->volumeSizeForDepth( maxDepth );
    float maxResolution = minResolution * 4.f;

    minResolution *= 0.99f;
    maxResolution *= 1.01f;

    for ( int d = maxDepth; d >= 0; d-- )     {
        const float processResolution = this->octree_->volumeSizeForDepth( d );

        if( processResolution < minResolution || processResolution > maxResolution )
            continue;

        buildSurfelPairsHashmapOnDepth( d );
    }
}

inline bool MultiResolutionSurfelMap::buildSurfelPair( SurfelPairSignature & signature, const Surfel& src, const Surfel& dst ) {

    // surflet pair relation as in "model globally match locally"
    Eigen::Vector3d p1 = src.mean_.block<3,1>(0,0);
    Eigen::Vector3d p2 = dst.mean_.block<3,1>(0,0);
    Eigen::Vector3d n1 = src.normal_;
    Eigen::Vector3d n2 = dst.normal_;

    Eigen::Vector3d d = p2-p1;
    Eigen::Vector3d d_normalized = d / d.norm();

    // normalize ranges to [0,1]
    signature.shape_signature_(0) = d.norm();
    signature.shape_signature_(1) = 0.5 * (n1.dot( d_normalized )+1.0);
    signature.shape_signature_(2) = 0.5 * (n2.dot( d_normalized )+1.0);
    signature.shape_signature_(3) = 0.5 * (n1.dot( n2 )+1.0);

//    if ( signature.shape_signature_(3) > MAX_ANGLE_DIFF ) {
//        if ( gsl_rng_uniform( r ) < 0.5f )
//            return false;
//    }

    // color comparison with mean L alpha beta
    // normalize ranges to [0,1]
    signature.color_signature_ = dst.mean_.block<3,1>(3,0) - src.mean_.block<3,1>(3,0);

	if ( signature.shape_signature_(3) > MAX_ANGLE_DIFF && fabsf(signature.color_signature_(0)) < MIN_LUMINANCE_DIFF && fabsf(signature.color_signature_(1)) < MIN_COLOR_DIFF && fabsf(signature.color_signature_(2)) < MIN_COLOR_DIFF ) {
//		if ( gsl_rng_uniform( r ) < 0.5f )
			return false;
	}


    signature.color_signature_(0) = 0.5 * (signature.color_signature_(0)+1.0); // L in [0,1]
    signature.color_signature_(1) = 0.25 * (signature.color_signature_(1)+2.0); // alpha in [-1,1]
    signature.color_signature_(2) = 0.25 * (signature.color_signature_(2)+2.0); // beta in [-1,1]



//    signature.color_signature_src_ = src.mean_.block<3,1>(3,0);
//    signature.color_signature_dst_ = dst.mean_.block<3,1>(3,0);
//    signature.color_signature_src_(1) = 0.5 * ( signature.color_signature_src_(1) + 1.0 );
//    signature.color_signature_src_(2) = 0.5 * ( signature.color_signature_src_(2) + 1.0 );
//    signature.color_signature_dst_(1) = 0.5 * ( signature.color_signature_dst_(1) + 1.0 );
//    signature.color_signature_dst_(2) = 0.5 * ( signature.color_signature_dst_(2) + 1.0 );

    // transform from map frame to reference surfel frame
    // rotate x-axis along normal
//	Eigen::Vector3d axis = n1.cross( Eigen::Vector3d::UnitX() ).normalized();
//	float angle = acos ( n1.dot( Eigen::Vector3d::UnitX () ) );
//	Eigen::Matrix3d refRot( Eigen::AngleAxisd ( angle, axis ) );

    Eigen::Matrix4d transform_map_ref = Eigen::Matrix4d::Identity();

    const Eigen::Vector7d & srcRefPose = src.reference_pose_;
    Eigen::Vector3d refTrans = srcRefPose.head<3>();
    Eigen::Matrix3d refRot = Eigen::Quaterniond ( srcRefPose.tail<4>() ).matrix();

    transform_map_ref.block<3,3>( 0,0 ) = refRot.transpose();
    transform_map_ref.block<3,1>( 0,3 ) = -( refRot.transpose() * refTrans );

    // compute the angle to rotate around the x-axis to align with positive y-axis
    Eigen::Vector4d p2_ref = transform_map_ref * Eigen::Vector4d( p2(0), p2(1), p2(2), 1.f);
    float alpha = atan2f ( -p2_ref(2), p2_ref(1));
    signature.alpha_ = alpha;

    return true;

}


int MultiResolutionSurfelMap::buildSurfelPairsForSurfel(
        spatialaggregate::OcTreeNode< float, NodeValue >* node,
        Surfel* srcSurfel,
        int surfelIdx,
        std::vector< spatialaggregate::OcTreeNode< float, NodeValue >* > & nodes,
        SurfelPairVector & pairs,
        float & maxDist,
        float samplingRate
        ) {

    int step = (int)1.f / samplingRate;

    if ( !srcSurfel->reference_pose_set )
        srcSurfel->updateReferencePose();

    pairs.reserve( nodes.size() );

    float maxResDist = node->resolution() * params_.surfelPairMaxDistResFactor_;

    for ( std::vector< spatialaggregate::OcTreeNode< float, NodeValue >* >::iterator nodeIt =
          nodes.begin(); nodeIt < nodes.end(); std::advance( nodeIt, step ) ) {
        spatialaggregate::OcTreeNode< float, NodeValue >* currentNode = *nodeIt;
        if ( currentNode == node )
            continue;

        Surfel* tgtSurfel = &( currentNode->value_.surfels_[surfelIdx] );

        if( tgtSurfel->num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
            continue;

        if ( boost::math::isnan( tgtSurfel->normal_(0) ) ||
             boost::math::isnan( tgtSurfel->normal_(1) ) ||
             boost::math::isnan( tgtSurfel->normal_(2) ) )
            continue;

        float dist = (srcSurfel->mean_.block<3,1>(0,0) - tgtSurfel->mean_.block<3,1>(0,0)).norm();

        if( dist > maxResDist )
        	continue;

        SurfelPairSignature signature;
        bool success = buildSurfelPair( signature, *srcSurfel, *tgtSurfel );

        if (!success)
            continue;

//        SurfelPair pair( srcSurfel, tgtSurfel, signature );

//        float dist = (float)signature.shape_signature_(0);

        if ( dist > maxDist )
            maxDist = dist;

        pairs.push_back( SurfelPair( srcSurfel, tgtSurfel, signature ) );
    }

    return pairs.size();
}


template< typename TNodeValue >
class BuildSurfelPairsFunctor {
public:

    BuildSurfelPairsFunctor(
            MultiResolutionSurfelMap* map,
            tbb::concurrent_vector< std::vector< MultiResolutionSurfelMap::SurfelPair, Eigen::aligned_allocator< MultiResolutionSurfelMap::SurfelPair > > >* surfel_pairs,
            tbb::concurrent_vector< MultiResolutionSurfelMap::Surfel* >* reference_surfels,
//            std::vector< MultiResolutionSurfelMap::SurfelPairVector >* surfel_pairs,
//            std::vector< MultiResolutionSurfelMap::Surfel* >* reference_surfels,
//            tbb::concurrent_unordered_map< uint64, unsigned int >* surfel_pair_keys,
            std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >* nodes,
            tbb::atomic<float>* max_dist,
            const float samplingRate,
            const int processDepth
             ) {
        map_ = map;
        surfel_pairs_ = surfel_pairs;
        reference_surfels_ = reference_surfels;
//        surfel_pair_keys_ = surfel_pair_keys;
        nodes_ = nodes;
        max_dist_ = max_dist;
        processDepth_ = processDepth;
        totalPairs_ = 0;
        samplingRate_ = samplingRate;
    }

    void operator()( const tbb::blocked_range<size_t>& r ) const {
        for( size_t i=r.begin(); i!=r.end(); ++i )
            (*this)((*nodes_)[i]);
    }

    void operator()( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >*& node ) const {

        float maxDist = -std::numeric_limits<float>::max();

        if ( gsl_rng_uniform( map_->r ) < ( 1.f - samplingRate_ ) )
            return;

        // loop around the surfels and build surfel pair relations
        for ( unsigned int i = 0; i < MultiResolutionSurfelMap::NodeValue::num_surfels_; i++ ) {
            if( node->value_.surfels_[i].num_points_ >= MultiResolutionSurfelMap::Surfel::min_points_ ) {
            	MultiResolutionSurfelMap::Surfel* currentRefSurfel = &(node->value_.surfels_[i]);
                unsigned int currentSurfelIdx = i;

                if ( boost::math::isnan( currentRefSurfel->normal_(0) ) ||
                     boost::math::isnan( currentRefSurfel->normal_(1) ) ||
                     boost::math::isnan( currentRefSurfel->normal_(2) ) )
                    continue;

                typename tbb::concurrent_vector< MultiResolutionSurfelMap::Surfel* >::iterator refSurfelIt = reference_surfels_->push_back( currentRefSurfel );
//                unsigned int refIdx = reference_surfels_->size();
//                reference_surfels_->push_back( currentRefSurfel );
                unsigned int refIdx = refSurfelIt - reference_surfels_->begin();
                currentRefSurfel->idx_ = refIdx;

                tbb::concurrent_vector< std::vector<MultiResolutionSurfelMap::SurfelPair, Eigen::aligned_allocator< MultiResolutionSurfelMap::SurfelPair > > >::iterator pairit =
                        surfel_pairs_->grow_by( 1 );

//                surfel_pairs_->resize( surfel_pairs_->size() + 1, MultiResolutionSurfelMap::SurfelPairVector() );

                MultiResolutionSurfelMap::SurfelPairVector & pairs = *pairit;
//                MultiResolutionSurfelMap::SurfelPairVector & pairs = (*surfel_pairs_)[refIdx]; //*( surfel_pairs_->rbegin() );

                map_->buildSurfelPairsForSurfel( node, currentRefSurfel, currentSurfelIdx,
                                                                     *nodes_, pairs, maxDist/*, samplingRate_*/ );


//                for ( std::vector< spatialaggregate::OcTreeNode< float, NodeValue >* >::iterator nodeIt =
//                      nodes_->begin(); nodeIt != nodes_->end(); nodeIt ) {
//                    spatialaggregate::OcTreeNode< float, NodeValue >* currentNode = *nodeIt;
//                    if ( currentNode == node )
//                        continue;

//                    Surfel* tgtSurfel = &( currentNode->value_.surfels_[currentSurfelIdx] );

//                    if( tgtSurfel->num_points_ < NUM_SURFEL_POINTS_ROBUST )
//                        continue;

//                    if ( boost::math::isnan( tgtSurfel->normal_(0) ) ||
//                         boost::math::isnan( tgtSurfel->normal_(1) ) ||
//                         boost::math::isnan( tgtSurfel->normal_(2) ) )
//                        continue;


//                    SurfelPairSignature signature;
//                    bool success = buildSurfelPair( signature, *currentRefSurfel, *tgtSurfel );

//                    if (!success)
//                        continue;

//                    SurfelPair pair( currentRefSurfel, tgtSurfel, signature );

//                    float dist = (float)signature.shape_signature_(0);

//                    if ( dist > maxDist )
//                        maxDist = dist;

//                    pairs.push_back( pair );
//                    ++numFoundPairs;
//                }

                totalPairs_ += pairs.size();
            }
        }

        if ( maxDist > *max_dist_ ) {
            tbb::atomic<float> old;
            do {
                old = *max_dist_;
                if ( maxDist < old )
                    break;
            } while ( max_dist_->compare_and_swap( maxDist, old ) != old );
        }
    }

    MultiResolutionSurfelMap* map_;
    tbb::concurrent_vector< std::vector< MultiResolutionSurfelMap::SurfelPair, Eigen::aligned_allocator< MultiResolutionSurfelMap::SurfelPair > > >* surfel_pairs_;
    tbb::concurrent_vector< MultiResolutionSurfelMap::Surfel* >* reference_surfels_;
//    std::vector< MultiResolutionSurfelMap::SurfelPairVector >* surfel_pairs_;
//    std::vector< Surfel* >* reference_surfels_;
//    tbb::concurrent_unordered_map< uint64, unsigned int >* surfel_pair_keys_;
    std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >* nodes_;
    tbb::atomic<float>* max_dist_;
    int processDepth_;
    mutable int totalPairs_;
    float samplingRate_;
};


void MultiResolutionSurfelMap::buildSurfelPairsOnDepthParallel( std::vector< spatialaggregate::OcTreeNode< float, NodeValue >* >& nodes, int processDepth, float & maxDist ) {


    tbb::concurrent_vector< std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > > surfel_pairs_on_depth;
    tbb::concurrent_vector< Surfel* > reference_surfels_on_depth;
//    std::vector< std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > > surfel_pairs_on_depth( 0, std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > >() );
//    std::vector< Surfel* > reference_surfels_on_depth;

//    std::vector< Surfel* > & reference_surfels_on_depth = reference_surfels_[processDepth];
//    std::vector< std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > > & surfel_pairs_on_depth =
//            all_surfel_pairs_[processDepth];

    tbb::atomic<float> max_dist;
    max_dist = 0.f;
//    tbb::concurrent_unordered_map< uint64, unsigned int > surfel_pair_keys_on_depth;

    float processResolution = this->octree_->volumeSizeForDepth( processDepth );
    float targetNodesForDepth = 1.f / processResolution * 15;
    LOG_STREAM( "Target nodes: " << targetNodesForDepth );
    float samplingRate = std::max( 0.2f, std::min( params_.surfelPairSamplingRate_, targetNodesForDepth / (float)nodes.size() ) );
    if( params_.surfelPairSamplingRate_ > 2.f )
    	samplingRate = 1.f;

    LOG_STREAM("Sampling rate: " << samplingRate << " param: " << params_.surfelPairSamplingRate_ );

    reference_surfels_on_depth.reserve( (int) samplingRate * (float)nodes.size() );
    surfel_pairs_on_depth.reserve( (int) samplingRate * (float)nodes.size() );

    BuildSurfelPairsFunctor< NodeValue > bf( this, &surfel_pairs_on_depth, &reference_surfels_on_depth, &nodes, &max_dist, samplingRate, processDepth );

    LOG_STREAM("Processing depth [" << processDepth << "], nodes: " << nodes.size() );

    if( params_.parallel_ )
        tbb::parallel_for_each( nodes.begin(), nodes.end(), bf );
    else
        std::for_each( nodes.begin(), nodes.end(), bf );


    all_surfel_pairs_[processDepth].resize( surfel_pairs_on_depth.size(), std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > >( ) );
    reference_surfels_[processDepth].resize( reference_surfels_on_depth.size(), 0 );

    LOG_STREAM("Num ref surfels: " << reference_surfels_on_depth.size() );
    LOG_STREAM("Num pair vectors: " << surfel_pairs_on_depth.size() );

    for( unsigned int i = 0; i<surfel_pairs_on_depth.size(); ++i ) {
        reference_surfels_[processDepth][i] = reference_surfels_on_depth[i];
        all_surfel_pairs_[processDepth][i].insert( all_surfel_pairs_[processDepth][i].end(),
                                                   surfel_pairs_on_depth[i].begin(),
                                                   surfel_pairs_on_depth[i].end() );
    }

//    reference_surfels_[processDepth].insert( reference_surfels_[processDepth].end(), reference_surfels_on_depth.begin(), reference_surfels_on_depth.end() );
//    all_surfel_pairs_[processDepth].insert( all_surfel_pairs_[processDepth].end(), surfel_pairs_on_depth.begin(), surfel_pairs_on_depth.end() );

    LOG_STREAM("Surfel pairs at depth [" << processDepth << "]: " << bf.totalPairs_ );

    maxDist = *(bf.max_dist_);

//    surfel_pair_list_map_[d].insert( surfel_pair_keys_on_depth.begin(), surfel_pair_keys_on_depth.end() );
}


void MultiResolutionSurfelMap::buildSurfelPairsHashmapOnDepth( int processDepth  ) {
    const float bin_angle = params_.surfelPairFeatureBinAngle_;
    const float bin_dist = params_.surfelPairFeatureBinDist_ * octree_->resolutions_[processDepth];

    std::unordered_map<SurfelPairKey, std::vector< SurfelPair*> > & key_map = surfel_pair_list_map_[processDepth];
//    std::vector< std::vector< SurfelPair*> > & key_pairs = surfel_pairs_[processDepth];
    std::vector< std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > > & ref_pairs = all_surfel_pairs_[processDepth];

    for ( std::vector< std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > >::iterator it =
          ref_pairs.begin(); it != ref_pairs.end(); ++it ) {
        std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > & current_ref_pairs =
                *it;

        for ( std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > >::iterator pair_it =
              current_ref_pairs.begin(); pair_it != current_ref_pairs.end(); ++pair_it ) {
            SurfelPair & pair = *pair_it;
            const SurfelPairKey key = pair.signature_.getKey( this->surfelMaxDist_, bin_dist, bin_angle, params_.surfelPairUseColor_ );

            // insert into key map

            std::unordered_map<SurfelPairKey, std::vector< SurfelPair*> >::const_iterator got =
                    key_map.find( key );

            if ( got == key_map.end() ) {
                // insert!

//                const unsigned int idx = key_pairs.size();
//                key_pairs.resize( key_pairs.size() + 1, std::vector < SurfelPair* >() );
//                key_pairs[idx].push_back( &pair );
//                key_map[key] = idx;

                key_map.insert(std::make_pair(key, std::vector<SurfelPair*>() ) );

            } //else {

            key_map[key].push_back( &pair );
            //}
        }
    }
}

void MultiResolutionSurfelMap::buildSamplingMap( ) {
    this->samplingMap_ = algorithm::downsampleVectorOcTree( *(this->octree_), false, this->octree_->max_depth_ );
}



