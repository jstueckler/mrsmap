/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 24.09.2012
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

#include "mrsmap/utilities/utilities.h"
#include <boost/algorithm/string.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/round.hpp>

using namespace mrsmap;


double mrsmap::colormapjet::interpolate( double val, double y0, double x0, double y1, double x1 ) {
    return (val-x0)*(y1-y0)/(x1-x0) + y0;
}

double mrsmap::colormapjet::base( double val ) {
    if ( val <= -0.75 ) return 0;
    else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
    else if ( val <= 0.25 ) return 1.0;
    else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
    else return 0.0;
}

double mrsmap::colormapjet::red( double gray ) {
    return base( gray - 0.5 );
}

double mrsmap::colormapjet::green( double gray ) {
    return base( gray );
}

double mrsmap::colormapjet::blue( double gray ) {
    return base( gray + 0.5 );
}


cv::Mat mrsmap::visualizeDepth( const cv::Mat& depthImg, float minDepth, float maxDepth ) {

	cv::Mat img( depthImg.rows, depthImg.cols, CV_8UC3, 0.f );

	const float depthRange = maxDepth - minDepth;
	for( unsigned int y = 0; y < depthImg.rows; y++ ) {

		for( unsigned int x = 0; x < depthImg.cols; x++ ) {

			if( depthImg.at<unsigned short>(y,x) == 0 ) {
				img.at< cv::Vec3b >( y, x ) = cv::Vec3b( 255, 255, 255 );
			}
			else {

				float gray = 2.f * (depthImg.at<unsigned short>(y,x) - minDepth) / depthRange - 1.f;

				float rf = std::min( 1., std::max( 0., mrsmap::colormapjet::red( gray ) ) );
				float gf = std::min( 1., std::max( 0., mrsmap::colormapjet::green( gray ) ) );
				float bf = std::min( 1., std::max( 0., mrsmap::colormapjet::blue( gray ) ) );

				img.at< cv::Vec3b >( y, x ) = cv::Vec3b( 255 * bf, 255 * gf, 255 * rf );

			}

		}

	}

	return img;

}


Eigen::Vector2f mrsmap::pointImagePos( const Eigen::Vector4f& p ) {

	if( boost::math::isnan( p(0) ) )
		return Eigen::Vector2f( p(0), p(0) );

	return Eigen::Vector2f( 525.0 * p(0) / p(2), 525.0 * p(1) / p(2) );

}


bool mrsmap::pointInImage( const Eigen::Vector4f& p ) {

	if( boost::math::isnan( p(0) ) )
		return false;

	double px = 525.0 * p(0) / p(2);
	double py = 525.0 * p(1) / p(2);

	if( px < -320.0 || px > 320.0 || py < -240.0 || py > 240.0 ) {
		return false;
	}

	return true;

}

bool mrsmap::pointInImage( const Eigen::Vector4f& p, const unsigned int imageBorder ) {

	if( boost::math::isnan( p(0) ) )
		return false;

	double px = 525.0 * p(0) / p(2);
	double py = 525.0 * p(1) / p(2);

	if( px < -320.0 + imageBorder || px > 320.0 - imageBorder || py < -240.0 + imageBorder || py > 240.0 - imageBorder ) {
		return false;
	}

	return true;

}

void mrsmap::convertRGB2LAlphaBeta( float r, float g, float b, float& L, float& alpha, float& beta ) {

	static const float sqrt305 = 0.5f*sqrtf(3);

	// RGB to L-alpha-beta:
	// normalize RGB to [0,1]
	// M := max( R, G, B )
	// m := min( R, G, B )
	// L := 0.5 ( M + m )
	// alpha := 0.5 ( 2R - G - B )
	// beta := 0.5 sqrt(3) ( G - B )
	L = 0.5f * ( std::max( std::max( r, g ), b ) + std::min( std::min( r, g ), b ) );
	alpha = 0.5f * ( 2.f*r - g - b );
	beta = sqrt305 * (g-b);

}

void mrsmap::convertLAlphaBeta2RGB( float L, float alpha, float beta, float& r, float& g, float& b ) {

	static const float pi3 = M_PI / 3.f;
	static const float pi3_inv = 1.f / pi3;

	// L-alpha-beta to RGB:
	// the mean should not lie beyond the RGB [0,1] range
	// sampled points could lie beyond, so we transform first to HSL,
	// "saturate" there, and then transform back to RGB
	// H = atan2(beta,alpha)
	// C = sqrt( alpha*alpha + beta*beta)
	// S = C / (1 - abs(2L-1))
	// saturate S' [0,1], L' [0,1]
	// C' = (1-abs(2L-1)) S'
	// X = C' (1- abs( (H/60) mod 2 - 1 ))
	// calculate luminance-free R' G' B'
	// m := L - 0.5 C
	// R, G, B := R1+m, G1+m, B1+m

	float h = atan2f( beta, alpha );
	float c = std::max( 0.f, std::min( 1.f, sqrtf( alpha*alpha + beta*beta ) ) );
	float s_norm = (1.f-fabsf(2.f*L - 1.f));
	float s = 0.f;
	if( s_norm > 1e-4f ) {
		s = std::max( 0.f, std::min( 1.f, c / s_norm ) );
		c = s_norm * s;
	}
	else
		c = 0.f;

	if( h < 0 )
		h += 2.f*M_PI;
	float h2 = pi3_inv * h;
	float h_sector = h2 - 2.f*floor(0.5f*h2);
	float x = c * (1.f-fabsf( h_sector-1.f ));

	float r1 = 0, g1 = 0, b1 = 0;
	if( h2 >= 0.f && h2 < 1.f )
		r1 = c, g1 = x;
	else if( h2 >= 1.f && h2 < 2.f )
		r1 = x, g1 = c;
	else if( h2 >= 2.f && h2 < 3.f )
		g1 = c, b1 = x;
	else if( h2 >= 3.f && h2 < 4.f )
		g1 = x, b1 = c;
	else if( h2 >= 4.f && h2 < 5.f )
		r1 = x, b1 = c;
	else
		r1 = c, b1 = x;

	float m = L - 0.5f * c;
	r = r1+m;
	b = b1+m;
	g = g1+m;

}


void mrsmap::convertLAlphaBeta2RGBDamped( float L, float alpha, float beta, float& r, float& g, float& b ) {

	static const float pi3 = M_PI / 3.f;
	static const float pi3_inv = 1.f / pi3;

	// L-alpha-beta to RGB:
	// the mean should not lie beyond the RGB [0,1] range
	// sampled points could lie beyond, so we transform first to HSL,
	// "saturate" there, and then transform back to RGB
	// H = atan2(beta,alpha)
	// C = sqrt( alpha*alpha + beta*beta)
	// S = C / (1 - abs(2L-1))
	// saturate S' [0,1], L' [0,1]
	// C' = (1-abs(2L-1)) S'
	// X = C' (1- abs( (H/60) mod 2 - 1 ))
	// calculate luminance-free R' G' B'
	// m := L - 0.5 C
	// R, G, B := R1+m, G1+m, B1+m

	float h = atan2f( beta, alpha );
	float c = std::max( 0.f, std::min( 1.f, sqrtf( alpha*alpha + beta*beta ) ) );
	float s_norm = (1.f-fabsf(2.f*L - 1.f));
	float s = 0.f;
	if( s_norm > 1e-4f ) {
		s = std::max( 0.f, std::min( 1.f, c / s_norm ) );
		// damp saturation stronger when lightness is bad
		s *= expf( -0.5f * 10.f * (L-0.5f) * (L-0.5f) );
		c = s_norm * s;
	}
	else
		c = 0.f;



	if( h < 0 )
		h += 2.f*M_PI;
	float h2 = pi3_inv * h;
	float h_sector = h2 - 2.f*floor(0.5f*h2);
	float x = c * (1.f-fabsf( h_sector-1.f ));

	float r1 = 0, g1 = 0, b1 = 0;
	if( h2 >= 0.f && h2 < 1.f )
		r1 = c, g1 = x;
	else if( h2 >= 1.f && h2 < 2.f )
		r1 = x, g1 = c;
	else if( h2 >= 2.f && h2 < 3.f )
		g1 = c, b1 = x;
	else if( h2 >= 3.f && h2 < 4.f )
		g1 = x, b1 = c;
	else if( h2 >= 4.f && h2 < 5.f )
		r1 = x, b1 = c;
	else
		r1 = c, b1 = x;

	float m = L - 0.5f * c;
	r = r1+m;
	b = b1+m;
	g = g1+m;

}


cv::Mat mrsmap::visualizeAlphaBetaPlane( float L, unsigned int imgSize ) {
	if( imgSize % 2 == 0 ) imgSize += 1;
	cv::Mat img( imgSize, imgSize, CV_8UC3, 0.f );

	const int radius = (imgSize-1) / 2;
	const float sqrt305 = 0.5f*sqrtf(3.f);

	for( int a = -radius; a <= radius; a++ ) {

		float alpha = (float)a / (float)radius;

		for( int b = -radius; b <= radius; b++ ) {

			float beta = (float)b / (float)radius;

			float rf, gf, bf;
			convertLAlphaBeta2RGB( L, alpha, beta, rf, gf, bf );
			img.at< cv::Vec3b >( radius-b, a+radius ) = cv::Vec3b( 255*bf, 255*gf, 255*rf );

		}

	}

	return img;
}



void mrsmap::imagesToPointCloud( const cv::Mat& depthImg, const cv::Mat& colorImg, const std::string& timeStamp, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling ) {

	cloud->header.frame_id = "openni_rgb_optical_frame";
	cloud->is_dense = true;
	cloud->height = depthImg.rows / downsampling;
	cloud->width = depthImg.cols / downsampling;
	cloud->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 1.f );
	cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
	cloud->points.resize( cloud->height*cloud->width );

	const float invfocalLength = 1.f / 525.f;
	const float centerX = 319.5f;
	const float centerY = 239.5f;
	const float factor = 1.f / 5000.f;

	const unsigned short* depthdata = reinterpret_cast<const unsigned short*>( &depthImg.data[0] );
	const unsigned char* colordata = &colorImg.data[0];
	int idx = 0;
	for( unsigned int y = 0; y < depthImg.rows; y++ ) {
		for( unsigned int x = 0; x < depthImg.cols; x++ ) {

			if( x % downsampling != 0 || y % downsampling != 0 ) {
				colordata += 3;
				depthdata++;
				continue;
			}

			pcl::PointXYZRGB& p = cloud->points[idx];

			if( *depthdata == 0 ) { //|| factor * (float)(*depthdata) > 10.f ) {
				p.x = std::numeric_limits<float>::quiet_NaN();
				p.y = std::numeric_limits<float>::quiet_NaN();
				p.z = std::numeric_limits<float>::quiet_NaN();
			}
			else {
				float xf = x;
				float yf = y;
				float dist = factor * (float)(*depthdata);
				p.x = (xf-centerX) * dist * invfocalLength;
				p.y = (yf-centerY) * dist * invfocalLength;
				p.z = dist;
			}

			depthdata++;

			int b = (*colordata++);
			int g = (*colordata++);
			int r = (*colordata++);

			int rgb = ( r << 16 ) + ( g << 8 ) + b;
			p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

			idx++;


		}
	}

}


double mrsmap::averageDepth( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud ) {

	double sum = 0.0;
	double num = 0.0;
	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			if( !boost::math::isnan( p.z ) ) {
				sum += p.z;
				num += 1.0;
			}

			idx++;

		}
	}

	return sum / num;

}


double mrsmap::medianDepth( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud ) {

	std::vector< double > depths;
	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			if( !boost::math::isnan( p.z ) ) {
				depths.push_back(p.z);
			}

			idx++;

		}
	}

	std::sort( depths.begin(), depths.end() );
	depths.push_back(0);
	return depths[depths.size()/2];

}


void mrsmap::imagesToPointCloudUnorganized( const cv::Mat& depthImg, const cv::Mat& colorImg, const std::string& timeStamp, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling ) {

	cloud->header.frame_id = "openni_rgb_optical_frame";
	cloud->is_dense = false;
	cloud->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 1.f );
	cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
	cloud->points.reserve( cloud->height*cloud->width );

	const float invfocalLength = 1.f / 525.f;
	const float centerX = 319.5f;
	const float centerY = 239.5f;
	const float factor = 1.f / 5000.f;

	const unsigned short* depthdata = reinterpret_cast<const unsigned short*>( &depthImg.data[0] );
	const unsigned char* colordata = &colorImg.data[0];
	int idx = 0;
	for( unsigned int y = 0; y < depthImg.rows; y++ ) {
		for( unsigned int x = 0; x < depthImg.cols; x++ ) {

			if( x % downsampling != 0 || y % downsampling != 0 ) {
				colordata += 3;
				depthdata++;
				continue;
			}

			pcl::PointXYZRGB p;

			if( *depthdata == 0 || factor * (float)(*depthdata) > 5.f ) {
			}
			else {
				float xf = x;
				float yf = y;
				float dist = factor * (float)(*depthdata);
				p.x = (xf-centerX) * dist * invfocalLength;
				p.y = (yf-centerY) * dist * invfocalLength;
				p.z = dist;
			}

			depthdata++;

			int b = (*colordata++);
			int g = (*colordata++);
			int r = (*colordata++);

			int rgb = ( r << 16 ) + ( g << 8 ) + b;
			p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

			cloud->push_back( p );

			idx++;


		}
	}

	cloud->width = cloud->points.size();

}


void mrsmap::getCameraCalibration( cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs ) {

	distortionCoeffs = cv::Mat( 1, 5, CV_32FC1, 0.f );
	cameraMatrix = cv::Mat( 3, 3, CV_32FC1, 0.f );

	cameraMatrix.at<float>(0,0) = 525.f;
	cameraMatrix.at<float>(1,1) = 525.f;
	cameraMatrix.at<float>(2,2) = 1.f;

	cameraMatrix.at<float>(0,2) = 319.5f;
	cameraMatrix.at<float>(1,2) = 239.5f;

}


void mrsmap::pointCloudToImage( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img ) {

	img = cv::Mat( cloud->height, cloud->width, CV_8UC3, 0.f );

	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			cv::Vec3b px;
			px[0] = p.b;
			px[1] = p.g;
			px[2] = p.r;

			img.at< cv::Vec3b >( y, x ) = px;

			idx++;

		}
	}

}


void mrsmap::pointCloudToImages( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img_rgb, cv::Mat& img_depth ) {

	img_rgb = cv::Mat( cloud->height, cloud->width, CV_8UC3, 0.f );
	img_depth = cv::Mat( cloud->height, cloud->width, CV_32FC1, 0.f );

	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			cv::Vec3b px;
			px[0] = p.b;
			px[1] = p.g;
			px[2] = p.r;

			img_rgb.at< cv::Vec3b >( y, x ) = px;
			img_depth.at< float >( y, x ) = p.z;

			idx++;

		}
	}

}



void mrsmap::reprojectPointCloudToImages( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img_rgb, cv::Mat& img_depth ) {

//	cv::Mat mapX( cloud->height, cloud->width, CV_32FC1, 0.f );
//	cv::Mat mapY( cloud->height, cloud->width, CV_32FC1, 0.f );

//	img_depth = cv::Mat( cloud->height, cloud->width, CV_16UC1, 0.f );

//	cv::Mat orig_img_rgb( cloud->height, cloud->width, CV_8UC3, 0.f );
//	cv::Mat orig_img_depth( cloud->height, cloud->width, CV_16UC1, 0.f );

	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			cv::Vec3b px;
			px[0] = p.b;
			px[1] = p.g;
			px[2] = p.r;

//			orig_img_rgb.at< cv::Vec3b >( y, x ) = px;

			if( boost::math::isnan( p.x ) ) {
//				mapX.at< float >( y, x ) = -1.f;
//				mapY.at< float >( y, x ) = -1.f;
//				orig_img_depth.at< unsigned short >( y, x ) = 0;
			}
			else {

				float nx = p.x / p.z * 525.f + 319.5f;
				float ny = p.y / p.z * 525.f + 239.5f;

//				mapX.at< float >( y, x ) = nx;
//				mapY.at< float >( y, x ) = ny;

//				if( p.z > 0.2 )
//					orig_img_depth.at< unsigned short >( y, x ) = p.z*5000.f;


				int ix = boost::math::round(nx);
				int iy = boost::math::round(ny);

				if( p.z > 0.2f && ix >= 0 && ix < img_rgb.cols && iy >= 0 && iy < img_rgb.rows ) {
					img_depth.at< unsigned short >( iy, ix ) = p.z * 5000.f;
					img_rgb.at< cv::Vec3b >( iy, ix ) = px;
				}
			}


			idx++;

		}
	}

//	cv::remap( orig_img_rgb, img_rgb, mapX, mapY, CV_INTER_LINEAR );
//	cv::remap( orig_img_depth, img_depth, mapX, mapY, CV_INTER_NN );

}


void mrsmap::reprojectPointCloudToImagesF( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img_rgb, cv::Mat& img_depth ) {

	int idx = 0;
	for( unsigned int y = 0; y < cloud->height; y++ ) {
		for( unsigned int x = 0; x < cloud->width; x++ ) {

			const pcl::PointXYZRGB& p = cloud->points[idx];

			cv::Vec3b px;
			px[0] = p.b;
			px[1] = p.g;
			px[2] = p.r;

			if( boost::math::isnan( p.x ) ) {
			}
			else {

				float nx = p.x / p.z * 525.f + 319.5f;
				float ny = p.y / p.z * 525.f + 239.5f;

				int ix = boost::math::round(nx);
				int iy = boost::math::round(ny);

				if( p.z > 0.2f && ix >= 0 && ix < img_rgb.cols && iy >= 0 && iy < img_rgb.rows ) {
					img_depth.at< float >( iy, ix ) = p.z;
					img_rgb.at< cv::Vec3b >( iy, ix ) = px;
				}
			}


			idx++;

		}
	}

}


void mrsmap::pointCloudsToOverlayImage( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& rgb_cloud, const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& overlay_cloud, cv::Mat& img ) {

	img = cv::Mat( rgb_cloud->height, rgb_cloud->width, CV_8UC3, 0.f );

	float alpha = 0.2;

	int idx = 0;
	for( unsigned int y = 0; y < rgb_cloud->height; y++ ) {
		for( unsigned int x = 0; x < rgb_cloud->width; x++ ) {

			const pcl::PointXYZRGB& p1 = rgb_cloud->points[idx];
			const pcl::PointXYZRGB& p2 = overlay_cloud->points[idx];

			cv::Vec3b px;
			px[0] = (1-alpha) * p1.b + alpha * p2.b;
			px[1] = (1-alpha) * p1.g + alpha * p2.g;
			px[2] = (1-alpha) * p1.r + alpha * p2.r;

			img.at< cv::Vec3b >( y, x ) = px;

			idx++;

		}
	}

}



void mrsmap::downsamplePointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling ) {

	cloudOut = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );

	cloudOut->header = cloudIn->header;
	cloudOut->is_dense = cloudIn->is_dense;
	cloudOut->width = cloudIn->width / downsampling;
	cloudOut->height = cloudIn->height / downsampling;
	cloudOut->sensor_origin_ = cloudIn->sensor_origin_;
	cloudOut->sensor_orientation_ = cloudIn->sensor_orientation_;

	cloudOut->points.resize( cloudOut->width*cloudOut->height );

	unsigned int idx = 0;
	for( unsigned int y = 0; y < cloudIn->height; y++ ) {

		if( y % downsampling != 0 )
			continue;

		for( unsigned int x = 0; x < cloudIn->width; x++ ) {

			if( x % downsampling != 0 )
				continue;

			cloudOut->points[idx++] = cloudIn->points[ y*cloudIn->width + x ];

		}
	}



}


void mrsmap::downsamplePointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling ) {

	cloudOut = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );

	cloudOut->header = cloudIn->header;
	cloudOut->is_dense = cloudIn->is_dense;
	cloudOut->width = cloudIn->width / downsampling;
	cloudOut->height = cloudIn->height / downsampling;
	cloudOut->sensor_origin_ = cloudIn->sensor_origin_;
	cloudOut->sensor_orientation_ = cloudIn->sensor_orientation_;

	cloudOut->points.resize( cloudOut->width*cloudOut->height );

	unsigned int idx = 0;
	for( unsigned int y = 0; y < cloudIn->height; y++ ) {

		if( y % downsampling != 0 )
			continue;

		for( unsigned int x = 0; x < cloudIn->width; x++ ) {

			if( x % downsampling != 0 )
				continue;

			cloudOut->points[idx++] = cloudIn->points[ y*cloudIn->width + x ];

		}
	}



}



void mrsmap::downsamplePointCloudMean( const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling ) {

	cloudOut = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );

	cloudOut->header = cloudIn->header;
	cloudOut->is_dense = cloudIn->is_dense;
	cloudOut->width = cloudIn->width / downsampling;
	cloudOut->height = cloudIn->height / downsampling;
	cloudOut->sensor_origin_ = cloudIn->sensor_origin_;
	cloudOut->sensor_orientation_ = cloudIn->sensor_orientation_;

	cloudOut->points.resize( cloudOut->width*cloudOut->height );

	unsigned int idx = 0;
	float numIntegratedPos = 0.f;
	float numIntegratedCol = 0.f;
	Eigen::Matrix<float, 6, 1> point = Eigen::Matrix<float, 6, 1>::Zero();
	for( unsigned int y = 0; y < cloudIn->height; y++ ) {

//		if( y % downsampling != 0 )
//			continue;

		for( unsigned int x = 0; x < cloudIn->width; x++ ) {

//			if( x % downsampling != 0 )
//				continue;

			if( x % downsampling == 0 && y % downsampling == 0 ) {

				if( numIntegratedPos == 0.f ) {

					cloudOut->points[idx].x = std::numeric_limits<float>::quiet_NaN();
					cloudOut->points[idx].y = std::numeric_limits<float>::quiet_NaN();
					cloudOut->points[idx].z = std::numeric_limits<float>::quiet_NaN();

				}
				else {

					cloudOut->points[idx].x = point(0) / numIntegratedPos;
					cloudOut->points[idx].y = point(1) / numIntegratedPos;
					cloudOut->points[idx].z = point(2) / numIntegratedPos;

				}

				cloudOut->points[idx].r = std::max( 0, std::min( 255, (int)(point(3) / numIntegratedCol) ) );
				cloudOut->points[idx].g = std::max( 0, std::min( 255, (int)(point(4) / numIntegratedCol) ) );
				cloudOut->points[idx].b = std::max( 0, std::min( 255, (int)(point(5) / numIntegratedCol) ) );

				point = Eigen::Matrix<float, 6, 1>::Zero();
				numIntegratedPos = 0.f;
				numIntegratedCol = 0.f;

				idx++;

			}


			if( !boost::math::isnan( cloudIn->points[y*cloudIn->width+x].x ) ) {

				point(0) += cloudIn->points[y*cloudIn->width+x].x;
				point(1) += cloudIn->points[y*cloudIn->width+x].y;
				point(2) += cloudIn->points[y*cloudIn->width+x].z;
				numIntegratedPos += 1.f;

			}

			point(3) += cloudIn->points[y*cloudIn->width+x].r;
			point(4) += cloudIn->points[y*cloudIn->width+x].g;
			point(5) += cloudIn->points[y*cloudIn->width+x].b;
			numIntegratedCol += 1.f;

		}
	}



}


void mrsmap::downsamplePointCloudMean( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling ) {

	cloudOut = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );

	cloudOut->header = cloudIn->header;
	cloudOut->is_dense = cloudIn->is_dense;
	cloudOut->width = cloudIn->width / downsampling;
	cloudOut->height = cloudIn->height / downsampling;
	cloudOut->sensor_origin_ = cloudIn->sensor_origin_;
	cloudOut->sensor_orientation_ = cloudIn->sensor_orientation_;

	cloudOut->points.resize( cloudOut->width*cloudOut->height );

	unsigned int idx = 0;
	float numIntegratedPos = 0.f;
	float numIntegratedCol = 0.f;
	Eigen::Matrix<float, 6, 1> point = Eigen::Matrix<float, 6, 1>::Zero();
	for( unsigned int y = 0; y < cloudIn->height; y++ ) {

//		if( y % downsampling != 0 )
//			continue;

		for( unsigned int x = 0; x < cloudIn->width; x++ ) {

//			if( x % downsampling != 0 )
//				continue;

			if( x % downsampling == 0 && y % downsampling == 0 ) {

				if( numIntegratedPos == 0.f ) {

					cloudOut->points[idx].x = std::numeric_limits<float>::quiet_NaN();
					cloudOut->points[idx].y = std::numeric_limits<float>::quiet_NaN();
					cloudOut->points[idx].z = std::numeric_limits<float>::quiet_NaN();

				}
				else {

					cloudOut->points[idx].x = point(0) / numIntegratedPos;
					cloudOut->points[idx].y = point(1) / numIntegratedPos;
					cloudOut->points[idx].z = point(2) / numIntegratedPos;

				}

				cloudOut->points[idx].r = std::max( 0, std::min( 255, (int)(point(3) / numIntegratedCol) ) );
				cloudOut->points[idx].g = std::max( 0, std::min( 255, (int)(point(4) / numIntegratedCol) ) );
				cloudOut->points[idx].b = std::max( 0, std::min( 255, (int)(point(5) / numIntegratedCol) ) );

				point = Eigen::Matrix<float, 6, 1>::Zero();
				numIntegratedPos = 0.f;
				numIntegratedCol = 0.f;

				idx++;

			}


			if( !boost::math::isnan( cloudIn->points[y*cloudIn->width+x].x ) ) {

				point(0) += cloudIn->points[y*cloudIn->width+x].x;
				point(1) += cloudIn->points[y*cloudIn->width+x].y;
				point(2) += cloudIn->points[y*cloudIn->width+x].z;
				numIntegratedPos += 1.f;

			}

			point(3) += cloudIn->points[y*cloudIn->width+x].r;
			point(4) += cloudIn->points[y*cloudIn->width+x].g;
			point(5) += cloudIn->points[y*cloudIn->width+x].b;
			numIntegratedCol += 1.f;

		}
	}



}



void mrsmap::downsamplePointCloudClosest( const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling ) {

	cloudOut = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );

	cloudOut->header = cloudIn->header;
	cloudOut->is_dense = cloudIn->is_dense;
	cloudOut->width = cloudIn->width / downsampling;
	cloudOut->height = cloudIn->height / downsampling;
	cloudOut->sensor_origin_ = cloudIn->sensor_origin_;
	cloudOut->sensor_orientation_ = cloudIn->sensor_orientation_;

	cloudOut->points.resize( cloudOut->width*cloudOut->height );

	int idx = -1;
	float closestDepth = std::numeric_limits<float>::max();
	Eigen::Matrix<float, 6, 1> point = Eigen::Matrix<float, 6, 1>::Zero();
	for( unsigned int y = 0; y < cloudIn->height; y++ ) {

		for( unsigned int x = 0; x < cloudIn->width; x++ ) {

			if( x % downsampling == 0 && y % downsampling == 0 ) {


				idx++;

				cloudOut->points[idx].x = cloudIn->points[y*cloudIn->width+x].x;
				cloudOut->points[idx].y = cloudIn->points[y*cloudIn->width+x].y;
				cloudOut->points[idx].z = cloudIn->points[y*cloudIn->width+x].z;

				cloudOut->points[idx].r = cloudIn->points[y*cloudIn->width+x].r;
				cloudOut->points[idx].g = cloudIn->points[y*cloudIn->width+x].g;
				cloudOut->points[idx].b = cloudIn->points[y*cloudIn->width+x].b;

				closestDepth = std::numeric_limits<float>::max();

			}

			if( !boost::math::isnan( cloudIn->points[y*cloudIn->width+x].x ) && cloudIn->points[y*cloudIn->width+x].z < closestDepth ) {

				cloudOut->points[idx].x = cloudIn->points[y*cloudIn->width+x].x;
				cloudOut->points[idx].y = cloudIn->points[y*cloudIn->width+x].y;
				cloudOut->points[idx].z = cloudIn->points[y*cloudIn->width+x].z;

				cloudOut->points[idx].r = cloudIn->points[y*cloudIn->width+x].r;
				cloudOut->points[idx].g = cloudIn->points[y*cloudIn->width+x].g;
				cloudOut->points[idx].b = cloudIn->points[y*cloudIn->width+x].b;

			}

		}
	}



}



void mrsmap::fillDepthFromRight( cv::Mat& imgDepth ) {

	for( unsigned int y = 0; y < imgDepth.rows; y++ ) {

		for( int x = imgDepth.cols-2; x >= 0; x-- ) {

			unsigned short& d = imgDepth.at< unsigned short >( y, x );
			if( d == 0 )
				d = imgDepth.at< unsigned short >( y, x+1 );

		}

	}

}


void mrsmap::fillDepthFromLeft( cv::Mat& imgDepth ) {

	for( unsigned int y = 0; y < imgDepth.rows; y++ ) {

		for( unsigned int x = 1; x < imgDepth.cols; x++ ) {

			unsigned short& d = imgDepth.at< unsigned short >( y, x );
			if( d == 0 )
				d = imgDepth.at< unsigned short >( y, x-1 );

		}

	}

}


void mrsmap::fillDepthFromTop( cv::Mat& imgDepth ) {

	for( unsigned int y = 1; y < imgDepth.rows; y++ ) {

		for( unsigned int x = 0; x < imgDepth.cols; x++ ) {

			unsigned short& d = imgDepth.at< unsigned short >( y, x );
			if( d == 0 )
				d = imgDepth.at< unsigned short >( y-1, x );

		}

	}

}


void mrsmap::fillDepthFromBottom( cv::Mat& imgDepth ) {

	for( int y = imgDepth.rows-2; y >= 0; y-- ) {

		for( unsigned int x = 0; x < imgDepth.cols; x++ ) {

			unsigned short& d = imgDepth.at< unsigned short >( y, x );
			if( d == 0 )
				d = imgDepth.at< unsigned short >( y+1, x );

		}

	}

}



void mrsmap::poseToTransform(const Eigen::Matrix<double, 7, 1> & pose, Eigen::Matrix4d & transform) {
	transform.setIdentity();
    transform.block<3, 3>(0, 0) = Eigen::Quaterniond( pose(6), pose(3), pose(4), pose(5) ).matrix();
    transform.block<3, 1>(0, 3) = pose.block<3, 1>(0, 0);
}

static inline double mrsmap::sampleUniform( gsl_rng* rng, double min, double max ) {
    return min + gsl_rng_uniform( rng ) * ( max - min );
}

static inline Eigen::Vector3d mrsmap::sampleVectorGaussian( gsl_rng* rng, const float variance ) {
    Eigen::Vector3d sample;
    sample(0) = gsl_ran_gaussian( rng, variance );
    sample(1) = gsl_ran_gaussian( rng, variance );
    sample(2) = gsl_ran_gaussian( rng, variance );

    return sample;
}



