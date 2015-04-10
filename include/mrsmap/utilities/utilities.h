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

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <set>

#include <Eigen/Core>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <mrsmap/mrsmap_api.h>

namespace mrsmap {

	const int UNASSIGNED_LABEL = -1;
	const int OUTLIER_LABEL = -2;

	namespace colormapjet {
		double interpolate( double val, double y0, double x0, double y1, double x1 );
		double base( double val );
		double red( double gray );
		double green( double gray );
		double blue( double gray );
	}

	cv::Mat visualizeDepth( const cv::Mat& depthImg, float minDepth, float maxDepth );

	bool MRSMAP_API pointInImage( const Eigen::Vector4f& p );
	bool MRSMAP_API pointInImage( const Eigen::Vector4f& p, const unsigned int imageBorder );
	Eigen::Vector2f MRSMAP_API pointImagePos( const Eigen::Vector4f& p );

	void MRSMAP_API convertRGB2LAlphaBeta( float r, float g, float b, float& L, float& alpha, float& beta );
	void MRSMAP_API convertLAlphaBeta2RGB( float L, float alpha, float beta, float& r, float& g, float& b );
	void MRSMAP_API convertLAlphaBeta2RGBDamped( float L, float alpha, float beta, float& r, float& g, float& b );

	cv::Mat MRSMAP_API visualizeAlphaBetaPlane( float L, unsigned int imgSize );

	void MRSMAP_API imagesToPointCloud( const cv::Mat& depthImg, const cv::Mat& colorImg, const std::string& timeStamp, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling = 1 );
	void MRSMAP_API imagesToPointCloudUnorganized( const cv::Mat& depthImg, const cv::Mat& colorImg, const std::string& timeStamp, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling = 1 );

	void MRSMAP_API pointCloudToImage( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img );
	void MRSMAP_API pointCloudToImages( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img_rgb, cv::Mat& img_depth );
	void MRSMAP_API reprojectPointCloudToImages( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img_rgb, cv::Mat& img_depth );
	void MRSMAP_API reprojectPointCloudToImagesF( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud, cv::Mat& img_rgb, cv::Mat& img_depth );

	void MRSMAP_API pointCloudsToOverlayImage( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& rgb_cloud, const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& overlay_cloud, cv::Mat& img );


	void MRSMAP_API downsamplePointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling );
	void MRSMAP_API downsamplePointCloud( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling );

	void MRSMAP_API downsamplePointCloudMean( const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling );
	void MRSMAP_API downsamplePointCloudMean( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling );

	void MRSMAP_API downsamplePointCloudClosest( const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling );
	void MRSMAP_API downsamplePointCloudClosest( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloudIn, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloudOut, unsigned int downsampling );

	double MRSMAP_API averageDepth( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud );
	double MRSMAP_API medianDepth( const pcl::PointCloud< pcl::PointXYZRGB >::ConstPtr& cloud );

	void MRSMAP_API getCameraCalibration( cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs );


	void MRSMAP_API fillDepthFromRight( cv::Mat& imgDepth );
	void MRSMAP_API fillDepthFromLeft( cv::Mat& imgDepth );
	void MRSMAP_API fillDepthFromTop( cv::Mat& imgDepth );
	void MRSMAP_API fillDepthFromBottom( cv::Mat& imgDepth );


	void MRSMAP_API poseToTransform(const Eigen::Matrix<double, 7, 1> & pose, Eigen::Matrix4d & transform);

	static inline double sampleUniform( gsl_rng* rng, double min, double max );

	static inline Eigen::Vector3d sampleVectorGaussian( gsl_rng* rng, const float variance );


};



#endif /* UTILITIES_H_ */



