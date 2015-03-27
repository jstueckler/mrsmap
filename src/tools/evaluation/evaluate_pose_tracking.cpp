/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 16.09.2011
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


#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>

#include <pcl/segmentation/extract_polygonal_prism_data.h>
#define uchar flann_uchar
#include <pcl/surface/convex_hull.h>
#undef uchar //Prevent ambiguous symbol error when OpenCV defines uchar

#include <mrsmap/map/multiresolution_surfel_map.h>
#include <mrsmap/registration/multiresolution_surfel_registration.h>

#include <boost/thread/thread.hpp>
#include "pcl/common/common_headers.h"
#include "pcl/visualization/pcl_visualizer.h"

#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <mrsmap/visualization/visualization_map.h>
#include <mrsmap/utilities/utilities.h>


using namespace std;
using namespace mrsmap;


//typedef MultiResolutionColorSurfelMap MultiResolutionSurfelMap;



class PoseInfo {
public:
	PoseInfo( const std::string& time, int id, const Eigen::Matrix4d tf )
	: stamp( time ), referenceID( id ), transform( tf ) {}
	~PoseInfo() {}

	std::string stamp;
	int referenceID;
	Eigen::Matrix4d transform;
	double tracking_time_;
	double connect_time_;
	double generate_keyview_time_;
	double optimize_time_;
	double total_time_;
	unsigned int num_vertices_, num_edges_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

class EvaluatePoseTracking
{
public:

	EvaluatePoseTracking( int argc, char** argv ) {

		imageAllocator_ = boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator >( new MultiResolutionSurfelMap::ImagePreAllocator() );
		treeNodeAllocator_ = boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > >( new spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue >( 10000 ) );

		po::options_description desc("Allowed options");
		desc.add_options()
		    ("help,h", "help")
		    ("object,o", po::value<std::string>(&object_name_)->default_value("object"), "object map file")
		    ("inputpath,i", po::value<std::string>(&input_path_)->default_value("."), "path to input data")
		    ("startframe,s", po::value<int>(&start_frame_)->default_value(0), "start frame")
		    ("endframe,e", po::value<int>(&end_frame_)->default_value(1000), "end frame")
		    ("skippastframes,k", po::value<bool>(&skip_past_frames_)->default_value(false), "skip past frames for real-time evaluation")
		    ("debug,d", po::value<int>(&debug_)->default_value(0), "debug visualization")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if( vm.count("help") || vm.count("h") ) {
		std::cout << desc << "\n";
		exit(0);
	}

    }


    void load() {

	std::cout << "loading " << object_name_.c_str() << "\n";

	// prepare map
		map_ = new MultiResolutionSurfelMap( 0.05f, 30.f );

		map_->load( object_name_ );
		map_->octree_->root_->establishNeighbors();
		map_->buildShapeTextureFeatures();

		map_->extents( map_mean_, map_cov_inv_ );
		map_cov_inv_ = map_cov_inv_.inverse().eval();


		std::cout << "ready\n";

		return;

    }


    void evaluate() {

	// prepare output file
	// memorize parameters

		// parse groundtruth.txt to get initial transform
		std::ifstream gtFile( (input_path_ + std::string("/groundtruth.txt")).c_str() );
		while( gtFile.good() ) {

			// read in line
			char lineCStr[1024];
			gtFile.getline( lineCStr, 1024, '\n' );

			std::string lineStr( lineCStr );

			// split line at blanks
			std::vector< std::string > entryStrs;
			boost::split( entryStrs, lineStr, boost::is_any_of("\t ") );

			if( entryStrs.size() != 8 )
				continue;

			std::stringstream sstr;
			sstr << entryStrs[0];
			double stamp = 0.0;
			sstr >> stamp;

			double tx,ty,tz,qx,qy,qz,qw;
			sstr.clear();
			sstr << entryStrs[1];
			sstr >> tx;
			sstr.clear();
			sstr << entryStrs[2];
			sstr >> ty;
			sstr.clear();
			sstr << entryStrs[3];
			sstr >> tz;
			sstr.clear();
			sstr << entryStrs[4];
			sstr >> qx;
			sstr.clear();
			sstr << entryStrs[5];
			sstr >> qy;
			sstr.clear();
			sstr << entryStrs[6];
			sstr >> qz;
			sstr.clear();
			sstr << entryStrs[7];
			sstr >> qw;

			Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
			transform.block<3,3>(0,0) = Eigen::Quaterniond( qw, qx, qy, qz ).matrix();
			transform.block<3,1>(0,3) = Eigen::Vector3d( tx, ty, tz );

			groundTruth_[stamp] = transform;

		}


	// parse associations.txt
	std::ifstream assocFile( (input_path_ + std::string("/associations.txt")).c_str() );
		std::ofstream fileEstimate( (input_path_ + std::string("/pose_estimate.txt")).c_str() );

	std::vector< std::vector< std::string > > assocs;

	int count = -1;

	unsigned int frameIdx = 0;

	double nextTime = 0;


	bool recordFrame = true;

	while( assocFile.good() ) {

		count++;

		// read in line
		char lineCStr[1024];
		assocFile.getline( lineCStr, 1024, '\n' );

		std::string lineStr( lineCStr );

		// split line at blanks
			std::vector< std::string > entryStrs;
			boost::split( entryStrs, lineStr, boost::is_any_of("\t ") );

			// parse entries, load images, generate point cloud, process images...
			if( entryStrs.size() == 4 ) {

//			while( !viewer_.processFrame && viewer_.is_running ) {
//							boost::this_thread::sleep( boost::posix_time::microseconds(10) ); //usleep(10);
//			}

				if( count == 1 )
					viewer_.processFrame = false;

				// load images
				cv::Mat depthImg = cv::imread( input_path_ + "/" + entryStrs[1], CV_LOAD_IMAGE_ANYDEPTH );
				cv::Mat rgbImg = cv::imread( input_path_ + "/" + entryStrs[3], CV_LOAD_IMAGE_ANYCOLOR );

				if( frameIdx >= start_frame_ && frameIdx <= end_frame_ ) {

					std::cout << "processing frame " << frameIdx << "\n";

					double stamp = 0.0;
					std::stringstream sstr;
					sstr << entryStrs[0];
					sstr >> stamp;

					if( frameIdx == start_frame_ ) {
						referenceTransform_.setIdentity();

						GTMap::iterator it = groundTruth_.upper_bound( stamp );
						if( it != groundTruth_.end() ) {
							referenceTransform_ = it->second;
							std::cout << "initialized reference frame from ground truth\n";
							std::cout << referenceTransform_;
						}
					}

					if( skip_past_frames_ && nextTime > stamp ) {
						std::cout << "================= SKIP =================\n";
						frameIdx++;
						continue;
					}

					// extract point cloud from image pair
					pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud( new pcl::PointCloud< pcl::PointXYZRGB >() );
					imagesToPointCloud( depthImg, rgbImg, entryStrs[0], cloud );

					double deltat = processFrame( cloud );

					nextTime = stamp + 0.001 * deltat;


					Eigen::Vector3d t_est( referenceTransform_(0,3), referenceTransform_(1,3), referenceTransform_(2,3) );
					double dist_est = (t_est - map_mean_).norm();

					// dump estimated and ground truth transform
					Eigen::Quaterniond q( referenceTransform_.block<3,3>(0,0) );
					fileEstimate << fixed << setprecision(10) << entryStrs[0] << " " << referenceTransform_(0,3) << " " << referenceTransform_(1,3) << " "  << referenceTransform_(2,3) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " << dist_est << " " << deltat << "\n";

				}

				if( frameIdx == end_frame_ )
					break;

//				if( !viewer_.is_running )
//					exit(-1);

				if( recordFrame ) {
					static unsigned int frameId = 0;
					char frameStr[255];
					sprintf( frameStr, "slam%05d.png", frameId++ );
					viewer_.viewer->saveScreenshot( frameStr );
				}

				if( debug_ ) {
					viewer_.spinOnce();
					boost::this_thread::sleep( boost::posix_time::milliseconds(1) ); //usleep( 1000 );
				}

				frameIdx++;

			}
	}

    }


    double processFrame( pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud ) {

		// transform point cloud to reference frame
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr transformedCloud( new pcl::PointCloud< pcl::PointXYZRGB >() );
		pcl::transformPointCloud( *cloud, *transformedCloud, referenceTransform_.cast<float>() );

		// consider image borders
		std::vector< int > imageBorderIndices;
		map_->findVirtualBorderPoints( *transformedCloud, imageBorderIndices );

		Eigen::Vector3d t_est( referenceTransform_(0,3), referenceTransform_(1,3), referenceTransform_(2,3) );
		double dist_est = (t_est - map_mean_).norm();
		double covfactor = std::max( 1.5, 0.5*dist_est );

		// concentrate on window at last object pose estimate
		for( unsigned int i = 0; i < transformedCloud->points.size(); i++ ) {

			const pcl::PointXYZRGB& p = transformedCloud->points[i];
			if( boost::math::isnan( p.x ) ) {
				continue;
			}

			Eigen::Vector3d point( p.x, p.y, p.z );
			if( (point-map_mean_).dot( map_cov_inv_ * (point-map_mean_) ) > covfactor*8.0 ) {
				transformedCloud->points[i].x = std::numeric_limits< float >::quiet_NaN();
			}

		}

		transformedCloud->sensor_origin_ = referenceTransform_.block<4,1>(0,3).cast<float>();
		transformedCloud->sensor_orientation_ = Eigen::Quaterniond( referenceTransform_.block<3,3>(0,0) ).cast<float>();

		if( viewer_.displayScene ) {

			viewer_.displayPointCloud( "current frame", transformedCloud );

		}
		else
			viewer_.viewer->removePointCloud( "current frame" );


		pcl::StopWatch stopwatch;
		stopwatch.reset();

		// add points to local map
		treeNodeAllocator_->reset();
		MultiResolutionSurfelMap target( map_->min_resolution_, map_->max_range_, treeNodeAllocator_ );
		target.imageAllocator_ = imageAllocator_;
		target.addImage( *transformedCloud, false );
		target.octree_->root_->establishNeighbors();
		target.markNoUpdateAtPoints( *transformedCloud, imageBorderIndices );
		target.evaluateSurfels();
		target.buildShapeTextureFeatures();


		Eigen::Matrix4d incTransform = Eigen::Matrix4d::Identity();
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
		MultiResolutionSurfelRegistration reg;
		reg.estimateTransformation( *map_, target, incTransform, 16.f * map_->min_resolution_, map_->min_resolution_, corrSrc, corrTgt, 20, 0, 5 );

		double deltat = stopwatch.getTimeSeconds() * 1000.0;
		std::cout << "registration took: " << deltat << "\n";

		referenceTransform_ = (incTransform * referenceTransform_).eval();


		if( viewer_.displayMap ) {
			pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud2 = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );
			map_->visualize3DColorDistribution( cloud2, viewer_.selectedDepth, viewer_.selectedViewDir, false );
			viewer_.displayPointCloud( "map cloud", cloud2 );
		}
		else
			viewer_.viewer->removePointCloud( "map cloud" );


		return deltat;

    }


public:

	std::string input_path_;
	std::string object_name_;
    double minHeight_, maxHeight_;

    MultiResolutionSurfelMap* map_;
    Eigen::Matrix< double, 3, 1 > map_mean_;
    Eigen::Matrix< double, 3, 3 > map_cov_inv_;

    boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator > imageAllocator_;
    boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > > treeNodeAllocator_;

    double min_resolution_, max_range_;
    int start_frame_, end_frame_;

    bool skip_past_frames_;

    Eigen::Matrix4d initialTransform_, referenceTransform_;

    Viewer viewer_;

    typedef std::map< double, Eigen::Matrix4d, std::less< double >, Eigen::aligned_allocator< std::pair< const double, Eigen::Matrix4d > > > GTMap;
    GTMap groundTruth_;

	int debug_;

};


int main(int argc, char** argv) {

	EvaluatePoseTracking ept( argc, argv );
	ept.load();
	ept.evaluate();

	return 0;
}

