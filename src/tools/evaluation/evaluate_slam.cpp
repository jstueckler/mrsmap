/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 02.01.2012
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
#include <opencv2/opencv.hpp>

#include <pcl/common/transforms.h>

#include <Eigen/Core>

#include <boost/algorithm/string.hpp>

#include <boost/thread/thread.hpp>
#include "pcl/common/common_headers.h"
#include "pcl/visualization/pcl_visualizer.h"

#include <mrsmap/slam/slam.h>
#include <mrsmap/visualization/visualization_slam.h>
#include <mrsmap/utilities/utilities.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace mrsmap;

//typedef MultiResolutionColorSurfelMap MultiResolutionSurfelMap;

// parses Juergen Sturm's datasets (tgz archives + timestamp associations)
// simply takes the base path of the dataset

class EvaluateSLAM {
public:

	EvaluateSLAM(  int argc, char** argv  )
			: viewer_( &slam_ ) {


		po::options_description desc("Allowed options");
		desc.add_options()
		    ("help,h", "help")
		    ("inputpath,i", po::value<std::string>(&path_)->default_value("."), "path to input data")
		    ("maxresolution,r", po::value<double>(&min_resolution_)->default_value(0.0125f), "maximum resolution")
		    ("skippastframes,k", po::value<bool>(&skip_past_frames_)->default_value(false), "skip past frames for real-time evaluation")
		    ("usepointfeatures,p", po::value<int>(&use_pointfeatures_)->default_value(0), "use point features")
		    ("downsampling", po::value<int>(&downsampling_)->default_value(1), "downsampling of image for mrsmap")
		    ("debug,d", po::value<int>(&debug_)->default_value(0), "debug visualization")
		;

    	po::variables_map vm;
    	po::store(po::parse_command_line(argc, argv, desc), vm);
    	po::notify(vm);

    	if( vm.count("help") || vm.count("h") ) {
    		std::cout << desc << "\n";
    		exit(0);
    	}

		max_radius_ = 30.f;

		imageAllocator_ = boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator >( new MultiResolutionSurfelMap::ImagePreAllocator() );

		for( int i = 0; i < 2; i++ ) {
			treeNodeAllocator_[ i ] = boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > >(
					new spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue >( 1000 ) );
		}

		graphChanged_ = true;

	}

	class PoseInfo {
	public:
		PoseInfo( const std::string& time, int id, const Eigen::Matrix4d tf )
				: stamp( time ), referenceID( id ), transform( tf ) {
		}
		~PoseInfo() {
		}

		std::string stamp;
		int referenceID;
		Eigen::Matrix4d transform;
		double tracking_time_;
		double connect_time_;
		double generate_keyview_time_;
		double optimize_time_;
		double total_time_;
		unsigned int num_vertices_, num_edges_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		;
	};

	void evaluate() {

		float register_start_resolution = min_resolution_;
		const float register_stop_resolution = 32.f * min_resolution_;

		// parse associations.txt
		std::ifstream assocFile( ( path_ + std::string( "/associations.txt" ) ).c_str() );

		Eigen::Matrix4f totalTransform;
		totalTransform.setIdentity();

		lastTransform_.setIdentity();

		int count = -1;

		std::vector< PoseInfo, Eigen::aligned_allocator< PoseInfo > > trajectoryEstimate;

		double nextTime = 0;


		bool recordFrame = true;

		while( assocFile.good() ) {

			// read in line
			char lineCStr[ 1024 ];
			assocFile.getline( lineCStr, 1024, '\n' );

			count++;

			std::string lineStr( lineCStr );

			// split line at blanks
			std::vector< std::string > entryStrs;
			boost::split( entryStrs, lineStr, boost::is_any_of( "\t " ) );

			// parse entries, load images, generate point cloud, process images...
			if( entryStrs.size() == 4 ) {

//				while( !viewer_.processFrame && viewer_.is_running ) {
//					boost::this_thread::sleep( boost::posix_time::microseconds(10) ); //usleep( 10 );
//				}

				if( count == 1 )
					viewer_.processFrame = false;


				double stamp = 0.0;
				std::stringstream sstr;
				sstr << entryStrs[0];
				sstr >> stamp;

				if( skip_past_frames_ && nextTime > stamp ) {
					std::cout << "================= SKIP =================\n";
					continue;
				}


				// load images
				cv::Mat depthImg = cv::imread( path_ + "/" + entryStrs[ 1 ], CV_LOAD_IMAGE_ANYDEPTH );
				cv::Mat rgbImg = cv::imread( path_ + "/" + entryStrs[ 3 ], CV_LOAD_IMAGE_ANYCOLOR );

				// extract point cloud from image pair
				pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud( new pcl::PointCloud< pcl::PointXYZRGB >() );
				imagesToPointCloud( depthImg, rgbImg, entryStrs[ 0 ], cloud );

	    		// measure time to skip frames
	    		pcl::StopWatch stopwatch;
	    		stopwatch.reset();

	    		unsigned int numEdges = slam_.optimizer_->edges().size();
				unsigned int numVertices = slam_.optimizer_->vertices().size();
				unsigned int referenceID = slam_.referenceKeyFrameId_;

				slam_.params_.usePointFeatures_ = use_pointfeatures_;
				slam_.params_.debugPointFeatures_ = debug_;
				slam_.params_.downsamplingMRSMapImage_ = downsampling_;

				if( skip_past_frames_ )
					slam_.params_.regularizePose_ = true;

				bool retVal = slam_.addImage( rgbImg, cloud, register_start_resolution, register_stop_resolution, min_resolution_, true );

				double deltat = stopwatch.getTimeSeconds() * 1000.0;
				std::cout << "slam iteration took: " << deltat << "\n";

				nextTime = stamp + 0.001 * deltat;

				if( retVal ) {
					// store relative translation to reference keyframe
					PoseInfo pi = PoseInfo( entryStrs[ 0 ], slam_.referenceKeyFrameId_, slam_.lastTransform_ );
					pi.tracking_time_ = slam_.tracking_time_;
					pi.connect_time_ = slam_.connect_time_;
					pi.generate_keyview_time_ = slam_.generate_keyview_time_;
					pi.optimize_time_ = slam_.optimize_time_;
					pi.total_time_ = deltat;
					pi.num_vertices_ = slam_.optimizer_->vertices().size();
					pi.num_edges_ = slam_.optimizer_->edges().size();
					trajectoryEstimate.push_back( pi );
				}

				if( slam_.optimizer_->vertices().size() != numVertices || slam_.optimizer_->edges().size() != numEdges || slam_.referenceKeyFrameId_ != referenceID )
					graphChanged_ = true;


				if( debug_ && slam_.optimizer_->vertices().size() > 0 ) {
					g2o::VertexSE3* v_ref = dynamic_cast< g2o::VertexSE3* >( slam_.optimizer_->vertex( slam_.keyFrames_[ slam_.referenceKeyFrameId_ ]->nodeId_ ) );
					Eigen::Matrix4d pose_ref = v_ref->estimate().matrix();

					viewer_.displayPose( pose_ref * slam_.lastTransform_ );

				}

//				if( !viewer_.is_running )
//					exit( -1 );

				if( debug_ ) {
					if( graphChanged_ || viewer_.forceRedraw ) {
						viewer_.visualizeSLAMGraph();
						viewer_.forceRedraw = false;
					}

					g2o::VertexSE3* v_ref = dynamic_cast< g2o::VertexSE3* >( slam_.optimizer_->vertex( slam_.keyFrames_[ slam_.referenceKeyFrameId_ ]->nodeId_ ) );
					Eigen::Matrix4d pose_ref = v_ref->estimate().matrix();

					pcl::PointCloud< pcl::PointXYZRGB >::Ptr transformedCloud( new pcl::PointCloud< pcl::PointXYZRGB >() );
					pcl::transformPointCloud( *cloud, *transformedCloud, (pose_ref * slam_.lastTransform_).cast<float>() );
					viewer_.displayPointCloud( "scene", transformedCloud );
				}

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

			}

		}

/*
		if( graphChanged_ || viewer_.forceRedraw ) {
			viewer_.visualizeSLAMGraph();
			viewer_.forceRedraw = false;
			viewer_.spinOnce();
		}
*/
		// dump pose estimates to file
		std::ofstream outFile1( ( path_ + "/" + std::string( "slam_result.txt" ) ).c_str() );
		outFile1 << "# minres: " << min_resolution_ << ", max depth: " << max_radius_ << "\n";
		for( unsigned int i = 0; i < trajectoryEstimate.size(); i++ ) {

			g2o::VertexSE3* v_curr = dynamic_cast< g2o::VertexSE3* >( slam_.optimizer_->vertex( slam_.keyFrames_[ trajectoryEstimate[ i ].referenceID ]->nodeId_ ) );
			Eigen::Matrix4d vtransform = v_curr->estimate().matrix();

			Eigen::Matrix4d transform = vtransform * trajectoryEstimate[ i ].transform;

			Eigen::Quaterniond q( Eigen::Matrix3d( transform.block< 3, 3 >( 0, 0 ) ) );
			outFile1 << trajectoryEstimate[ i ].stamp << " " << transform( 0, 3 ) << " " << transform( 1, 3 ) << " " << transform( 2, 3 ) << " " << q.x() << " " << q.y() << " " << q.z() << " "
					<< q.w() << " " << trajectoryEstimate[ i ].tracking_time_ << " " << trajectoryEstimate[ i ].generate_keyview_time_ << " " << trajectoryEstimate[ i ].connect_time_ << " " << trajectoryEstimate[ i ].optimize_time_ << " " << trajectoryEstimate[ i ].total_time_ << " "
					<< trajectoryEstimate[ i ].num_vertices_ << " " << trajectoryEstimate[ i ].num_edges_ << "\n";

		}

	}

public:

	std::string path_;

	SLAM slam_;

	Eigen::Matrix4d lastTransform_;

	double min_resolution_, max_radius_;

	bool skip_past_frames_;

	boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator > imageAllocator_;
	boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > > treeNodeAllocator_[ 2 ];

	ViewerSLAM viewer_;

	bool graphChanged_;

	int use_pointfeatures_, downsampling_, debug_;

};

int main( int argc, char** argv ) {

	EvaluateSLAM ev( argc, argv );
	ev.evaluate();

	while( ev.viewer_.is_running ) {

		if( ev.viewer_.forceRedraw ) {
			ev.viewer_.visualizeSLAMGraph();
			ev.viewer_.forceRedraw = false;
		}

		ev.viewer_.spinOnce();
		boost::this_thread::sleep( boost::posix_time::milliseconds(1) ); //usleep( 1000 );
	}

	return 0;
}

