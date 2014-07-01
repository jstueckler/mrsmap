/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 14.11.2011
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

#include <mrsmap/map/multiresolution_surfel_map.h>
#include <mrsmap/registration/multiresolution_surfel_registration.h>

#include <mrsmap/visualization/visualization_map.h>
#include <mrsmap/utilities/utilities.h>

#include <boost/algorithm/string.hpp>

#include <boost/thread/thread.hpp>
#include "pcl/common/common_headers.h"
#include "pcl/visualization/pcl_visualizer.h"


#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <sys/types.h>
#include <sys/stat.h>


using namespace mrsmap;

// parses Juergen Sturm's datasets (tgz archives + timestamp associations)
// simply takes the base path of the dataset


class EvaluateVisualOdometry
{
public:

	EvaluateVisualOdometry( const std::string& path, int K, int intermediateskips, int usefeatures, int usecolor, int usepointfeatures, int usesurfels, int mapprops, int downsampling, int debug, double dist_dep ) {

		path_ = path;
		K_ = K;
		intermediateSkips_ = intermediateskips;
		usefeatures_ = usefeatures;
		usecolor_ = usecolor;
		mapprops_ = mapprops;
		usepointfeatures_ = usepointfeatures;
		usesurfels_ = usesurfels;
		downsampling_ = downsampling;
		debug_ = debug;
		dist_dep_ = dist_dep;

		min_resolution_ = 0.0125f;
		max_radius_ = 30.f;

		alloc_idx_ = 0;

		for( int i = 0; i < 2; i++ ) {
			imageAllocator_[i] = boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator >( new MultiResolutionSurfelMap::ImagePreAllocator() );
			treeNodeAllocator_[i] = boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > >( new spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue >( 1000 ) );
		}


    }


    Eigen::Matrix4f processFrame( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud ) {

    	pcl::StopWatch stopwatch;

    	alloc_idx_ = (alloc_idx_+1) % 2;

		// prepare map
		// provide dynamic node allocator
		// use double buffers for image node and tree node allocators
    	treeNodeAllocator_[alloc_idx_]->reset();
    	boost::shared_ptr< MultiResolutionSurfelMap > currFrameMap = boost::shared_ptr< MultiResolutionSurfelMap >( new MultiResolutionSurfelMap( min_resolution_, max_radius_, treeNodeAllocator_[alloc_idx_] ) );


		// add points to local map
//		std::vector< int > contourIndices;
		std::vector< int > imageBorderIndices;
//		currFrameMap->findContourPoints( *cloud, contourIndices );

		currFrameMap->imageAllocator_ = imageAllocator_[alloc_idx_];

		currFrameMap->params_.dist_dependency = dist_dep_;


		if( usesurfels_ ) {
			stopwatch.reset();

			pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudMRS = cloud;
			if( downsampling_ > 1 )
				downsamplePointCloud( cloud, cloudMRS, downsampling_ );

			currFrameMap->addImage( *cloudMRS, false, false );
			t_construct_image_tree_ = stopwatch.getTimeSeconds() * 1000.0;

			stopwatch.reset();
			currFrameMap->octree_->root_->establishNeighbors();
			t_precompute_neighbors_ = stopwatch.getTimeSeconds() * 1000.0;

			stopwatch.reset();
			currFrameMap->findVirtualBorderPoints( *cloudMRS, imageBorderIndices );
			currFrameMap->markNoUpdateAtPoints( *cloudMRS, imageBorderIndices );
			t_virtual_borders_ = stopwatch.getTimeSeconds() * 1000.0;

	//		currFrameMap->findImageBorderPoints( *cloud, imageBorderIndices );
	//		currFrameMap->markNoUpdateAtPoints( *cloud, imageBorderIndices );
			stopwatch.reset();
			currFrameMap->evaluateSurfels();
			t_precompute_surfels_ = stopwatch.getTimeSeconds() * 1000.0;

			std::cout << "build: " << t_construct_image_tree_+t_precompute_neighbors_+t_virtual_borders_+t_precompute_surfels_ << "\n";

			if( usefeatures_ ) {
				stopwatch.reset();
				currFrameMap->buildShapeTextureFeatures();
				t_shapetexture_features_ = stopwatch.getTimeSeconds() * 1000.0;
				std::cout << "feature: " << t_shapetexture_features_ << "\n";
			}

			stopwatch.reset();
			currFrameMap->findForegroundBorderPoints( *cloudMRS, imageBorderIndices );
			currFrameMap->markBorderAtPoints( *cloudMRS, imageBorderIndices );
			t_foreground_borders_ = stopwatch.getTimeSeconds() * 1000.0;

		}

		if( usepointfeatures_ ) {
			currFrameMap->params_.debugPointFeatures = debug_;
			stopwatch.reset();
			currFrameMap->addImagePointFeatures( img_rgb, *cloud );
			t_point_features_ = stopwatch.getTimeSeconds() * 1000.0;
			std::cout << "point features: " << t_point_features_ << "\n";
		}

		// register frames
		Eigen::Matrix4d transform;
		transform.setIdentity();
//		transform = lastTransform_;



		if( lastFrameMap_ ) {

			stopwatch.reset();
			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
			MultiResolutionSurfelRegistration reg;

			reg.params_.use_features_ = usefeatures_;
			reg.params_.registration_use_color_ = usecolor_;

			reg.params_.registerFeatures_ = usepointfeatures_;
			reg.params_.registerSurfels_ = usesurfels_;

			reg.params_.debugFeatures_ = false;//debug_;

//			for( unsigned int i = 0; i < 20; i++ )
//				reg.estimateTransformation( *lastFrameMap_, *currFrameMap, transform, 32.f * currFrameMap->min_resolution_, currFrameMap->min_resolution_, corrSrc, corrTgt, 10, 0, 0 );
			reg.estimateTransformation( *lastFrameMap_, *currFrameMap, transform, 32.f * currFrameMap->min_resolution_, currFrameMap->min_resolution_, corrSrc, corrTgt, 100, 0, 5 );


			double deltat = stopwatch.getTime();
			std::cout << "register: " << deltat << "\n";


		}

		lastFrameMap_ = currFrameMap;
		lastTransform_ = transform;

		return transform.cast<float>();

    }


    void evaluate() {

    	// prepare output file
    	// memorize parameters

    	// parse associations.txt
    	std::ifstream assocFile( (path_ + std::string("/associations.txt")).c_str() );

    	Eigen::Matrix4f totalTransform;
    	totalTransform.setIdentity();

    	lastTransform_.setIdentity();

    	std::vector< std::vector< std::string > > assocs;

    	int count = -1;

		std::cout << path_ << "\n";

    	while( assocFile.good() ) {

    		count++;

    		// read in line
    		char lineCStr[1024];
    		assocFile.getline( lineCStr, 1024, '\n' );

    		std::string lineStr( lineCStr );

    		// split line at blanks
			std::vector< std::string > entryStrs;
			boost::split( entryStrs, lineStr, boost::is_any_of("\t ") );

			if( entryStrs.size() == 4 )
				assocs.push_back( entryStrs );

    	}

    	int k_start = K_;
    	if( intermediateSkips_ )
    		k_start = 0;

		std::ofstream outFileMap( (path_ + "/mrsmap_map_properties.txt").c_str() );

    	for( unsigned int k = k_start; k <= K_; k++ ) {

			lastFrameMap_.reset();
			totalTransform.setIdentity();
			lastTransform_.setIdentity();

			char filenum[255];
			sprintf(filenum,"%i",k+1);

			std::ofstream outFile( (path_ + "/" + "mrsmap_" + (usesurfels_ ? "" : "nosurfels_") + (usefeatures_ ? "" : "nofeatures_") + (usecolor_ ? "" : "nocolor_") + (usepointfeatures_ ? "pf_" : "") + std::string("visual_odometry_result_delta") + std::string(filenum) + ".txt").c_str() );
			outFile << "# minres: " << min_resolution_ << ", max depth: " << max_radius_ << "\n";


//			unsigned int t_k = 0;

			for( unsigned int t_k = 0; t_k <= k; t_k++ ) {

				for( unsigned int t = t_k; t < assocs.size(); t+=k+1 ) {

					std::vector< std::string > entryStrs = assocs[t];

					// parse entries, load images, generate point cloud, process images...

//					cv::Mat alphabetaplane = visualizeAlphaBetaPlane( 0.75f, 512);
//					cv::imwrite( "alphabeta.png", alphabetaplane );

					if( debug_ ) {
						bool once = false;
						while( !once || !viewer_.processFrame ) {

							// display last point cloud
							if( lastFrameMap_ ) {

								if( viewer_.displayFeatureSimilarity ) {

									Eigen::Vector4f pos = viewer_.selectedPoint.getVector4fMap();
									spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* n = lastFrameMap_->octree_->findRepresentative( pos, viewer_.selectedDepth );

									pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudMap( new pcl::PointCloud< pcl::PointXYZRGB >() );

	//								lastFrameMap_->visualizeSimilarity( n, cloudMap, n->depth_, viewer_.selectedViewDir, false );
									lastFrameMap_->visualizeBorders( cloudMap, viewer_.selectedDepth, viewer_.selectedViewDir, false );

									pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud2( new pcl::PointCloud< pcl::PointXYZRGB >() );
									pcl::transformPointCloud( *cloudMap, *cloud2, totalTransform );
									viewer_.displayPointCloud( "map cloud", cloud2 );

								}
								else {

									pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudMap( new pcl::PointCloud< pcl::PointXYZRGB >() );

									lastFrameMap_->visualize3DColorDistribution( cloudMap, viewer_.selectedDepth, viewer_.selectedViewDir, false );

									pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud2( new pcl::PointCloud< pcl::PointXYZRGB >() );
									pcl::transformPointCloud( *cloudMap, *cloud2, totalTransform );
									viewer_.displayPointCloud( "map cloud", cloud2 );

//									pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud2( new pcl::PointCloud< pcl::PointXYZRGB >() );
//									pcl::transformPointCloud( *lastFrameCloud_, *cloud2, totalTransform );
//									viewer_.displayPointCloud( "map cloud", cloud2, 1 );
//
//									viewer_.displayPose( totalTransform.cast<double>() );
//
//									static unsigned int frameId = 0;
//									char frameStr[255];
//									sprintf( frameStr, "/home/stueckler/.ros/vo%05d.png", frameId++ );
//
//									Eigen::Matrix4d lastTotalTransform = totalTransform.cast<double>() * lastTransform_.inverse();
//									pcl::PointXYZ p1, p2;
//									p1.x = totalTransform(0,3);
//									p1.y = totalTransform(1,3);
//									p1.z = totalTransform(2,3);
//									p2.x = lastTotalTransform(0,3);
//									p2.y = lastTotalTransform(1,3);
//									p2.z = lastTotalTransform(2,3);
//									viewer_.viewer->addLine( p1, p2, std::string( frameStr ) );
//									viewer_.viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, std::string( frameStr ) );
//
//
//
//									viewer_.viewer->saveScreenshot( frameStr );
//
//									std::cout << "frame " << frameId << "\n";

								}

		//						pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudMap( new pcl::PointCloud< pcl::PointXYZRGBNormal >() );
		//
		//						lastFrameMap_->visualize3DColorDistributionWithNormals( cloudMap, -1, viewer_.selectedViewDir, false );
		////						lastFrameMap_->visualize3DColorDistributionWithNormals( cloudMap, viewer_.selectedDepth, viewer_.selectedViewDir, false );
		//
		//						pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloud2( new pcl::PointCloud< pcl::PointXYZRGBNormal >() );
		//						pcl::transformPointCloudWithNormals( *cloudMap, *cloud2, totalTransform );
		//
		//						float resolution = lastFrameMap_->min_resolution_;
		//						if( viewer_.selectedDepth != -1 ) {
		//							resolution = lastFrameMap_->octree_->resolutions_[viewer_.selectedDepth];
		//						}
		////						viewer_.displayMesh( "map cloud", cloud2, 1.f / resolution * 5.f );
		//						viewer_.displayMesh( "map cloud", cloud2, resolution );
		////						viewer_.displaySurfaceNormals( "map cloud", cloud2 );


							}
							viewer_.spinOnce();
							usleep(1000);
							once = true;

						}
					}


//					if( t != 400 )
//						continue;

					// load images
					cv::Mat depthImg = cv::imread( path_ + "/" + entryStrs[1], CV_LOAD_IMAGE_ANYDEPTH );
					cv::Mat rgbImg = cv::imread( path_ + "/" + entryStrs[3], CV_LOAD_IMAGE_ANYCOLOR );

//					cv::Mat depthVizImg = mrsmap::visualizeDepth( depthImg, 0.f, 5.f*5000.f );
////					cv::imshow( "depth", depthVizImg );
////					cv::waitKey(20);
//					cv::imwrite( "depth400.png", depthVizImg );


//					cv::Mat lImg( rgbImg.rows, rgbImg.cols, CV_8UC1 );
//					cv::Mat aImg( rgbImg.rows, rgbImg.cols, CV_8UC1 );
//					cv::Mat bImg( rgbImg.rows, rgbImg.cols, CV_8UC1 );
//
//					const float inv_255 = 1.f / 255.f;
//					const float sqrt305 = 0.5f*sqrtf(3.f);
//					const unsigned char* colordata = &rgbImg.data[0];
//					for( unsigned int y = 0; y < depthImg.rows; y++ ) {
//						for( unsigned int x = 0; x < depthImg.cols; x++ ) {
//
//							int b = (*colordata++);
//							int g = (*colordata++);
//							int r = (*colordata++);
//
//							// HSL by Luminance and Cartesian Hue-Saturation (L-alpha-beta)
//							float rf = inv_255*r, gf = inv_255*g, bf = inv_255*b;
//
//							// RGB to L-alpha-beta:
//							float L = 0.5f * ( std::max( std::max( rf, gf ), bf ) + std::min( std::min( rf, gf ), bf ) );
//							float alpha = 0.5f * ( 2.f*rf - gf - bf );
//							float beta = sqrt305 * (gf-bf);
//
//							lImg.at<unsigned char>(y,x) = std::max( 0.f, std::min( 255.f, 255*L ) );
//							aImg.at<unsigned char>(y,x) = std::max( 0.f, std::min( 255.f, 255*(0.5f+alpha) ) );
//							bImg.at<unsigned char>(y,x) = std::max( 0.f, std::min( 255.f, 255*(0.5f+beta) ) );
//
//						}
//					}
//
////					cv::imshow( "rgb", rgbImg );
////					cv::imshow( "l", lImg );
////					cv::imshow( "a", aImg );
////					cv::imshow( "b", bImg );
////					cv::waitKey(20);
//
//					char lstr[255];
//					sprintf( lstr, "imgl_%i.png", t );
//					char astr[255];
//					sprintf( astr, "imga_%i.png", t );
//					char bstr[255];
//					sprintf( bstr, "imgb_%i.png", t );
//
//					cv::imwrite( lstr, lImg );
//					cv::imwrite( astr, aImg );
//					cv::imwrite( bstr, bImg );




//					cv::imshow(depthImg);
//					cv::waitKey(20);

					// extract point cloud from image pair
					pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud( new pcl::PointCloud< pcl::PointXYZRGB >() );
					imagesToPointCloud( depthImg, rgbImg, entryStrs[0], cloud );


					// process data
					pcl::StopWatch stopwatch;
					stopwatch.reset();
					Eigen::Matrix4f transform = processFrame( rgbImg, cloud );
					double processTime = stopwatch.getTimeSeconds() * 1000.0;
					totalTransform = (totalTransform * transform).eval();

					std::cout << "total: " << processTime << "\n";

					// write transform to output file
					Eigen::Quaternionf q( Eigen::Matrix3f( totalTransform.block<3,3>(0,0) ) );
					outFile << entryStrs[0] << " " << totalTransform(0,3) << " " << totalTransform(1,3) << " " << totalTransform(2,3) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " << processTime << "\n";

					if( mapprops_ ) {
						lastFrameMap_->indexNodes( 0, 16, true );
						unsigned int numNodes = lastFrameMap_->indexedNodes_.size();

						algorithm::OcTreeSamplingVectorMap<float, MultiResolutionSurfelMap::NodeValue> target_sampling_map = algorithm::downsampleVectorOcTree(*lastFrameMap_->octree_, false, lastFrameMap_->octree_->max_depth_);

						lastFrameMap_->save("tmp.map");
						struct stat filestatus;
						stat( "tmp.map", &filestatus );
						unsigned int filesize = filestatus.st_size;

						double avgDepth = mrsmap::averageDepth( cloud );
						double medianDepth = mrsmap::medianDepth( cloud );

						outFileMap << t << " " << entryStrs[0] << " " << lastFrameMap_->min_resolution_ << " " << numNodes << " " << avgDepth << " " << medianDepth << " " << filesize << " " << t_construct_image_tree_ << " " << t_precompute_neighbors_ << " " << t_virtual_borders_ << " " << t_precompute_surfels_ << " " << t_shapetexture_features_ << " " << t_foreground_borders_ << " " << t_point_features_ << " ";

						for( unsigned int d = 0; d <= lastFrameMap_->octree_->max_depth_; d++ )
							outFileMap << d << " " << lastFrameMap_->octree_->resolutions_[d] << " " << target_sampling_map[d].size() << " ";

						outFileMap << "\n";

					}

					lastFrameCloud_ = cloud;



				}

			}

    	}

    }


public:

	std::string path_;
	int K_;
	int intermediateSkips_;
	int usefeatures_;
	int usecolor_;
	int usepointfeatures_;
	int usesurfels_;
	int mapprops_;
	int debug_;
	int downsampling_;
	double dist_dep_;

    boost::shared_ptr< MultiResolutionSurfelMap > lastFrameMap_;
    pcl::PointCloud< pcl::PointXYZRGB >::Ptr lastFrameCloud_;

    Eigen::Matrix4d lastTransform_;

    double t_construct_image_tree_, t_precompute_neighbors_, t_virtual_borders_, t_precompute_surfels_, t_shapetexture_features_, t_point_features_, t_foreground_borders_;

    float min_resolution_, max_radius_;

    unsigned int alloc_idx_;
    boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator > imageAllocator_[2];
    boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > > treeNodeAllocator_[2];

    Viewer viewer_;

};


int main(int argc, char** argv) {

	po::options_description desc("Allowed options");

	std::string inputpath = "";
	int frameskips = 0;
	int intermediateskips = 0;
	int mapprops = 0;
	int usefeatures = 1;
	int usepointfeatures = 0;
	int usesurfels = 1;
	int usecolor = 1;
	int downsampling = 1;
	int debug = 1;
	double lambda = 0.01;

	desc.add_options()
	    ("help,h", "help")
	    ("inputpath,i", po::value<std::string>(&inputpath)->default_value("."), "path to input data")
	    ("frameskips,s", po::value<int>(&frameskips)->default_value(0), "number of skipped frames")
	    ("intermediateskips,m", po::value<int>(&intermediateskips)->default_value(0), "evaluate intermediate frame skips")
	    ("usefeatures,f", po::value<int>(&usefeatures)->default_value(1), "use shape-texture features")
	    ("usecolor,c", po::value<int>(&usecolor)->default_value(1), "use color")
	    ("usepointfeatures,p", po::value<int>(&usepointfeatures)->default_value(0), "use point features")
	    ("usesurfels", po::value<int>(&usesurfels)->default_value(1), "use surfels")
	    ("mapprops", po::value<int>(&mapprops)->default_value(0), "evaluate map properties")
	    ("downsampling", po::value<int>(&downsampling)->default_value(1), "downsampling of image for mrsmap")
	    ("lambda,l", po::value<double>(&lambda)->default_value(0.01), "distance dependency (lambda)")
	    ("debug,d", po::value<int>(&debug)->default_value(0), "debug visualization")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if( vm.count("help") || vm.count("h") ) {
		std::cout << desc << "\n";
		exit(0);
	}


	EvaluateVisualOdometry ev( inputpath, frameskips, intermediateskips, usefeatures, usecolor, usepointfeatures, usesurfels, mapprops, downsampling, debug, lambda );
	ev.evaluate();

	if( debug ) {
		while( ev.viewer_.is_running ) {
			ev.viewer_.spinOnce();
			usleep(1000);
		}
	}

	return 0;
}



