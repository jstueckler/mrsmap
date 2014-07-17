/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 12.12.2011
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

#include <mrsmap/slam/slam.h>

#include <pcl/registration/transforms.h>


#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/surface/convex_hull.h>

#include <g2o/core/optimization_algorithm_levenberg.h>

#include <mrsmap/utilities/utilities.h>



using namespace mrsmap;

#define GRADIENT_ITS 100
#define NEWTON_FEAT_ITS 0
#define NEWTON_ITS 5

#define REGISTER_TWICE 0

#define SOFT_REGISTRATION 1


#if SOFT_REGISTRATION
#include <mrsmap/registration/multiresolution_soft_surfel_registration.h>
#else
#include <mrsmap/registration/multiresolution_surfel_registration.h>
#endif

SLAM::Params::Params() {
	usePointFeatures_ = false;
	debugPointFeatures_ = false;
	downsamplingMRSMapImage_ = 1;
	regularizePose_ = true;
	map_dist_dependency_ = 0.01;
	connectRandom_ = true;
	loglikelihood_threshold_ = -80000.0;
	optimize_ = true;
	optimize_full_trajectory_ = false;

	pose_close_angle_ = 0.2;
	pose_close_dist_ = 0.3;
	pose_far_angle_ = 0.4;
	pose_far_dist_ = 0.7;
}


SLAM::SLAM() {

	srand(time(NULL));

	for( unsigned int i = 0; i < 2; i++ ) {
		imageAllocator_[i] = boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator >( new MultiResolutionSurfelMap::ImagePreAllocator() );
		treeNodeAllocator_[i] = boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > >( new spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue >( 1000 ) );
	}
	currentAllocIdx_ = 0;


	referenceKeyFrameId_ = 0;
	lastTransform_.setIdentity();
	lastFrameTransform_.setIdentity();

	// allocating the optimizer
	optimizer_ = new g2o::SparseOptimizer();
	optimizer_->setVerbose(true);
	SlamLinearSolver* linearSolver = new SlamLinearSolver();
	linearSolver->setBlockOrdering(false);
	SlamBlockSolver* solver = new SlamBlockSolver(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solverLevenberg = new g2o::OptimizationAlgorithmLevenberg(solver);

	optimizer_->setAlgorithm( solverLevenberg );

	deltaIncTransform_.setIdentity();
}

SLAM::~SLAM() {

	delete optimizer_;

}

unsigned int SLAM::addKeyFrame( unsigned int kf_prev_id, boost::shared_ptr< KeyFrame >& keyFrame, const Eigen::Matrix4d& transform ) {

	keyFrame->nodeId_ = optimizer_->vertices().size();

	// anchor first frame at origin
	if( keyFrames_.empty() ){

		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId( keyFrame->nodeId_ );
		v->setEstimate( g2o::SE3Quat() );
		v->setFixed( true );
		optimizer_->addVertex( v );
		keyFrames_.push_back( keyFrame );
		keyFrameNodeMap_[ keyFrame->nodeId_ ] = keyFrame;

	}
	else {

		g2o::SE3Quat measurement_mean( Eigen::Quaterniond( transform.block<3,3>(0,0) ), transform.block<3,1>(0,3) );

		g2o::VertexSE3* v_prev = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[ kf_prev_id ]->nodeId_ ) );

		// create vertex in slam graph for new key frame
		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId( keyFrame->nodeId_ );
		v->setEstimate( v_prev->estimate() * measurement_mean );
		optimizer_->addVertex( v );
		keyFrames_.push_back( keyFrame );
		keyFrameNodeMap_[ keyFrame->nodeId_ ] = keyFrame;

	}

	return keyFrames_.size()-1;

}



unsigned int SLAM::addIntermediateFrame( unsigned int kf_ref_id, boost::shared_ptr< IntermediateFrame >& intermediateFrame, const Eigen::Matrix4d& transform ) {

	intermediateFrame->nodeId_ = optimizer_->vertices().size();

	// anchor first frame at origin
	if( keyFrames_.empty() ){

		std::cerr << "ERROR: first frame should not be intermediate frame!\n";
		exit( -1 );
		return 0;

	}
	else {

		g2o::SE3Quat measurement_mean( Eigen::Quaterniond( transform.block<3,3>(0,0) ), transform.block<3,1>(0,3) );

		g2o::VertexSE3* v_prev = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[ kf_ref_id ]->nodeId_ ) );

		// create vertex in slam graph for new key frame
		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId( intermediateFrame->nodeId_ );
		v->setEstimate( v_prev->estimate() * measurement_mean );
		optimizer_->addVertex( v );

	}

	return intermediateFrames_.size()-1;

}


bool SLAM::addEdge( unsigned int v1_id, unsigned int v2_id, float register_start_resolution, float register_stop_resolution, bool checkMatchingLikelihood ) {

	g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
	g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

	// diff transform from v2 to v1
	Eigen::Matrix4d diffTransform = (v1->estimate().inverse() * v2->estimate()).matrix();

	// add edge to graph
	return addEdge( v1_id, v2_id, diffTransform, register_start_resolution, register_stop_resolution, checkMatchingLikelihood );

}


bool SLAM::addEdge( unsigned int v1_id, unsigned int v2_id, const Eigen::Matrix4d& transformGuess, float register_start_resolution, float register_stop_resolution, bool checkMatchingLikelihood ) {

	g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
	g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

	Eigen::Matrix4d transform = transformGuess;

	// register maps with pose guess from graph
	Eigen::Matrix< double, 6, 6 > poseCov;

	pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;

#if SOFT_REGISTRATION

	MultiResolutionSoftSurfelRegistration reg;
	bool retVal = reg.estimateTransformation( *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), transform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS );

	if( REGISTER_TWICE )
		retVal = reg.estimateTransformation( *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), transform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS );
	if( !retVal )
		return false;


	if( checkMatchingLikelihood ) {

		std::cout << "warning: slam checkMatchingLikelihood not implemented for soft reg" << std::endl;

	}


	retVal = reg.estimatePoseCovarianceLM( poseCov, *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), transform, register_start_resolution, register_stop_resolution );


	if( !retVal )
		return false;

#else

	MultiResolutionSurfelRegistration reg;
	reg.params_.registerFeatures_ = params_.usePointFeatures_;
	reg.params_.debugFeatures_ = params_.debugPointFeatures_;
	bool retVal = reg.estimateTransformation( *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), transform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );

	if( REGISTER_TWICE )
		retVal = reg.estimateTransformation( *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), transform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
	if( !retVal )
		return false;


	if( checkMatchingLikelihood ) {

		Eigen::Matrix4d transforminv = transform.inverse();
		double logLikelihood1 = reg.matchLogLikelihood( *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), transform );
		std::cout << "new edge likelihood1: " << logLikelihood1 << "\n";

		double logLikelihood2 = reg.matchLogLikelihood( *(keyFrameNodeMap_[v2_id]->map_), *(keyFrameNodeMap_[v1_id]->map_), transforminv );
		std::cout << "new edge likelihood2: " << logLikelihood2 << "\n";

		double baseLogLikelihood1 = keyFrameNodeMap_[v1_id]->sumLogLikelihood_ / keyFrameNodeMap_[v1_id]->numEdges_;
		double baseLogLikelihood2 = keyFrameNodeMap_[v2_id]->sumLogLikelihood_ / keyFrameNodeMap_[v2_id]->numEdges_;
		std::cout << "key frame1 base log likelihood is " << baseLogLikelihood1 << "\n";
		std::cout << "key frame2 base log likelihood is " << baseLogLikelihood2 << "\n";


		if( logLikelihood1 < baseLogLikelihood1 + params_.loglikelihood_threshold_ || logLikelihood2 < baseLogLikelihood2 + params_.loglikelihood_threshold_ ) {
			std::cout << "============= BAD MATCHING LIKELIHOOD ============\n";
			return false;
		}
	}


	retVal = reg.estimatePoseCovariance( poseCov, *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), transform, register_start_resolution, register_stop_resolution );


	if( !retVal )
		return false;

#endif

	// add edge to graph
	return addEdge( v1_id, v2_id, transform, poseCov );

}


// returns true, iff node could be added to the cloud
bool SLAM::addEdge( unsigned int v1_id, unsigned int v2_id, const Eigen::Matrix4d& transform, const Eigen::Matrix< double, 6, 6 >& covariance ) {

	unsigned int edges = optimizer_->edges().size();

	g2o::SE3Quat measurement_mean( Eigen::Quaterniond( transform.block<3,3>(0,0) ), transform.block<3,1>(0,3) );
	Eigen::Matrix< double, 6, 6 > measurement_information = covariance.inverse();

	g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
	g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

	// create edge between new key frame and previous key frame with the estimated transformation
	g2o::EdgeSE3* edge = new g2o::EdgeSE3();
	edge->vertices()[0] = v1;
	edge->vertices()[1] = v2;
	edge->setMeasurement( measurement_mean );
	edge->setInformation( measurement_information );

	return optimizer_->addEdge( edge );

}


bool SLAM::poseIsClose( const Eigen::Matrix4d& transform ) {

	double angle = Eigen::AngleAxisd( transform.block<3,3>(0,0) ).angle();
	double dist = transform.block<3,1>(0,3).norm();

//	return fabsf( angle ) + dist < 0.2f;
//	return fabsf( angle ) < 0.2f && dist < 0.3f;
//	return fabsf( angle ) < 0.1f && dist < 0.1f;
	return fabsf( angle ) < params_.pose_close_angle_ && dist < params_.pose_close_dist_;
}

bool SLAM::poseIsFar( const Eigen::Matrix4d& transform ) {

	double angle = Eigen::AngleAxisd( transform.block<3,3>(0,0) ).angle();
	double dist = transform.block<3,1>(0,3).norm();

//	return fabsf( angle ) > 0.2f || dist > 0.2f;
	return fabsf( angle ) > params_.pose_far_angle_ || dist > params_.pose_far_dist_;
}

bool SLAM::addImage( const cv::Mat& img_rgb, const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& pointCloudIn, float startResolution, float stopResolution, float minResolution, bool storeCloud ) {

	pcl::StopWatch sw;
	sw.reset();

//	cv::imshow( "rgb", img_rgb );
//	cv::waitKey(10);

	const int numPoints = pointCloudIn->points.size();

	std::vector< int > indices( numPoints );
	for( int i = 0; i < numPoints; i++ )
		indices[i] = i;

	const float register_start_resolution = startResolution;
	const float register_stop_resolution = stopResolution;

	const float min_resolution = minResolution;
	const float max_radius = 30.f;

	// slam graph: list of key frames
	// match current frame to last key frame
	// create new key frame after some delta in translation or rotation


	treeNodeAllocator_[currentAllocIdx_]->reset();
	boost::shared_ptr< MultiResolutionSurfelMap > target = boost::shared_ptr< MultiResolutionSurfelMap >( new MultiResolutionSurfelMap( min_resolution, max_radius, treeNodeAllocator_[currentAllocIdx_] ) );
	Eigen::Matrix4d incTransform = lastTransform_ * deltaIncTransform_;

	// add points to local map
	target->imageAllocator_ = imageAllocator_[currentAllocIdx_];

	target->params_.debugPointFeatures = params_.debugPointFeatures_;
	target->params_.dist_dependency = params_.map_dist_dependency_;


	if( params_.downsamplingMRSMapImage_ > 1 ) {
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudDownsampled = pcl::PointCloud<pcl::PointXYZRGB>::Ptr( new pcl::PointCloud<pcl::PointXYZRGB>() );
		mrsmap::downsamplePointCloud( pointCloudIn, cloudDownsampled, params_.downsamplingMRSMapImage_ );
		target->addImage( *cloudDownsampled );
		std::vector< int > imageBorderIndices;
		target->findVirtualBorderPoints( *cloudDownsampled, imageBorderIndices );
		target->markNoUpdateAtPoints( *cloudDownsampled, imageBorderIndices );
	}
	else {

		if( pointCloudIn->height > 1 ) {

			target->addImage( *pointCloudIn );
			std::vector< int > imageBorderIndices;
			target->findVirtualBorderPoints( *pointCloudIn, imageBorderIndices );
			target->markNoUpdateAtPoints( *pointCloudIn, imageBorderIndices );

		}
		else {

			std::vector< int > indices( pointCloudIn->points.size() );
			for( unsigned int i = 0; i < indices.size(); i++ ) {
				indices[i] = i;
			}
			target->addPoints( *pointCloudIn, indices );

		}

	}


	target->evaluateSurfels();
	target->octree_->root_->establishNeighbors();
	target->buildShapeTextureFeatures();

	if( params_.usePointFeatures_ ) {
//		target->params_.pixelNoise = 4.0;
//		target->params_.depthNoiseFactor = 2.0;
		target->addImagePointFeatures( img_rgb, *pointCloudIn );
	}


	bool generateKeyFrame = false;

	Eigen::Matrix4d currFrameTransform = Eigen::Matrix4d::Identity();

	if( keyFrames_.empty() ) {

		generateKeyFrame = true;

	}
	else {

#if SOFT_REGISTRATION

		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
		MultiResolutionSoftSurfelRegistration reg;

		if( params_.regularizePose_ ) {
			Eigen::Matrix< double, 6, 1 > pose_mean, pose_var;
			pose_mean.block<3,1>(0,0) = incTransform.block<3,1>(0,3);
			pose_mean.block<3,1>(3,0) = Eigen::Quaterniond( incTransform.block<3,3>(0,0) ).coeffs().block<3,1>(0,0);
			pose_var = 1.0*Eigen::Matrix< double, 6, 1 >::Ones();
			reg.setPriorPose( true, pose_mean, pose_var );
		}

		bool retVal = reg.estimateTransformation( *(keyFrames_[referenceKeyFrameId_]->map_), *target, incTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS );
		if( REGISTER_TWICE )
			retVal = reg.estimateTransformation( *(keyFrames_[referenceKeyFrameId_]->map_), *target, incTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS );


#else

		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
		MultiResolutionSurfelRegistration reg;
		reg.params_.registerFeatures_ = params_.usePointFeatures_;
		reg.params_.debugFeatures_ = params_.debugPointFeatures_;
	//	reg.params_.pointFeatureMatchingCoarseImagePosMahalDist_ = 2.0*reg.params_.pointFeatureMatchingFineImagePosMahalDist_;

		if( params_.regularizePose_ ) {
			Eigen::Matrix< double, 6, 1 > pose_mean, pose_var;
			pose_mean.block<3,1>(0,0) = incTransform.block<3,1>(0,3);
			pose_mean.block<3,1>(3,0) = Eigen::Quaterniond( incTransform.block<3,3>(0,0) ).coeffs().block<3,1>(0,0);
			pose_var = 1.0*Eigen::Matrix< double, 6, 1 >::Ones();
			reg.setPriorPose( true, pose_mean, pose_var );
		}

		bool retVal = reg.estimateTransformation( *(keyFrames_[referenceKeyFrameId_]->map_), *target, incTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
		if( REGISTER_TWICE )
			retVal = reg.estimateTransformation( *(keyFrames_[referenceKeyFrameId_]->map_), *target, incTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );

#endif

		if( !retVal ) {
			std::cout << "SLAM: lost track in current frame\n";
			incTransform = lastTransform_;
		}

		deltaIncTransform_ = lastTransform_.inverse() * incTransform;

		if( retVal ) {
			lastTransform_ = incTransform;

			// check for sufficient pose delta to generate a new key frame

			if( !poseIsClose( lastTransform_ ) ) {
				generateKeyFrame = true;
			}
		}


	}

	tracking_time_ = sw.getTime();
	std::cout << "creating and registering the image took " << sw.getTime() << "\n";
	sw.reset();


	if( generateKeyFrame ) {

		boost::shared_ptr< KeyFrame > keyFrame = boost::shared_ptr< KeyFrame >( new KeyFrame() );

		if( storeCloud ) {
			keyFrame->cloud_ = pointCloudIn;
			keyFrame->img_rgb = img_rgb;
		}

		keyFrame->map_ = target;

		pcl::StopWatch allocTimer;
		treeNodeAllocator_[currentAllocIdx_] = boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > >( new spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue >( 1000 ) );
		std::cout << "alloc took: " << allocTimer.getTime() << "\n";

		// evaluate pose covariance between keyframes..
		Eigen::Matrix< double, 6, 6 > poseCov;

		if( !keyFrames_.empty() ) {

#if SOFT_REGISTRATION

			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
			MultiResolutionSoftSurfelRegistration reg;
			reg.estimatePoseCovarianceLM( poseCov, *(keyFrames_[referenceKeyFrameId_]->map_), *(keyFrame->map_), lastTransform_, register_start_resolution, register_stop_resolution );

#else

			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
			MultiResolutionSurfelRegistration reg;
			reg.estimatePoseCovariance( poseCov, *(keyFrames_[referenceKeyFrameId_]->map_), *(keyFrame->map_), lastTransform_, register_start_resolution, register_stop_resolution );

#endif


		}
		else
			poseCov.setZero();


#if SOFT_REGISTRATION

		std::cout << "warning loglikelihood check not implemented for soft reg" << std::endl;

#else

		MultiResolutionSurfelRegistration reg;
		double logLikelihood2 = reg.selfMatchLogLikelihood( *(keyFrame->map_) );
		keyFrame->sumLogLikelihood_ += logLikelihood2;
		keyFrame->numEdges_ += 1.0;

#endif

		// extend slam graph with vertex for new key frame and with one edge towards the last keyframe..
		unsigned int keyFrameId = addKeyFrame( referenceKeyFrameId_, keyFrame, lastTransform_ );
		if( optimizer_->vertices().size() > 1 ) {
			if( !addEdge( keyFrames_[referenceKeyFrameId_]->nodeId_, keyFrames_[keyFrameId]->nodeId_, lastTransform_, poseCov ) ) {
				std::cout << "WARNING: new key frame not connected to graph!\n";
				assert(false);
			}
		}


		assert( optimizer_->vertices().size() == keyFrames_.size() );

	}
	else if( params_.optimize_full_trajectory_ ) {

		boost::shared_ptr< IntermediateFrame > intermediateFrame = boost::shared_ptr< IntermediateFrame >( new IntermediateFrame() );

		Eigen::Matrix< double, 6, 6 > poseCov;

		if( !keyFrames_.empty() ) {

#if SOFT_REGISTRATION

			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
			MultiResolutionSoftSurfelRegistration reg;
			reg.estimatePoseCovarianceLM( poseCov, *(keyFrames_[referenceKeyFrameId_]->map_), *(target), lastTransform_, register_start_resolution, register_stop_resolution );

#else

			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
			pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
			MultiResolutionSurfelRegistration reg;
			reg.estimatePoseCovariance( poseCov, *(keyFrames_[referenceKeyFrameId_]->map_), *(target), lastTransform_, register_start_resolution, register_stop_resolution );

#endif

		}
		else
			poseCov.setZero();

		unsigned int intermediateFrameId = addIntermediateFrame( referenceKeyFrameId_, intermediateFrame, lastTransform_ );
		if( optimizer_->vertices().size() > 1 ) {
			if( !addEdge( keyFrames_[referenceKeyFrameId_]->nodeId_, intermediateFrame->nodeId_, lastTransform_, poseCov ) ) {
				std::cout << "WARNING: new intermediate frame not connected to graph!\n";
				assert(false);
			}
		}


		// register current frame to last frame - and add edge
		Eigen::Matrix4d seqTransform = Eigen::Matrix4d::Identity();

#if SOFT_REGISTRATION

		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
		MultiResolutionSoftSurfelRegistration reg;

		bool retVal2 = reg.estimateTransformation( *lastFrameMap_, *target, seqTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS );
		if( REGISTER_TWICE )
			retVal2 = reg.estimateTransformation( *lastFrameMap_, *target, seqTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS );

		if( !retVal2 ) {
			std::cout << "SLAM: could not connect sequential frames\n";
		}
		else {

			reg.estimatePoseCovarianceLM( poseCov, *lastFrameMap_, *target, seqTransform, register_start_resolution, register_stop_resolution );

			unsigned int v1_id = intermediateFrame->nodeId_-1;
			unsigned int v2_id = intermediateFrame->nodeId_;

			addEdge( v1_id, v2_id, seqTransform, poseCov );

		}

#else

		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
		pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
		MultiResolutionSurfelRegistration reg;

		bool retVal2 = reg.estimateTransformation( *lastFrameMap_, *target, seqTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
		if( REGISTER_TWICE )
			retVal2 = reg.estimateTransformation( *lastFrameMap_, *target, seqTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );

		if( !retVal2 ) {
			std::cout << "SLAM: could not connect sequential frames\n";
		}
		else {

			reg.estimatePoseCovariance( poseCov, *lastFrameMap_, *target, seqTransform, register_start_resolution, register_stop_resolution );

			unsigned int v1_id = intermediateFrame->nodeId_-1;
			unsigned int v2_id = intermediateFrame->nodeId_;

			addEdge( v1_id, v2_id, seqTransform, poseCov );

		}

#endif

	}

	generate_keyview_time_ = sw.getTime();

	std::cout << "generating key frame took " << sw.getTime() << "\n";
	sw.reset();

	if( params_.optimize_ ) { //&& !generateKeyFrame ) {
		// try to match between older key frames (that are close in optimized pose)
		if( !connectClosePoses( startResolution, stopResolution, params_.connectRandom_ ) )
			;
//			refineWorstEdgeRandom( startResolution, stopResolution );
	}

	connect_time_ = sw.getTime();

	std::cout << "connect close poses took " << sw.getTime() << "\n";

	sw.reset();

	if( optimizer_->vertices().size() >= 3 ) {

//		static std::ofstream outfile("g2o_timing.txt");

		// optimize slam graph
		std::cout << "optimizing...\n";
		optimizer_->initializeOptimization();
		optimizer_->optimize(5);
		optimizer_->computeActiveErrors();
		std::cout << optimizer_->vertices().size() << " nodes, "
				<< optimizer_->edges().size() << " edges, "
				<< "chi2: " << optimizer_->chi2() << "\n";


//		outfile << std::fixed << std::setprecision(10) << deltat << "\n";

	}

	optimize_time_ = sw.getTime();
	sw.reset();




	// get estimated transform in map frame
	unsigned int oldReferenceId_ = referenceKeyFrameId_;
	g2o::VertexSE3* v_ref_old = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[oldReferenceId_]->nodeId_ ) );
	Eigen::Matrix4d pose_ref_old = v_ref_old->estimate().matrix();
	Eigen::Matrix4d tracked_pose = pose_ref_old * lastTransform_;

	unsigned int bestId = optimizer_->vertices().size()-1;


	// select closest key frame to current camera pose for further tracking
	// in this way, we do not create unnecessary key frames..
	float bestAngle = std::numeric_limits<float>::max();
	float bestDist = std::numeric_limits<float>::max();
	for( unsigned int kf_id = 0; kf_id < keyFrames_.size(); kf_id++ ) {

		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[kf_id]->nodeId_ ) );

		Eigen::Matrix4d v_pose = v->estimate().matrix();

		Eigen::Matrix4d diffTransform = v_pose.inverse() * tracked_pose;

		double angle = Eigen::AngleAxisd( diffTransform.block<3,3>(0,0) ).angle();
		double dist = diffTransform.block<3,1>(0,3).norm();

		if( poseIsClose(diffTransform) && fabsf( angle ) < bestAngle && dist < bestDist ) {
			bestAngle = angle;
			bestDist = dist;
			bestId = kf_id;
		}

	}

	// try to add new edge between the two reference
	// if not possible, we keep the old reference frame such that a new key frame will added later that connects the two reference frames
	bool switchReferenceID = true;
	g2o::VertexSE3* v_ref = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[referenceKeyFrameId_]->nodeId_ ) );


	if( switchReferenceID ) {
		referenceKeyFrameId_ = bestId;
	}

	// set lastTransform_ to pose wrt reference key frame
	v_ref = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[referenceKeyFrameId_]->nodeId_ ) );
	Eigen::Matrix4d pose_ref = v_ref->estimate().matrix();
	lastTransform_ = pose_ref.inverse() * tracked_pose;



	// swap allocators and buffer current frame
	if( params_.optimize_full_trajectory_ ) {

		currentAllocIdx_ = (currentAllocIdx_+1) % 2;

		lastFrameMap_ = target;

	}



	return true;

}

bool SLAM::connectClosePoses( float register_start_resolution, float register_stop_resolution, bool random ) {

//	return false;

	// random == true: randomly check only one vertex, the closer, the more probable the check
	if( random ) {

		const double sigma2_dist = 0.7*0.7;
		const double sigma2_angle = 0.5*0.5;

//		for( unsigned int kf1_id = 0; kf1_id < keyFrames_.size(); kf1_id++ ) {
		for( unsigned int kf1_id = referenceKeyFrameId_; kf1_id <= referenceKeyFrameId_; kf1_id++ ) {

			unsigned int v1_id = keyFrames_[kf1_id]->nodeId_;
			g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );

			std::vector< int > vertices, keyframes;
			std::vector< double > probs;
			double sumProbs = 0.0;

			double bestProb = 0.0;
			unsigned int bestIdx = 0;

			for( unsigned int kf2_id = 0; kf2_id < kf1_id; kf2_id++ ) {

				if( keyFrames_[kf1_id]->checkedKeyFrames_.find( kf2_id ) != keyFrames_[kf1_id]->checkedKeyFrames_.end() )
					continue;

				unsigned int v2_id = keyFrames_[kf2_id]->nodeId_;
				g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

				// check if edge already exists between the vertices
				bool foundEdge = false;
				for( EdgeSet::iterator it = v1->edges().begin(); it != v1->edges().end(); ++it ) {
					g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( *it );
					if( ( edge->vertices()[0]->id() == v1_id && edge->vertices()[1]->id() == v2_id ) || ( edge->vertices()[0]->id() == v2_id && edge->vertices()[1]->id() == v1_id ) ) {
						foundEdge = true;
						break;
					}
				}
				if( foundEdge )
					continue;


				// diff transform from v2 to v1
				Eigen::Matrix4d diffTransform = (v1->estimate().inverse() * v2->estimate()).matrix();

				if( poseIsFar( diffTransform ) )
					continue;

				double angle = Eigen::AngleAxisd( diffTransform.block<3,3>(0,0) ).angle();
				double dist = diffTransform.block<3,1>(0,3).norm();

				// probability of drawing v2 to check for an edge
				double probDist = exp( -0.5 * dist*dist / sigma2_dist );
				double probAngle = exp( -0.5 * angle*angle / sigma2_angle );

				if( probDist > 0.1 && probAngle > 0.1 ) {

					sumProbs += probDist*probAngle;
					probs.push_back( sumProbs );
					vertices.push_back( v2_id );
					keyframes.push_back( kf2_id );

					if( probDist*probAngle > bestProb ) {
						bestProb = probDist*probAngle;
						bestIdx = vertices.size()-1;
					}

				}

			}

			if( probs.size() == 0 )
				continue;

			unsigned int i = bestIdx;

			// draw random number in [0,sumProbs]
//			double checkProb = (double)rand() / (double)(RAND_MAX + 1.0) * sumProbs;
//			for( int i = 0; i < vertices.size(); i++ ) {
//				if( checkProb <= probs[i] ) {
					int v2_id = vertices[i];
					g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );
					Eigen::Matrix4d diffTransform = (v1->estimate().inverse() * v2->estimate()).matrix();

					bool retVal = addEdge( v1_id, v2_id, diffTransform, register_start_resolution, register_stop_resolution );
					if( retVal ) {
						keyFrames_[kf1_id]->checkedKeyFrames_.clear();
						keyFrames_[keyframes[i]]->checkedKeyFrames_.clear();
					}
					else
						keyFrames_[kf1_id]->checkedKeyFrames_.insert( keyframes[i] );
					return true;
//				}
//			}

		}

	}
	else {

		// add all new edges to slam graph
		for( unsigned int kf1_id = 0; kf1_id < keyFrames_.size(); kf1_id++ ) {

			for( unsigned int kf2_id = 0; kf2_id < kf1_id; kf2_id++ ) {

				unsigned int v1_id = keyFrames_[kf1_id]->nodeId_;
				unsigned int v2_id = keyFrames_[kf2_id]->nodeId_;
				g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
				g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

				// check if edge already exists between the vertices
				bool foundEdge = false;
				for( EdgeSet::iterator it = v1->edges().begin(); it != v1->edges().end(); ++it ) {
					g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( *it );
					if( ( edge->vertices()[0]->id() == v1_id && edge->vertices()[1]->id() == v2_id ) || ( edge->vertices()[0]->id() == v2_id && edge->vertices()[1]->id() == v1_id ) ) {
						foundEdge = true;
						break;
					}
				}
				if( foundEdge )
					continue;



				// check if poses close
				// diff transform from v2 to v1
				Eigen::Matrix4d diffTransform = (v1->estimate().inverse() * v2->estimate()).matrix();
				if( poseIsFar( diffTransform ) )
					continue;

				bool retVal = addEdge( v1_id, v2_id, diffTransform, register_start_resolution, register_stop_resolution );

			}

		}

		return true;

	}

	return false;

}


bool SLAM::refineEdge( g2o::EdgeSE3* edge, float register_start_resolution, float register_stop_resolution ) {

	unsigned int v1_id = edge->vertices()[0]->id();
	unsigned int v2_id = edge->vertices()[1]->id();

	g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
	g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

	Eigen::Matrix4d diffTransform = (v1->estimate().inverse() * v2->estimate()).matrix();

	// register maps with pose guess from graph
	Eigen::Matrix< double, 6, 6 > poseCov;

	if( keyFrameNodeMap_.find( v1_id ) == keyFrameNodeMap_.end() || keyFrameNodeMap_.find( v2_id ) == keyFrameNodeMap_.end() ) {
		return true; // dont delete this edge!
	}


#if SOFT_REGISTRATION

	pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
	MultiResolutionSoftSurfelRegistration reg;
	bool retVal = reg.estimateTransformation( *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), diffTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS );
	if( REGISTER_TWICE )
		retVal = reg.estimateTransformation( *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), diffTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS );
	if( !retVal )
		return false;

	retVal &= reg.estimatePoseCovarianceLM( poseCov, *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), diffTransform, register_start_resolution, register_stop_resolution );


#else

	pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrSrc;
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr corrTgt;
	MultiResolutionSurfelRegistration reg;
	reg.params_.registerFeatures_ = params_.usePointFeatures_;
	reg.params_.debugFeatures_ = params_.debugPointFeatures_;
	bool retVal = reg.estimateTransformation( *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), diffTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
	if( REGISTER_TWICE )
		retVal = reg.estimateTransformation( *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), diffTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
	if( !retVal )
		return false;

	retVal &= reg.estimatePoseCovariance( poseCov, *(keyFrameNodeMap_[v1_id]->map_), *(keyFrameNodeMap_[v2_id]->map_), diffTransform, register_start_resolution, register_stop_resolution );

#endif


	if( retVal ) {

		g2o::SE3Quat measurement_mean( Eigen::Quaterniond( diffTransform.block<3,3>(0,0) ), diffTransform.block<3,1>(0,3) );
		Eigen::Matrix< double, 6, 6 > measurement_information = poseCov.inverse();

		edge->setMeasurement( measurement_mean );
		edge->setInformation( measurement_information );

	}

	return retVal;

}


void SLAM::refine( unsigned int refineIterations, unsigned int optimizeIterations, float register_start_resolution, float register_stop_resolution ) {


	if( optimizer_->vertices().size() >= 3 ) {

		for( unsigned int i = 0; i < refineIterations; i++ ) {

			std::cout << "refining " << i << " / " << refineIterations << "\n";

			// reestimate all edges in the graph from the current pose estimates in the graph
			std::vector< g2o::EdgeSE3* > removeEdges;
			for( EdgeSet::iterator it = optimizer_->edges().begin(); it != optimizer_->edges().end(); ++it ) {

				g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( *it );

				bool retVal = refineEdge( edge, register_start_resolution, register_stop_resolution );

				if( !retVal ) {

					removeEdges.push_back( edge );

				}

			}

			for( unsigned int j = 0; j < removeEdges.size(); j++ )
				optimizer_->removeEdge( removeEdges[j] );


			// reoptimize for 10 iterations
			optimizer_->initializeOptimization();
			optimizer_->optimize( 10 );

		}


		// optimize slam graph
		std::cout << "optimizing...\n";
		optimizer_->initializeOptimization();
		optimizer_->optimize( optimizeIterations );
		optimizer_->computeActiveErrors();
		std::cout << optimizer_->vertices().size() << " nodes, "
				<< optimizer_->edges().size() << " edges, "
				<< "chi2: " << optimizer_->chi2() << "\n";
	}

}


void SLAM::refineInConvexHull( unsigned int refineIterations, unsigned int optimizeIterations, float register_start_resolution, float register_stop_resolution, float minResolution, const Eigen::Matrix4d& referenceTransform, float minHeight, float maxHeight, std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > convexHull ) {

	const float min_resolution = minResolution;
	const float max_radius = 30.f;

	// restrict all key frames to convex hull from current pose estimate
	pcl::ExtractPolygonalPrismData< pcl::PointXYZRGB > hull_limiter;

	// extract map and stitched point cloud from selected volume..
	// find convex hull for selected points in reference frame
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_selected_points( new pcl::PointCloud< pcl::PointXYZRGB >() );
	for( unsigned int j = 0; j < convexHull.size(); j++ ) {
		pcl::PointXYZRGB p;
		p.x = convexHull[j](0);
		p.y = convexHull[j](1);
		p.z = convexHull[j](2);
		cloud_selected_points->points.push_back( p );
	}

	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_convex_hull( new pcl::PointCloud< pcl::PointXYZRGB >() );
	pcl::ConvexHull< pcl::PointXYZRGB > chull;
	chull.setInputCloud( cloud_selected_points );
	chull.reconstruct( *cloud_convex_hull );


	for( unsigned int v_id = 0; v_id < optimizer_->vertices().size(); v_id++ ) {

		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v_id ) );

		Eigen::Matrix4d v_pose = v->estimate().matrix();

		Eigen::Matrix4d transform = referenceTransform * v_pose;


		if( keyFrameNodeMap_.find( v_id ) == keyFrameNodeMap_.end() )
			continue;

		boost::shared_ptr< KeyFrame > kf = keyFrameNodeMap_[ v_id ];


		pcl::PointCloud< pcl::PointXYZRGB >::Ptr transformedCloud = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );
		pcl::transformPointCloud( *(kf->cloud_), *transformedCloud, transform.cast<float>() );

		transformedCloud->sensor_origin_ = transform.block<4,1>(0,3).cast<float>();
		transformedCloud->sensor_orientation_ = Eigen::Quaternionf( transform.block<3,3>(0,0).cast<float>() );


		// get indices in convex hull
		pcl::PointIndices::Ptr object_indices( new pcl::PointIndices() );
		hull_limiter.setInputCloud( transformedCloud );
		hull_limiter.setInputPlanarHull( cloud_convex_hull );
		hull_limiter.setHeightLimits( minHeight, maxHeight );
		hull_limiter.setViewPoint( transformedCloud->sensor_origin_[0], transformedCloud->sensor_origin_[1], transformedCloud->sensor_origin_[2] );
		hull_limiter.segment( *object_indices );


		pcl::PointCloud< pcl::PointXYZRGB >::Ptr insideCloud = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );
		*insideCloud = *(keyFrames_[v_id]->cloud_);

		// mark points outside of convex hull nan
		std::vector<int> markNAN( insideCloud->points.size(), 1 );
		for( unsigned int i = 0; i < object_indices->indices.size(); i++ ) {

			markNAN[ object_indices->indices[i] ] = 0;

		}

		for( unsigned int i = 0; i < markNAN.size(); i++ ) {

			if( markNAN[i] ) {

				insideCloud->points[ i ].x =
						insideCloud->points[ i ].y =
								insideCloud->points[ i ].z = std::numeric_limits<float>::quiet_NaN();

			}

		}


		// generate new map for key frame
		kf->cloud_ = insideCloud;
		kf->map_ = boost::shared_ptr< MultiResolutionSurfelMap >( new MultiResolutionSurfelMap( min_resolution, max_radius ) );
		kf->map_->imageAllocator_ = imageAllocator_[currentAllocIdx_];
		kf->map_->params_.dist_dependency = params_.map_dist_dependency_;
		kf->map_->addImage( *insideCloud );
		std::vector< int > imageBorderIndices;
		kf->map_->findVirtualBorderPoints( *insideCloud, imageBorderIndices );
		kf->map_->markNoUpdateAtPoints( *insideCloud, imageBorderIndices );
		kf->map_->evaluateSurfels();
		kf->map_->octree_->root_->establishNeighbors();
		kf->map_->buildShapeTextureFeatures();



	}


	refine( refineIterations, optimizeIterations, register_start_resolution, register_stop_resolution );


}


bool edgeCompareChi( const g2o::HyperGraph::Edge* a, const g2o::HyperGraph::Edge* b ) { return  (dynamic_cast< const g2o::EdgeSE3* >( a ))->chi2() > (dynamic_cast< const g2o::EdgeSE3* >( b ))->chi2(); }


void SLAM::refineWorstEdges( float fraction, float register_start_resolution, float register_stop_resolution ) {

	if( optimizer_->vertices().size() >= 3 ) {

		int numRefineEdges = optimizer_->edges().size();
		int refineEdgeIdx = rand() % numRefineEdges;

		// reestimate fraction of edges with worst chi2
		std::vector< g2o::HyperGraph::Edge* > sortedEdges;
		sortedEdges.assign( optimizer_->edges().begin(), optimizer_->edges().end() );
		std::sort( sortedEdges.begin(), sortedEdges.end(), edgeCompareChi );

		std::cout << dynamic_cast< g2o::EdgeSE3* >(*sortedEdges.begin())->chi2() << " " << dynamic_cast< g2o::EdgeSE3* >(*(sortedEdges.end()-1))->chi2() << "\n";


		std::vector< g2o::EdgeSE3* > removeEdges;

		g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( sortedEdges[refineEdgeIdx] );

		bool retVal = refineEdge( edge, register_start_resolution, register_stop_resolution );


		for( unsigned int j = 0; j < removeEdges.size(); j++ )
			optimizer_->removeEdge( removeEdges[j] );

	}

}

void SLAM::refineWorstEdgeRandom( float register_start_resolution, float register_stop_resolution ) {

	if( optimizer_->vertices().size() >= 3 ) {

		// reestimate fraction of edges with worst chi2
		std::vector< g2o::HyperGraph::Edge* > sortedEdges;
		sortedEdges.assign( optimizer_->edges().begin(), optimizer_->edges().end() );
		std::sort( sortedEdges.begin(), sortedEdges.end(), edgeCompareChi );

		double sumChi2 = 0.0;
		for( unsigned int i = 0; i < sortedEdges.size(); i++ ) {
			sumChi2 += dynamic_cast< g2o::EdgeSE3* >( sortedEdges[i] )->chi2();
		}
		double checkProb = (double)rand() / (double)(RAND_MAX + 1.0) * sumChi2;

		size_t refineEdgeIdx = 0;
		sumChi2 = 0.0;
		for( unsigned int i = 0; i < sortedEdges.size(); i++ ) {
			sumChi2 += dynamic_cast< g2o::EdgeSE3* >( sortedEdges[i] )->chi2();

			if( sumChi2 > checkProb ) {
				refineEdgeIdx = i;
				break;
			}
		}

		g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( sortedEdges[refineEdgeIdx] );

		std::cout << "refining " << refineEdgeIdx << "\n";

		bool retVal = refineEdge( edge, register_start_resolution, register_stop_resolution );


	}

}


void SLAM::dumpError() {

	// dump error of all edges in the slam graph
	std::ofstream outfile( "slam_graph_error.dat" );

	for( EdgeSet::iterator it = optimizer_->edges().begin(); it != optimizer_->edges().end(); ++it ) {
		g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( *it );

		outfile << edge->chi2() << "\n";

	}
}



boost::shared_ptr< MultiResolutionSurfelMap > SLAM::getMap( const Eigen::Matrix4d& referenceTransform, float minResolution ) {


	const float min_resolution = minResolution;
	const float max_radius = 30.f;


	boost::shared_ptr< MultiResolutionSurfelMap > graphmap = boost::shared_ptr< MultiResolutionSurfelMap >( new MultiResolutionSurfelMap( min_resolution, max_radius ) );
	graphmap->imageAllocator_ = imageAllocator_[currentAllocIdx_];

	for( unsigned int v_id = 0; v_id < optimizer_->vertices().size(); v_id++ ) {

		if( keyFrameNodeMap_.find(v_id) == keyFrameNodeMap_.end() )
			continue;

		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v_id ) );

		if( v->edges().size() == 0 )
			continue;

		Eigen::Matrix4d v_pose = v->estimate().matrix();

		Eigen::Matrix4d transform = referenceTransform * v_pose;

		pcl::PointCloud< pcl::PointXYZRGB > transformedCloud;
		pcl::transformPointCloud( *(keyFrameNodeMap_[v_id]->cloud_), transformedCloud, transform.cast<float>() );

		transformedCloud.sensor_origin_ = transform.block<4,1>(0,3).cast<float>();
		transformedCloud.sensor_orientation_ = Eigen::Quaternionf( transform.block<3,3>(0,0).cast<float>() );


		// add keyframe to map
		graphmap->setApplyUpdate(false);
		std::vector< int > imageFGBorderIndices, imageBGBorderIndices;
		graphmap->findVirtualBorderPoints( *(keyFrameNodeMap_[v_id]->cloud_), imageFGBorderIndices );
		graphmap->markNoUpdateAtPoints( transformedCloud, imageFGBorderIndices );
		graphmap->findForegroundBorderPoints( *(keyFrameNodeMap_[v_id]->cloud_), imageBGBorderIndices );
		graphmap->markNoUpdateAtPoints( transformedCloud, imageBGBorderIndices );
		graphmap->params_.dist_dependency = params_.map_dist_dependency_;
		graphmap->addImage( transformedCloud );
		graphmap->clearUpdateSurfelsAtPoints( transformedCloud, imageFGBorderIndices );
		graphmap->clearUpdateSurfelsAtPoints( transformedCloud, imageBGBorderIndices );
		graphmap->setUpToDate( true );


	}

	graphmap->setApplyUpdate(true);
	graphmap->setUpToDate( false );
	graphmap->octree_->root_->establishNeighbors();
	graphmap->evaluateSurfels();
	graphmap->buildShapeTextureFeatures();

	return graphmap;

}


boost::shared_ptr< pcl::PointCloud< pcl::PointXYZRGB > > SLAM::getMapCloud( const Eigen::Matrix4d& referenceTransform ) {

	boost::shared_ptr< pcl::PointCloud< pcl::PointXYZRGB > > mapCloud = boost::shared_ptr< pcl::PointCloud< pcl::PointXYZRGB > >( new pcl::PointCloud< pcl::PointXYZRGB >() );

	for( unsigned int v_id = 0; v_id < optimizer_->vertices().size(); v_id++ ) {

		if( keyFrameNodeMap_.find(v_id) == keyFrameNodeMap_.end() )
			continue;

		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v_id ) );

		if( v->edges().size() == 0 )
			continue;

		Eigen::Matrix4d v_pose = v->estimate().matrix();

		Eigen::Matrix4d transform = referenceTransform * v_pose;

		pcl::PointCloud< pcl::PointXYZRGB > transformedCloud;
		pcl::transformPointCloud( *(keyFrameNodeMap_[v_id]->cloud_), transformedCloud, transform.cast<float>() );

		transformedCloud.sensor_origin_ = transform.block<4,1>(0,3).cast<float>();
		transformedCloud.sensor_orientation_ = Eigen::Quaternionf( transform.block<3,3>(0,0).cast<float>() );

		for( unsigned int i = 0; i < transformedCloud.points.size(); i++ ) {
			if( !std::isnan( transformedCloud.points[i].x ) )
				mapCloud->points.push_back( transformedCloud.points[i] );
		}

	}

	mapCloud->width = mapCloud->points.size();
	mapCloud->height = 1;

	return mapCloud;

}

//
//boost::shared_ptr< pcl::PolygonMesh > SLAM::getMapMesh( unsigned int res, unsigned int repetitions ) {
//
//	boost::shared_ptr< pcl::PolygonMesh > mesh = boost::shared_ptr< pcl::PolygonMesh >( new pcl::PolygonMesh() );
//
//	TSDFVolumeOctree::Ptr tsdf (new TSDFVolumeOctree);
//    tsdf->setIntegrateColor(true);
//    tsdf->setColorMode("RGB");
//    tsdf->setCameraIntrinsics( 525.0, 525.0, 319.5, 239.5 );
////    tsdf->setDepthTruncationLimits( 0.1, -0.1 );
//	tsdf->setGridSize (3., 3., 3.); // 10m x 10m x 10m
//	tsdf->setResolution (res, res, res); // Smallest cell size = 10m / 2048 = about half a centimeter
////	Eigen::Affine3d tsdf_center; // Optionally offset the center
////	tsdf->setGlobalTransform (tsdf_center);
////	tsdf->setIntegrateColor( true );
//	tsdf->reset (); // Initialize it to be empty
//
//
//	for( unsigned int v_id = 0; v_id < optimizer_->vertices().size(); v_id++ ) {
//
//		if( keyFrameNodeMap_.find(v_id) == keyFrameNodeMap_.end() )
//			continue;
//
//		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v_id ) );
//
//		if( v->edges().size() == 0 )
//			continue;
//
//		Eigen::Matrix4d v_pose = v->estimate().matrix();
//
//		Eigen::Matrix4d transform = v_pose;
//
////		pcl::PointCloud< pcl::PointXYZRGB > transformedCloud;
////		pcl::transformPointCloud( *(keyFrameNodeMap_[v_id]->cloud_), transformedCloud, transform.cast<float>() );
////
////		transformedCloud.sensor_origin_ = transform.block<4,1>(0,3).cast<float>();
////		transformedCloud.sensor_orientation_ = Eigen::Quaternionf( transform.block<3,3>(0,0).cast<float>() );
//
//		pcl::PointCloud< pcl::PointXYZRGB > imageCloud;
//		imageCloud.width = 640;
//		imageCloud.height = 480;
//		imageCloud.sensor_orientation_.setIdentity();
//		imageCloud.sensor_origin_ = Eigen::Vector4f::Zero();
//		imageCloud.sensor_origin_(3,0) = 1.0;
//
//		imageCloud.is_dense = true;
//
//		imageCloud.points.resize(640*480);
//		for( unsigned int i = 0; i < imageCloud.points.size(); i++ ) {
//
//			imageCloud.points[i].x = std::numeric_limits<float>::quiet_NaN();
//			imageCloud.points[i].y = std::numeric_limits<float>::quiet_NaN();
//			imageCloud.points[i].z = std::numeric_limits<float>::quiet_NaN();
//
//		}
//
//		for( unsigned int i = 0; i < keyFrameNodeMap_[v_id]->cloud_->points.size(); i++ ) {
//
//			const pcl::PointXYZRGB& p = keyFrameNodeMap_[v_id]->cloud_->points[i];
//			int x = std::max( 0, std::min( 639, (int)(p.x / p.z * 525.f + 319.5f) ) );
//			int y = std::max( 0, std::min( 479, (int)(p.y / p.z * 525.f + 239.5f) ) );
//
//			imageCloud.points[ y*640+x ] = p;
//
//		}
//
//
//		Eigen::Affine3d pose( transform );
//
//		pcl::PointCloud< pcl::Normal > normals;
//
//
//		std::cout << "TSDF integrate " << v_id << "\n";
////		tsdf->integrateCloud( *(keyFrameNodeMap_[v_id]->cloud_), normals, pose ); // Integrate the cloud
//
//		for( unsigned int j = 0; j < repetitions; j++ )
//			tsdf->integrateCloud( imageCloud, normals, pose ); // Integrate the cloud
//
//	}
//
//
//
//	// Now what do you want to do with it?
////	float distance; pcl::PointXYZ query_point (1.0, 2.0, -1.0);
////	tsdf->getFxn (query_point, distance); // distance is normalized by the truncation limit -- goes from -1 to 1
////	pcl::PointCloud<pcl::PointNormal>::Ptr raytraced = tsdf->renderView (pose_to_render_from); // Optionally can render it
////	tsdf->save ("output.vol"); // Save it?
//
//	// Mesh with marching cubes
//	MarchingCubesTSDFOctree mc;
//    mc.setColorByRGB(true);
//    mc.setInputTSDF (tsdf);
//	mc.reconstruct (*mesh);
//
//	return mesh;
//
//}


boost::shared_ptr< MultiResolutionSurfelMap > SLAM::getMapInConvexHull( const Eigen::Matrix4d& referenceTransform, float minResolution, float minHeight, float maxHeight, std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > convexHull ) {


	const float min_resolution = minResolution;
	const float max_radius = 30.f;


	// extract map and stitched point cloud from selected volume..
	// find convex hull for selected points in reference frame
	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_selected_points( new pcl::PointCloud< pcl::PointXYZRGB >() );
	for( unsigned int j = 0; j < convexHull.size(); j++ ) {
		pcl::PointXYZRGB p;
		p.x = convexHull[j](0);
		p.y = convexHull[j](1);
		p.z = convexHull[j](2);
		cloud_selected_points->points.push_back( p );

		std::cout << p.x << " " << p.y << " " << p.z << "\n";
	}

	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud_convex_hull( new pcl::PointCloud< pcl::PointXYZRGB >() );
	pcl::ConvexHull< pcl::PointXYZRGB > chull;
	chull.setInputCloud( cloud_selected_points );
	chull.reconstruct( *cloud_convex_hull );

	std::cout << "convex hull:\n";
	for( unsigned int j = 0; j < cloud_convex_hull->points.size(); j++ ) {
		const pcl::PointXYZRGB& p = cloud_convex_hull->points[j];
		std::cout << p.x << " " << p.y << " " << p.z << "\n";
	}

	cloud_convex_hull->points.push_back( cloud_convex_hull->points[0] );

	boost::shared_ptr< MultiResolutionSurfelMap > graphmap = boost::shared_ptr< MultiResolutionSurfelMap >( new MultiResolutionSurfelMap( min_resolution, max_radius ) );
	graphmap->imageAllocator_ = imageAllocator_[currentAllocIdx_];

	for( unsigned int v_id = 0; v_id < optimizer_->vertices().size(); v_id++ ) {

		if( keyFrameNodeMap_.find(v_id) == keyFrameNodeMap_.end() )
			continue;

		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v_id ) );

		if( v->edges().size() == 0 && optimizer_->vertices().size() > 1 )
			continue;

		Eigen::Matrix4d v_pose = v->estimate().matrix();

		Eigen::Matrix4d transform = referenceTransform * v_pose;

		pcl::PointCloud< pcl::PointXYZRGB >::Ptr transformedCloud = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );
		pcl::transformPointCloud( *(keyFrameNodeMap_[v_id]->cloud_), *transformedCloud, transform.cast<float>() );

		transformedCloud->sensor_origin_ = transform.block<4,1>(0,3).cast<float>();
		transformedCloud->sensor_orientation_ = Eigen::Quaternionf( transform.block<3,3>(0,0).cast<float>() );

		// get indices in convex hull
		pcl::PointIndices::Ptr object_indices( new pcl::PointIndices() );
		pcl::ExtractPolygonalPrismData< pcl::PointXYZRGB > hull_limiter;
		hull_limiter.setInputCloud( transformedCloud );
		hull_limiter.setInputPlanarHull( cloud_convex_hull );
		hull_limiter.setHeightLimits( minHeight, maxHeight );
		hull_limiter.setViewPoint( transformedCloud->sensor_origin_[0], transformedCloud->sensor_origin_[1], transformedCloud->sensor_origin_[2] );
		hull_limiter.segment( *object_indices );

		std::cout << object_indices->indices.size() << "\n";


////		// add keyframe to map
////		graphmap->setApplyUpdate( false );
////		graphmap->markUpdateImprovedEffViewDistSurfels( transformedCloud->sensor_origin_.block<3,1>(0,0) );
//		std::vector< int > imageBorderIndices;
//		graphmap->findVirtualBorderPoints( *(keyFrameNodeMap_[v_id]->cloud_), imageBorderIndices );
//		graphmap->markNoUpdateAtPoints( *transformedCloud, imageBorderIndices );
////		graphmap->unevaluateSurfels();
//
//		std::vector< int > contourIndices;
//		graphmap->findForegroundBorderPoints( *(keyFrameNodeMap_[v_id]->cloud_), contourIndices );
//
//		std::cout << "contour points: " << contourIndices.size() << "\n";
//
//		pcl::PointIndices::Ptr object_contour_indices( new pcl::PointIndices() );
//		for( unsigned int i = 0; i < object_indices->indices.size(); i++ ) {
//			if( std::find( contourIndices.begin(), contourIndices.end(), object_indices->indices[i] ) != contourIndices.end() ) {
//				object_contour_indices->indices.push_back( object_indices->indices[i] );
//			}
//		}
//
//		std::cout << "adding " << object_contour_indices->indices.size() << "\n";
//
//		graphmap->addPoints( *transformedCloud, object_contour_indices->indices );
////		graphmap->clearUpdateSurfelsAtPoints( *transformedCloud, imageBorderIndices ); // only new surfels at these points have up_to_date == false !
//		graphmap->octree_->root_->establishNeighbors();
////		graphmap->clearUnstableSurfels();
////		graphmap->setApplyUpdate( true );
////		graphmap->buildShapeTextureFeatures();


//         // add keyframe to map
//         std::vector< int > imageBorderIndices;
//         graphmap->findVirtualBorderPoints( *( keyFrameNodeMap_[ v_id ]->cloud_ ), imageBorderIndices );
//         graphmap->markNoUpdateAtPoints( *transformedCloud, imageBorderIndices );
//         graphmap->unevaluateSurfels();
//         graphmap->addPoints( *transformedCloud, object_indices->indices );
//         graphmap->octree_->root_->establishNeighbors();
//         graphmap->evaluateSurfels();
//         graphmap->buildShapeTextureFeatures();
		
		
		// add keyframe to map
		graphmap->setApplyUpdate(false);
		std::vector< int > imageFGBorderIndices, imageBGBorderIndices;
		graphmap->findVirtualBorderPoints( *(keyFrameNodeMap_[v_id]->cloud_), imageFGBorderIndices );
		graphmap->markNoUpdateAtPoints( *transformedCloud, imageFGBorderIndices );
		graphmap->findForegroundBorderPoints( *(keyFrameNodeMap_[v_id]->cloud_), imageBGBorderIndices );
		graphmap->markNoUpdateAtPoints( *transformedCloud, imageBGBorderIndices );
		graphmap->params_.dist_dependency = params_.map_dist_dependency_;
		graphmap->addPoints( *transformedCloud, object_indices->indices );
		graphmap->clearUpdateSurfelsAtPoints( *transformedCloud, imageFGBorderIndices );
		graphmap->clearUpdateSurfelsAtPoints( *transformedCloud, imageBGBorderIndices );
		graphmap->setUpToDate( true );


	}

	graphmap->setApplyUpdate(true);
	graphmap->setUpToDate( false );
	graphmap->octree_->root_->establishNeighbors();
	graphmap->evaluateSurfels();
	graphmap->buildShapeTextureFeatures();


	return graphmap;



//	return boost::shared_ptr< MultiResolutionSurfelMap >();

}



