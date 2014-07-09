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

#include "mrsmap/visualization/visualization_slam.h"

#include "pcl/common/common_headers.h"

#include <mrsmap/slam/slam.h>

#include <pcl/common/transforms.h>

#include <mrsmap/utilities/utilities.h>

using namespace mrsmap;

void ViewerSLAM::visualizeSLAMGraph() {

	for( unsigned int i = 0; i < currShapes.size(); i++ ) {

		char str[255];
		sprintf(str,"%i",currShapes[i]);
		viewer->removeShape( str );

	}

	currShapes.clear();


	for( unsigned int i = 0; i < slam_->keyFrames_.size(); i++ ) {

		g2o::VertexSE3* v_curr = dynamic_cast< g2o::VertexSE3* >( slam_->optimizer_->vertex( slam_->keyFrames_[i]->nodeId_ ) );
		char str[255];

		bool isConnectedToRef = displayAll;

		// add coordinate frames for vertices
		Eigen::Matrix4d camTransform = v_curr->estimate().matrix();

		if( i == slam_->referenceKeyFrameId_ || displayAll ) {

			double axislength = 0.2;
			double linewidth = 5;

			if( i == slam_->referenceKeyFrameId_ )
				linewidth = 10, axislength = 0.5;

			pcl::PointXYZRGB p1, p2;

			p1.x = camTransform(0,3);
			p1.y = camTransform(1,3);
			p1.z = camTransform(2,3);
			p2.x = p1.x + axislength*camTransform(0,0);
			p2.y = p1.y + axislength*camTransform(1,0);
			p2.z = p1.z + axislength*camTransform(2,0);
			sprintf(str,"%i",shapeIdx);
			viewer->addLine( p1, p2, 1.0, 0.0, 0.0, std::string( str ) );
			viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, linewidth, str );
			currShapes.push_back( shapeIdx++ );

			p2.x = p1.x + axislength*camTransform(0,1);
			p2.y = p1.y + axislength*camTransform(1,1);
			p2.z = p1.z + axislength*camTransform(2,1);
			sprintf(str,"%i",shapeIdx);
			viewer->addLine( p1, p2, 0.0, 1.0, 0.0, std::string( str ) );
			viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, linewidth, str );
			currShapes.push_back( shapeIdx++ );

			p2.x = p1.x + axislength*camTransform(0,2);
			p2.y = p1.y + axislength*camTransform(1,2);
			p2.z = p1.z + axislength*camTransform(2,2);
			sprintf(str,"%i",shapeIdx);
			viewer->addLine( p1, p2, 0.0, 0.0, 1.0, std::string( str ) );
			viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, linewidth, str );
			currShapes.push_back( shapeIdx++ );

		}

		if( isConnectedToRef && displayMap ) {
			// visualize maps at estimated pose
			pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud2 = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );

#if USE_POINTFEATURE_REGISTRATION
			slam_->keyFrames_[i]->map_->visualizeShapeDistribution( cloud2, selectedDepth, selectedViewDir, false );
#else
//			*cloud2 = *slam_->keyFrames_[i]->cloud_;
//			mrsmap::downsamplePointCloud( slam_->keyFrames_[i]->cloud_, cloud2, 8 );
			slam_->keyFrames_[i]->map_->visualize3DColorDistribution( cloud2, selectedDepth, selectedViewDir, false );
#endif

			pcl::PointCloud< pcl::PointXYZRGB >::Ptr transformedCloud = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );
			pcl::transformPointCloud( *cloud2, *transformedCloud, camTransform.cast<float>() );

			sprintf(str,"map%i",i);
			displayPointCloud( str, transformedCloud, 1 );
		}

	}


	if( displayAll ) {

		double minEdgeStrength = std::numeric_limits<double>::max();
		double maxEdgeStrength = -std::numeric_limits<double>::max();

		for( EdgeSet::iterator it = slam_->optimizer_->edges().begin(); it != slam_->optimizer_->edges().end(); ++it ) {

			g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( *it );

			// add lines for edges
			g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( edge->vertices()[0] );
			g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( edge->vertices()[1] );

			// add coordinate frames for vertices
			Eigen::Matrix4d pose1 = v1->estimate().matrix();
			Eigen::Matrix4d pose2 = v2->estimate().matrix();

			pcl::PointXYZRGB p1, p2;
			char str[255];
			sprintf(str,"%i",shapeIdx);

			p1.x = pose1(0,3);
			p1.y = pose1(1,3);
			p1.z = pose1(2,3);
			p2.x = pose2(0,3);
			p2.y = pose2(1,3);
			p2.z = pose2(2,3);

			viewer->addLine( p1, p2, 0.0, 0.0, 0.0, std::string( str ) );
//			viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, edge->chi2(), str );
			viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, str );
			currShapes.push_back( shapeIdx++ );

			minEdgeStrength = std::min( minEdgeStrength, edge->chi2() );
			maxEdgeStrength = std::max( maxEdgeStrength, edge->chi2() );

		}
//		std::cout << minEdgeStrength << " " << maxEdgeStrength << "\n";
	}

}



