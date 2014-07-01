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

#include "mrsmap/visualization/visualization_map.h"

#include "pcl/common/common_headers.h"

#include "pcl/common/transforms.h"

#include <boost/make_shared.hpp>


using namespace mrsmap;

Viewer::Viewer() {

	viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>( new pcl::visualization::PCLVisualizer( "MRS Viewer" ) );
	viewer->setBackgroundColor( 1, 1, 1 );
//	viewer->setBackgroundColor( 0,0,0 );
	viewer->addCoordinateSystem( 0.1 );
	viewer->initCameraParameters();

	viewer->registerKeyboardCallback( &Viewer::keyboardEventOccurred, *this, NULL );
	viewer->registerPointPickingCallback( &Viewer::pointPickingCallback, *this, NULL );

	index = -1;
	selectedDepth = 10; // d
	selectedViewDir = -1; // v
	processFrame = false; // p
	displayScene = true; // s
	displayMap = true; // m
	displayCorr = false; // c
	displayAll = true; // a
	displayFeatureSimilarity = false; // F
	recordFrame = false; // r
	forceRedraw = false; // f

	is_running = true;

}

Viewer::~Viewer() {
}


void Viewer::spinOnce() {

	if( !viewer->wasStopped() ) {
		viewer->spinOnce(1);
	}
	else {
		is_running = false;
	}

}


void Viewer::pointPickingCallback( const pcl::visualization::PointPickingEvent& event, void* data ) {

	pcl::PointXYZ p;
	event.getPoint( p.x, p.y, p.z );
	std::cout << "picked " << p.x << " " << p.y << " " << p.z << "\n";

	selectedPoint = p;

	viewer->removeShape( "selected_point" );
	viewer->addSphere( p, 0.025, 0, 1.0, 0, "selected_point" );

}

void Viewer::displayPointCloud( const std::string& name, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, int pointSize ) {

	pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud2 = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB >() );
	pcl::copyPointCloud( *cloud, *cloud2 );

	for( unsigned int i = 0; i < cloud2->points.size(); i++ )
		if( isnan( cloud2->points[i].x ) ) {
			cloud2->points[i].x = 0;
			cloud2->points[i].y = 0;
			cloud2->points[i].z = 0;
		}

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb = pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>( cloud2 );

	if( !viewer->updatePointCloud<pcl::PointXYZRGB>( cloud2, rgb, name ) ) {
		viewer->addPointCloud<pcl::PointXYZRGB>( cloud2, rgb, name );
	}
	viewer->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, name );
}




void Viewer::displayPose( const Eigen::Matrix4d& pose ) {

	static int poseidx = 0;

	double axislength = 0.2;

	pcl::PointXYZRGB p1, p2;

	char str[255];

	if( poseidx > 0 ) {
		sprintf( str, "posex%i", poseidx-1 );
		viewer->removeShape( str );
	}
	sprintf( str, "posex%i", poseidx );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,0);
	p2.y = p1.y + axislength*pose(1,0);
	p2.z = p1.z + axislength*pose(2,0);
	viewer->addLine( p1, p2, 1.0, 0.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );


	if( poseidx > 0 ) {
		sprintf( str, "posey%i", poseidx-1 );
		viewer->removeShape( str );
	}
	sprintf( str, "posey%i", poseidx );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,1);
	p2.y = p1.y + axislength*pose(1,1);
	p2.z = p1.z + axislength*pose(2,1);
	viewer->addLine( p1, p2, 0.0, 1.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );



	if( poseidx > 0 ) {
		sprintf( str, "posez%i", poseidx-1 );
		viewer->removeShape( str );
	}
	sprintf( str, "posez%i", poseidx );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,2);
	p2.y = p1.y + axislength*pose(1,2);
	p2.z = p1.z + axislength*pose(2,2);
	viewer->addLine( p1, p2, 0.0, 0.0, 1.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );

	poseidx++;

}


void Viewer::displayPose( const std::string& name, const Eigen::Matrix4d& pose ) {

	double axislength = 0.2;

	pcl::PointXYZRGB p1, p2;

	char str[255];

	sprintf( str, "%sposex", name.c_str() );
	viewer->removeShape( str );
	sprintf( str, "%sposex", name.c_str() );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,0);
	p2.y = p1.y + axislength*pose(1,0);
	p2.z = p1.z + axislength*pose(2,0);
	viewer->addLine( p1, p2, 1.0, 0.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );


	sprintf( str, "%sposey", name.c_str() );
	viewer->removeShape( str );
	sprintf( str, "%sposey", name.c_str() );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,1);
	p2.y = p1.y + axislength*pose(1,1);
	p2.z = p1.z + axislength*pose(2,1);
	viewer->addLine( p1, p2, 0.0, 1.0, 0.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );



	sprintf( str, "%sposez", name.c_str() );
	viewer->removeShape( str );
	sprintf( str, "%sposez", name.c_str() );
	p1.x = pose(0,3);
	p1.y = pose(1,3);
	p1.z = pose(2,3);
	p2.x = p1.x + axislength*pose(0,2);
	p2.y = p1.y + axislength*pose(1,2);
	p2.z = p1.z + axislength*pose(2,2);
	viewer->addLine( p1, p2, 0.0, 0.0, 1.0, str );
	viewer->setShapeRenderingProperties( pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, str );


}


void Viewer::displayCorrespondences( const std::string& name, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud1, const pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud2, const Eigen::Matrix4f& transform ) {

	Eigen::Affine3f transforma( transform );
	pcl::Correspondences corr;
	for( unsigned int i = 0; i < cloud1->points.size(); i++ ) {

		pcl::Correspondence c( i, i, 0 );
		corr.push_back( c );

	}

	viewer->addCorrespondences<pcl::PointXYZRGB>( cloud1, cloud2, corr, name );

}

void Viewer::removeCorrespondences( const std::string& name ) {

	viewer->removeCorrespondences( name );

}


void Viewer::displaySurfaceNormals( const std::string& name, const pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr& cloud ) {

	viewer->setBackgroundColor( 0,0,0 );

	if( !viewer->addPointCloudNormals< pcl::PointXYZRGBNormal >( cloud, 1, 0.1, name+"normals" ) ) {
		viewer->removePointCloud(name+"normals");
		viewer->addPointCloudNormals< pcl::PointXYZRGBNormal >( cloud, 1, 0.1, name+"normals" );
	}

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb = pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal>( cloud );

	if( !viewer->updatePointCloud<pcl::PointXYZRGBNormal>( cloud, rgb, name ) ) {
		viewer->addPointCloud<pcl::PointXYZRGBNormal>( cloud, rgb, name );
	}
	viewer->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name );

}

void Viewer::removeSurfaceNormals( const std::string& name ) {

	char str[255];

	for( unsigned int i = 0; i < currNormals.size(); i++ ) {
		sprintf( str, "normal%i", currNormals[i] );
		viewer->removeShape( str );
	}
	currNormals.clear();

}


void Viewer::keyboardEventOccurred( const pcl::visualization::KeyboardEvent &event, void* data ) {

	if( (event.getKeySym() == "d" || event.getKeySym() == "D") && event.keyDown() ) {

		if( event.getKeySym() == "d" ) {
			selectedDepth++;
		}
		else {
			selectedDepth--;
			if( selectedDepth < 0 )
				selectedDepth = 15;
		}

		selectedDepth = selectedDepth % 16;
		std::cout << "Selected Depth " << selectedDepth << "\n";
	}
	if( (event.getKeySym() == "v" || event.getKeySym() == "V") && event.keyDown() ) {

		if( event.getKeySym() == "v" ) {
			selectedViewDir++;
			if( selectedViewDir == 7 )
				selectedViewDir = -1;
		}
		else {
			selectedViewDir--;
			if( selectedViewDir < -1 )
				selectedViewDir = 6;
		}

		std::cout << "Selected View Dir " << selectedViewDir << "\n";

	}
	if( (event.getKeySym() == "p") && event.keyDown() ) {
		processFrame = true;
	}
	if( (event.getKeySym() == "h") && event.keyDown() ) {
		processFrame = !processFrame;
	}
	if( (event.getKeySym() == "s") && event.keyDown() ) {
		displayScene = !displayScene;
	}
	if( (event.getKeySym() == "m") && event.keyDown() ) {
		displayMap = !displayMap;
	}
	if( (event.getKeySym() == "a") && event.keyDown() ) {
		displayAll = !displayAll;
	}
	if( (event.getKeySym() == "c") && event.keyDown() ) {
		displayCorr = !displayCorr;
	}
	if( (event.getKeySym() == "S") && event.keyDown() ) {
		displayFeatureSimilarity = !displayFeatureSimilarity;
		std::cout << "feature similarity " << (displayFeatureSimilarity ? "on" : "off") << "\n";
	}
	if( (event.getKeySym() == "f" || event.getKeySym() == "r" || event.getKeySym() == "N") && event.keyDown() ) {
		forceRedraw = true;
	}
	if( (event.getKeySym() == "i") && event.keyDown() ) {
		forceRedraw = true;
		index++;
	}
}





