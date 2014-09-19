/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, University of Bonn, Computer Science Institute VI
 *  Author: Joerg Stueckler, 17.08.2011
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

#ifndef GRIDS_H_
#define GRIDS_H_

#include <ostream>
#include <list>

#define SIGNATURE_ANGLES 16
#define SIGNATURE_RADII 2
#define SIGNATURE_ELEMENTS 12



class Grid3D {
public:


	Grid3D( const Eigen::Vector3d& gridMax, const Eigen::Vector3d& gridMin, double gridResolution ) {

		gridMax_ = gridMax;
		gridMin_ = gridMin;
		gridResolution_ = gridResolution;
		invGridResolution_ = 1.0 / gridResolution;
		binsX_ = ceil( (gridMax(0) - gridMin(0)) / gridResolution );
		binsY_ = ceil( (gridMax(1) - gridMin(1)) / gridResolution );
		binsZ_ = ceil( (gridMax(2) - gridMin(2)) / gridResolution );
		binsXY_ = binsX_*binsY_;
		binsXYZ_ = binsXY_*binsZ_;
		grid_ = new double[ binsXYZ_ ];
		setZero();

	}

	~Grid3D() {
		delete[] grid_;
	}


	void setZero() {
		for( int i = 0; i < binsXYZ_; i++ )
			grid_[i] = 0.0;
	}

	void add( const Eigen::Vector3d& pos, double v ) {

		const double x = std::min( (double)(binsX_-1), std::max( 0.0, (pos(0) - gridMin_(0)) * invGridResolution_ ) );
		const double y = std::min( (double)(binsY_-1), std::max( 0.0, (pos(1) - gridMin_(1)) * invGridResolution_ ) );
		const double z = std::min( (double)(binsZ_-1), std::max( 0.0, (pos(2) - gridMin_(2)) * invGridResolution_ ) );

		const double xf = (double)((int)x); const double dxf = x-xf;
		const double yf = (double)((int)y); const double dyf = y-yf;
		const double zf = (double)((int)z); const double dzf = z-zf;

		const double xc = std::min( (double)(binsX_-1), xf+1.0 ); const double dxc = 1.0-dxf;
		const double yc = std::min( (double)(binsY_-1), yf+1.0 ); const double dyc = 1.0-dyf;
		const double zc = std::min( (double)(binsZ_-1), zf+1.0 ); const double dzc = 1.0-dzf;

		grid_[ ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ] += v * dxf * dyf * dzf;
		grid_[ ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ] += v * dxc * dyf * dzf;
		grid_[ ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ] += v * dxf * dyc * dzf;
		grid_[ ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ] += v * dxc * dyc * dzf;
		grid_[ ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ] += v * dxf * dyf * dzc;
		grid_[ ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ] += v * dxc * dyf * dzc;
		grid_[ ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ] += v * dxf * dyc * dzc;
		grid_[ ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ] += v * dxc * dyc * dzc;


//			int x = std::min( binsX_-1, std::max( 0, (int)( (pos(0) - gridMin_(0)) * invGridResolution_ ) ) );
//			int y = std::min( binsY_-1, std::max( 0, (int)( (pos(1) - gridMin_(1)) * invGridResolution_ ) ) );
//			int z = std::min( binsZ_-1, std::max( 0, (int)( (pos(2) - gridMin_(2)) * invGridResolution_ ) ) );
//
//			grid_[ z * binsXY_ + y * binsX_ + x ] += v;


	}

	int getMaximum( int& x, int& y, int& z ) {
		double maxValue = -std::numeric_limits<double>::max();
		int maxCell = -1;
		for( int i = 0; i < binsXYZ_; i++ ) {
			if( grid_[i] > maxValue ) {
				maxValue = grid_[i];
				maxCell = i;
			}
		}
		if( maxCell == -1 )
			return -1;
		z = floor(maxCell / (binsXY_));
		y = floor( (maxCell-z*binsXY_) / binsX_ );
		x = maxCell - z*binsXY_ - y*binsX_;

		return maxCell;
	}

	int getMaximum( Eigen::Vector3d& pos ) {
		int x = 0, y = 0, z = 0;
		int i = getMaximum( x, y, z );
		pos(0) = gridMin_(0) + ((double)x) * gridResolution_;
		pos(1) = gridMin_(1) + ((double)y) * gridResolution_;
		pos(2) = gridMin_(2) + ((double)z) * gridResolution_;
		return i;
	}

	void set( const Eigen::Vector3d& pos, double v ) {

		int x = std::min( (double)(binsX_-1), std::max( 0.0, (pos(0) - gridMin_(0)) * invGridResolution_ ) );
		int y = std::min( (double)(binsY_-1), std::max( 0.0, (pos(1) - gridMin_(1)) * invGridResolution_ ) );
		int z = std::min( (double)(binsZ_-1), std::max( 0.0, (pos(2) - gridMin_(2)) * invGridResolution_ ) );

		grid_[ z * binsXY_ + y * binsX_ + x ] = v;
	}


	double* grid_;
	int binsX_, binsY_, binsZ_;
	int binsXY_, binsXYZ_;
	Eigen::Vector3d gridMax_, gridMin_;
	double gridResolution_, invGridResolution_;
};


class PoseListGrid3D : public Grid3D {
public:

	typedef std::list< Eigen::Matrix< double, 8, 1 >, Eigen::aligned_allocator< Eigen::Matrix< double, 8, 1 > > > PoseList;
	typedef std::pair< PoseList::iterator, double > PoseValuePair;

	PoseListGrid3D( const Eigen::Vector3d& gridMax, const Eigen::Vector3d& gridMin, double gridResolution )
	: Grid3D( gridMax, gridMin, gridResolution ) {

		listGrid_ = new std::list< PoseValuePair >[ binsXYZ_ ];

	}

	~PoseListGrid3D() {
//		delete[] grid_;
		delete[] listGrid_;
	}

	void add( const Eigen::Vector3d& pos, double v, const PoseList::iterator& it ) {

		const double x = std::min( (double)(binsX_-1), std::max( 0.0, (pos(0) - gridMin_(0)) * invGridResolution_ ) );
		const double y = std::min( (double)(binsY_-1), std::max( 0.0, (pos(1) - gridMin_(1)) * invGridResolution_ ) );
		const double z = std::min( (double)(binsZ_-1), std::max( 0.0, (pos(2) - gridMin_(2)) * invGridResolution_ ) );

		const double xf = (double)((int)x); const double dxf = x-xf;
		const double yf = (double)((int)y); const double dyf = y-yf;
		const double zf = (double)((int)z); const double dzf = z-zf;

		const double xc = std::min( (double)(binsX_-1), xf+1.0 ); const double dxc = 1.0-dxf;
		const double yc = std::min( (double)(binsY_-1), yf+1.0 ); const double dyc = 1.0-dyf;
		const double zc = std::min( (double)(binsZ_-1), zf+1.0 ); const double dzc = 1.0-dzf;

		grid_[ ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ] += v * dxf * dyf * dzf;
		listGrid_[ ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ].push_back( PoseValuePair( it, v * dxf * dyf * dzf ) );

		grid_[ ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ] += v * dxc * dyf * dzf;
		listGrid_[ ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ].push_back( PoseValuePair( it, v * dxc * dyf * dzf ) );

		grid_[ ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ] += v * dxf * dyc * dzf;
		listGrid_[ ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ].push_back( PoseValuePair( it, v * dxf * dyc * dzf ) );

		grid_[ ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ] += v * dxc * dyc * dzf;
		listGrid_[ ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ].push_back( PoseValuePair( it, v * dxc * dyc * dzf ) );

		grid_[ ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ] += v * dxf * dyf * dzc;
		listGrid_[ ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ].push_back( PoseValuePair( it, v * dxf * dyf * dzc ) );

		grid_[ ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ] += v * dxc * dyf * dzc;
		listGrid_[ ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ].push_back( PoseValuePair( it, v * dxc * dyf * dzc ) );

		grid_[ ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ] += v * dxf * dyc * dzc;
		listGrid_[ ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ].push_back( PoseValuePair( it, v * dxf * dyc * dzc ) );

		grid_[ ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ] += v * dxc * dyc * dzc;
		listGrid_[ ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ].push_back( PoseValuePair( it, v * dxc * dyc * dzc ) );

	}


	std::list< PoseValuePair >* listGrid_;
};


class Grid4D {
public:


	Grid4D( const Eigen::Vector4d& gridMax, const Eigen::Vector4d& gridMin, double gridResolution ) {

		gridMax_ = gridMax;
		gridMin_ = gridMin;
		gridResolution_ = gridResolution;
		invGridResolution_ = 1.0 / gridResolution;
		binsX_ = ceil( (gridMax(0) - gridMin(0)) / gridResolution );
		binsY_ = ceil( (gridMax(1) - gridMin(1)) / gridResolution );
		binsZ_ = ceil( (gridMax(2) - gridMin(2)) / gridResolution );
		binsW_ = ceil( (gridMax(3) - gridMin(3)) / gridResolution );
		binsXY_ = binsX_*binsY_;
		binsXYZ_ = binsXY_*binsZ_;
		binsXYZW_ = binsXYZ_*binsW_;
		grid_ = new double[ binsXYZW_ ];
		setZero();

	}

	~Grid4D() {
		delete[] grid_;
	}


	void setZero() {
		for( int i = 0; i < binsXYZW_; i++ )
			grid_[i] = 0.0;
	}

	void add( const Eigen::Vector4d& pos, double v ) {

		const double x = std::min( (double)(binsX_-1), std::max( 0.0, (pos(0) - gridMin_(0)) * invGridResolution_ ) );
		const double y = std::min( (double)(binsY_-1), std::max( 0.0, (pos(1) - gridMin_(1)) * invGridResolution_ ) );
		const double z = std::min( (double)(binsZ_-1), std::max( 0.0, (pos(2) - gridMin_(2)) * invGridResolution_ ) );
		const double w = std::min( (double)(binsW_-1), std::max( 0.0, (pos(3) - gridMin_(3)) * invGridResolution_ ) );

		const double xf = (double)((int)x); const double dxf = x-xf;
		const double yf = (double)((int)y); const double dyf = y-yf;
		const double zf = (double)((int)z); const double dzf = z-zf;
		const double wf = (double)((int)w); const double dwf = w-wf;

		const double xc = std::min( (double)(binsX_-1), xf+1.0 ); const double dxc = 1.0-dxf;
		const double yc = std::min( (double)(binsY_-1), yf+1.0 ); const double dyc = 1.0-dyf;
		const double zc = std::min( (double)(binsZ_-1), zf+1.0 ); const double dzc = 1.0-dzf;
		const double wc = std::min( (double)(binsW_-1), wf+1.0 ); const double dwc = 1.0-dwf;

		grid_[ ((int)wc) * binsXYZ_ + ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ] += v * dxf * dyf * dzf * dwf;
		grid_[ ((int)wc) * binsXYZ_ + ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ] += v * dxc * dyf * dzf * dwf;
		grid_[ ((int)wc) * binsXYZ_ + ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ] += v * dxf * dyc * dzf * dwf;
		grid_[ ((int)wc) * binsXYZ_ + ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ] += v * dxc * dyc * dzf * dwf;
		grid_[ ((int)wc) * binsXYZ_ + ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ] += v * dxf * dyf * dzc * dwf;
		grid_[ ((int)wc) * binsXYZ_ + ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ] += v * dxc * dyf * dzc * dwf;
		grid_[ ((int)wc) * binsXYZ_ + ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ] += v * dxf * dyc * dzc * dwf;
		grid_[ ((int)wc) * binsXYZ_ + ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ] += v * dxc * dyc * dzc * dwf;

		grid_[ ((int)wf) * binsXYZ_ + ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ] += v * dxf * dyf * dzf * dwc;
		grid_[ ((int)wf) * binsXYZ_ + ((int)zc) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ] += v * dxc * dyf * dzf * dwc;
		grid_[ ((int)wf) * binsXYZ_ + ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ] += v * dxf * dyc * dzf * dwc;
		grid_[ ((int)wf) * binsXYZ_ + ((int)zc) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ] += v * dxc * dyc * dzf * dwc;
		grid_[ ((int)wf) * binsXYZ_ + ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xc) ] += v * dxf * dyf * dzc * dwc;
		grid_[ ((int)wf) * binsXYZ_ + ((int)zf) * binsXY_ + ((int)yc) * binsX_ + ((int)xf) ] += v * dxc * dyf * dzc * dwc;
		grid_[ ((int)wf) * binsXYZ_ + ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xc) ] += v * dxf * dyc * dzc * dwc;
		grid_[ ((int)wf) * binsXYZ_ + ((int)zf) * binsXY_ + ((int)yf) * binsX_ + ((int)xf) ] += v * dxc * dyc * dzc * dwc;

//			int x = std::min( binsX_-1, std::max( 0, (int)( (pos(0) - gridMin_(0)) * invGridResolution_ ) ) );
//			int y = std::min( binsY_-1, std::max( 0, (int)( (pos(1) - gridMin_(1)) * invGridResolution_ ) ) );
//			int z = std::min( binsZ_-1, std::max( 0, (int)( (pos(2) - gridMin_(2)) * invGridResolution_ ) ) );
//			int w = std::min( binsW_-1, std::max( 0, (int)( (pos(3) - gridMin_(3)) * invGridResolution_ ) ) );
//
//			grid_[ w * binsXYZ_ + z * binsXY_ + y * binsX_ + x ] += v;

	}

	int getMaximum( int& x, int& y, int& z, int& w ) {
		double maxValue = -std::numeric_limits<double>::max();
		int maxCell = -1;
		for( int i = 0; i < binsXYZW_; i++ ) {
			if( grid_[i] > maxValue ) {
				maxValue = grid_[i];
				maxCell = i;
			}
		}
		if( maxCell == -1 )
			return -1;

		w = floor( maxCell / (binsXYZ_) );
		z = floor( (maxCell-w*binsXYZ_) / (binsXY_));
		y = floor( (maxCell-w*binsXYZ_-z*binsXY_) / binsX_ );
		x = maxCell - w*binsXYZ_ - z*binsXY_ - y*binsX_;

		return maxCell;
	}

	int getMaximum( Eigen::Vector4d& pos ) {
		int x = 0, y = 0, z = 0, w = 0;
		int i = getMaximum( x, y, z, w );
		pos(0) = gridMin_(0) + ((double)x) * gridResolution_;
		pos(1) = gridMin_(1) + ((double)y) * gridResolution_;
		pos(2) = gridMin_(2) + ((double)z) * gridResolution_;
		pos(3) = gridMin_(3) + ((double)w) * gridResolution_;
		return i;
	}

	double* grid_;
	double* listGrid_;
	int binsX_, binsY_, binsZ_, binsW_;
	int binsXY_, binsXYZ_, binsXYZW_;
	Eigen::Vector4d gridMax_, gridMin_;
	double gridResolution_, invGridResolution_;
};






template< int NumChannels >
class PolarGrid2D {
public:

	PolarGrid2D( int angleBins, int radiusBins, double maxRadius )
	: angleBins_(angleBins), radiusBins_(radiusBins) {

		// the radius range [0,maxRadius] is divided into fixed number of bins
		// we place the bins into the old centers to evenly distribute them in the range and to
		// express both sides, the minima and maxima equally

		angleResolution_ = 2.0*M_PI / ((double)angleBins);
		invAngleResolution_ = 1.0 / angleResolution_;

		radiusResolution_ = maxRadius / ((double)radiusBins);
		invRadiusResolution_ = 1.0 / radiusResolution_;

		minAngle_ = 0.0;
		maxAngle_ = 2.0*M_PI - angleResolution_;

		minRadius_ = 0.5*radiusResolution_;
		maxRadius_ = maxRadius - 0.5*radiusResolution_;

		binsAC_ = NumChannels*angleBins;
		binsRAC_ = NumChannels*angleBins*radiusBins;
		grid_ = new double[ binsRAC_ ];
		setZero();

	}

	~PolarGrid2D() {
		delete[] grid_;
	}

	void setZero() {
		for( int i = 0; i < binsRAC_; i++ )
			grid_[i] = 0.0;
	}

	void multiply( double v ) {
		for( int i = 0; i < binsRAC_; i++ )
			grid_[i] *= v;
	}

	void add( const Eigen::Vector2d& pos, const Eigen::Matrix< double, NumChannels, 1 >& v ) {

		const double angle = std::min( (double)(angleBins_-1), std::max( 0.0, ((M_PI + atan2( pos(1), pos(0) ) - minAngle_) * invAngleResolution_ ) ) );
		const double radius = std::min( (double)(radiusBins_-1), std::max( 0.0, (pos.norm() * invRadiusResolution_ - minRadius_) ) );

		const double anglef = (double)((int)angle); const double danglef = angle-anglef;
		const double radiusf = (double)((int)radius); const double dradiusf = radius-radiusf;

		double anglec = anglef+1.0; const double danglec = 1.0-danglef;
		if( anglec >= angleBins_ ) anglec = 0.0; // angular wraparound!!
		const double radiusc = std::min( (double)(radiusBins_-1), radiusf+1.0 ); const double dradiusc = 1.0-dradiusf;

		const int cc = NumChannels * (((int)radiusc) * angleBins_ + ((int)anglec));
		const int cf = NumChannels * (((int)radiusc) * angleBins_ + ((int)anglef));
		const int fc = NumChannels * (((int)radiusf) * angleBins_ + ((int)anglec));
		const int ff = NumChannels * (((int)radiusf) * angleBins_ + ((int)anglef));
		const float dcc = danglec * dradiusc;
		const float dcf = danglec * dradiusf;
		const float dfc = danglef * dradiusc;
		const float dff = danglef * dradiusf;

		for( int c = 0; c < NumChannels; c++ ) {
			grid_[ cc + c ] += v(c) * dff;
			grid_[ cf + c ] += v(c) * dcf;
			grid_[ fc + c ] += v(c) * dfc;
			grid_[ ff + c ] += v(c) * dcc;
		}


//		int angle = std::min( angleBins_-1, std::max( 0, (int)((M_PI + atan2( pos(1), pos(0) )) * invAngleResolution_) ) );
//		int radius = std::min( radiusBins_-1, std::max( 0, (int)( pos.norm() * invRadiusResolution_  ) ) );
//
//		for( int c = 0; c < NumChannels; c++ ) {
//			grid_[ radius * binsAC_ + angle * NumChannels + c ] += v(c);
//		}


	}

	friend std::ostream& operator<<( std::ostream& os, const PolarGrid2D<NumChannels>& grid ) {

		for( int r = 0; r < grid.radiusBins_; r++ )
			for( int a = 0; a < grid.angleBins_; a++ ) {
				// position of radius/angle bin
				double radius = grid.minRadius_ + ((double)r) * grid.radiusResolution_;
				double angle = grid.minAngle_ + ((double)a) * grid.angleResolution_ - M_PI;
				double x = radius * cos(angle);
				double y = radius * sin(angle);

				os << radius << " " << angle << " " << x << " " << y;

				for( int c = 0; c < NumChannels; c++ ) {
					os << " " << grid.grid_[ r * grid.binsAC_ + a * NumChannels + c ];
				}
				os << "\n";
			}

		return os;
	}


	double* grid_;
	int angleBins_, radiusBins_, binsRAC_, binsAC_;
	double minRadius_, maxRadius_, minAngle_, maxAngle_;
	double angleResolution_, invAngleResolution_;
	double radiusResolution_, invRadiusResolution_;


};


template< int NumChannels >
class SignedPolarGrid2D {
public:

	SignedPolarGrid2D( int angleBins, int radiusBins, double maxRadius, const Eigen::Matrix< double, NumChannels, 1 >& thresholds )
	: angleBins_(angleBins), radiusBins_(radiusBins), thresholds_(thresholds) {

		// the radius range [0,maxRadius] is divided into fixed number of bins
		// we place the bins into the old centers to evenly distribute them in the range and to
		// express both sides, the minima and maxima equally

		angleResolution_ = 2.0*M_PI / ((double)angleBins);
		invAngleResolution_ = 1.0 / angleResolution_;

		radiusResolution_ = maxRadius / ((double)radiusBins);
		invRadiusResolution_ = 1.0 / radiusResolution_;

		minAngle_ = 0.0;
		maxAngle_ = 2.0*M_PI - angleResolution_;

		minRadius_ = 0.5*radiusResolution_;
		maxRadius_ = maxRadius - 0.5*radiusResolution_;

		binsAC_ = NumChannels*angleBins;
		binsRAC_ = NumChannels*angleBins*radiusBins;
		grid_pos_ = new double[ binsRAC_ ];
		grid_neg_ = new double[ binsRAC_ ];
		setZero();

	}

	~SignedPolarGrid2D() {
		delete[] grid_pos_;
		delete[] grid_neg_;
	}

	void setZero() {
		for( int i = 0; i < binsRAC_; i++ ) {
			grid_pos_[i] = 0.0;
			grid_neg_[i] = 0.0;
		}
	}

	void multiply( double v ) {
		for( int i = 0; i < binsRAC_; i++ ) {
			grid_pos_[i] *= v;
			grid_neg_[i] *= v;
		}
	}

	void add( const Eigen::Vector2d& pos, const Eigen::Matrix< double, NumChannels, 1 >& v ) {

		const double angle = std::min( (double)(angleBins_-1), std::max( 0.0, ((M_PI + atan2( pos(1), pos(0) ) - minAngle_) * invAngleResolution_ ) ) );
		const double radius = std::min( (double)(radiusBins_-1), std::max( 0.0, (pos.norm() * invRadiusResolution_ - minRadius_) ) );

		const double anglef = (double)((int)angle); const double danglef = angle-anglef;
		const double radiusf = (double)((int)radius); const double dradiusf = radius-radiusf;

		double anglec = anglef+1.0; const double danglec = 1.0-danglef;
		if( anglec >= angleBins_ ) anglec = 0.0; // angular wraparound!!
		const double radiusc = std::min( (double)(radiusBins_-1), radiusf+1.0 ); const double dradiusc = 1.0-dradiusf;

		const int cc = NumChannels * (((int)radiusc) * angleBins_ + ((int)anglec));
		const int cf = NumChannels * (((int)radiusc) * angleBins_ + ((int)anglef));
		const int fc = NumChannels * (((int)radiusf) * angleBins_ + ((int)anglec));
		const int ff = NumChannels * (((int)radiusf) * angleBins_ + ((int)anglef));
		const float dcc = danglec * dradiusc;
		const float dcf = danglec * dradiusf;
		const float dfc = danglef * dradiusc;
		const float dff = danglef * dradiusf;

		for( int c = 0; c < NumChannels; c++ ) {

			if( v(c) >= thresholds_(c) ) {
				grid_pos_[ cc + c ] += v(c) * dff;
				grid_pos_[ cf + c ] += v(c) * dcf;
				grid_pos_[ fc + c ] += v(c) * dfc;
				grid_pos_[ ff + c ] += v(c) * dcc;
			}
			else if( v(c) <= -thresholds_(c) ) {
				grid_neg_[ cc + c ] += v(c) * dff;
				grid_neg_[ cf + c ] += v(c) * dcf;
				grid_neg_[ fc + c ] += v(c) * dfc;
				grid_neg_[ ff + c ] += v(c) * dcc;
			}

		}


//		int angle = std::min( angleBins_-1, std::max( 0, (int)((M_PI + atan2( pos(1), pos(0) )) * invAngleResolution_) ) );
//		int radius = std::min( radiusBins_-1, std::max( 0, (int)( pos.norm() * invRadiusResolution_  ) ) );
//
//		for( int c = 0; c < NumChannels; c++ ) {
//			grid_[ radius * binsAC_ + angle * NumChannels + c ] += v(c);
//		}


	}


	friend std::ostream& operator<<( std::ostream& os, const SignedPolarGrid2D<NumChannels>& grid ) {

		for( int r = 0; r < grid.radiusBins_; r++ ) {
			for( int a = 0; a < grid.angleBins_; a++ ) {
				// position of radius/angle bin
				double radius = grid.minRadius_ + ((double)r) * grid.radiusResolution_;
				double angle = grid.minAngle_ + ((double)a) * grid.angleResolution_ - M_PI;
				double x = radius * cos(angle);
				double y = radius * sin(angle);

				os << radius << " " << angle << " " << x << " " << y;

				for( int c = 0; c < NumChannels; c++ ) {
					os << " " << grid.grid_pos_[ r * grid.binsAC_ + a * NumChannels + c ] << " " << grid.grid_neg_[ r * grid.binsAC_ + a * NumChannels + c ];
				}
				os << "\n";
			}
		}

		return os;
	}


	double* grid_pos_;
	double* grid_neg_;
	int angleBins_, radiusBins_, binsRAC_, binsAC_;
	double minRadius_, maxRadius_, minAngle_, maxAngle_;
	double angleResolution_, invAngleResolution_;
	double radiusResolution_, invRadiusResolution_;
	Eigen::Matrix< double, NumChannels, 1 > thresholds_;


};



template< int NumChannels >
class ThreeLayeredPolarGrid2D {
public:

	ThreeLayeredPolarGrid2D() {
		grid_positive_ = NULL;
		grid_center_ = NULL;
		grid_negative_ = NULL;
	}

	ThreeLayeredPolarGrid2D( int angleBins, int radiusBins, double maxRadius, const Eigen::Matrix< double, NumChannels, 1 >& centerRadii, const Eigen::Matrix< double, NumChannels, 1 >& margins )
	: angleBins_(angleBins), radiusBins_(radiusBins), centerRadii_(centerRadii), margins_(margins) {

		// the radius range [0,maxRadius] is divided into fixed number of bins
		// we place the bins into the old centers to evenly distribute them in the range and to
		// express both sides, the minima and maxima equally

		for( int i = 0; i < NumChannels; i++ )
			invMargins_(i) = 1.0 / margins_(i);

		angleResolution_ = 2.0*M_PI / ((double)angleBins);
		invAngleResolution_ = 1.0 / angleResolution_;

		radiusResolution_ = maxRadius / ((double)radiusBins);
		invRadiusResolution_ = 1.0 / radiusResolution_;

		minAngle_ = 0.0;
		maxAngle_ = 2.0*M_PI - angleResolution_;

		minRadius_ = 0.5*radiusResolution_;
		maxRadius_ = maxRadius - 0.5*radiusResolution_;

		binsAC_ = NumChannels*angleBins;
		binsRAC_ = NumChannels*angleBins*radiusBins;
		grid_positive_ = new double[ binsRAC_ ];
		grid_center_ = new double[ binsRAC_ ];
		grid_negative_ = new double[ binsRAC_ ];
		setZero();

	}

	ThreeLayeredPolarGrid2D( const ThreeLayeredPolarGrid2D& grid ) {

		angleBins_ = grid.angleBins_;
		radiusBins_ = grid.radiusBins_;
		centerRadii_ = grid.centerRadii_;
		margins_ = grid.margins_;
		invMargins_ = grid.invMargins_;
		angleResolution_ = grid.angleResolution_;
		invAngleResolution_ = grid.invAngleResolution_;
		radiusResolution_ = grid.radiusResolution_;
		invRadiusResolution_ = grid.invRadiusResolution_;
		minAngle_ = grid.minAngle_;
		maxAngle_ = grid.maxAngle_;
		minRadius_ = grid.minRadius_;
		maxRadius_ = grid.maxRadius_;
		binsAC_ = grid.binsAC_;
		binsRAC_ = grid.binsRAC_;

		grid_positive_ = new double[ binsRAC_ ];
		memcpy( grid_positive_, grid.grid_positive_, binsRAC_*sizeof(double) );

		grid_center_ = new double[ binsRAC_ ];
		memcpy( grid_center_, grid.grid_center_, binsRAC_*sizeof(double) );

		grid_negative_ = new double[ binsRAC_ ];
		memcpy( grid_negative_, grid.grid_negative_, binsRAC_*sizeof(double) );

	}

	ThreeLayeredPolarGrid2D& operator=( const ThreeLayeredPolarGrid2D& grid ) {

		if(grid_positive_)
			delete[] grid_positive_;
		if(grid_center_)
			delete[] grid_center_;
		if(grid_negative_)
			delete[] grid_negative_;

		angleBins_ = grid.angleBins_;
		radiusBins_ = grid.radiusBins_;
		centerRadii_ = grid.centerRadii_;
		margins_ = grid.margins_;
		invMargins_ = grid.invMargins_;
		angleResolution_ = grid.angleResolution_;
		invAngleResolution_ = grid.invAngleResolution_;
		radiusResolution_ = grid.radiusResolution_;
		invRadiusResolution_ = grid.invRadiusResolution_;
		minAngle_ = grid.minAngle_;
		maxAngle_ = grid.maxAngle_;
		minRadius_ = grid.minRadius_;
		maxRadius_ = grid.maxRadius_;
		binsAC_ = grid.binsAC_;
		binsRAC_ = grid.binsRAC_;

		grid_positive_ = new double[ binsRAC_ ];
		memcpy( grid_positive_, grid.grid_positive_, binsRAC_*sizeof(double) );

		grid_center_ = new double[ binsRAC_ ];
		memcpy( grid_center_, grid.grid_center_, binsRAC_*sizeof(double) );

		grid_negative_ = new double[ binsRAC_ ];
		memcpy( grid_negative_, grid.grid_negative_, binsRAC_*sizeof(double) );

		return *this;
	}


	~ThreeLayeredPolarGrid2D() {
		if(grid_positive_)
			delete[] grid_positive_;
		if(grid_center_)
			delete[] grid_center_;
		if(grid_negative_)
			delete[] grid_negative_;
	}

	void setZero() {
		for( int i = 0; i < binsRAC_; i++ ) {
			grid_positive_[i] = 0.0;
			grid_center_[i] = 0.0;
			grid_negative_[i] = 0.0;
		}
	}

	void setConstant( double v ) {
		for( int i = 0; i < binsRAC_; i++ ) {
			grid_positive_[i] = v;
			grid_center_[i] = v;
			grid_negative_[i] = v;
		}
	}

	void multiply( double v ) {
		for( int i = 0; i < binsRAC_; i++ ) {
			grid_positive_[i] *= v;
			grid_center_[i] *= v;
			grid_negative_[i] *= v;
		}
	}

	void add( const Eigen::Vector2d& pos, const Eigen::Matrix< double, NumChannels, 1 >& v ) {

		double angle = atan2( pos(1), pos(0) );
		if( angle < minAngle_ )
			angle += 2.0*M_PI;
		angle = std::min( (double)(angleBins_-1), std::max( 0.0, (angle - minAngle_) * invAngleResolution_ ) );

		const double radius = std::min( (double)(radiusBins_-1), std::max( 0.0, (pos.norm() * invRadiusResolution_ - minRadius_) ) );

		const double angle_ctr = (double)((int)angle); const double dangle_ctr = angle-angle_ctr;
		const double radius_ctr = (double)((int)radius); const double dradius_ctr = radius-radius_ctr;

		// smoothly overlapping bins:
		// distribute partially on neighboring bins (soft decisions)
		double angle_low = angle_ctr-1.0; const double wangle_low = 1.0-dangle_ctr;
		if( angle_low < 0 ) // angular wraparound!!
			angle_low = angleBins_-1.0;

		double angle_upp = angle_ctr+1.0; const double wangle_upp = dangle_ctr;
		if( angle_upp >= angleBins_ )
			angle_upp = 0.0; // angular wraparound!!

		const double radius_low = std::max( 0.0, radius_ctr-1.0 ); const double wradius_low = 1.0-dradius_ctr;
		const double radius_upp = std::min( (double)(radiusBins_-1), radius_ctr+1.0 ); const double wradius_upp = dradius_ctr;

		const int uu = NumChannels * (((int)radius_upp) * angleBins_ + ((int)angle_upp));
		const int uc = NumChannels * (((int)radius_upp) * angleBins_ + ((int)angle_ctr));
		const int ul = NumChannels * (((int)radius_upp) * angleBins_ + ((int)angle_low));

		const int cu = NumChannels * (((int)radius_ctr) * angleBins_ + ((int)angle_upp));
		const int cc = NumChannels * (((int)radius_ctr) * angleBins_ + ((int)angle_ctr));
		const int cl = NumChannels * (((int)radius_ctr) * angleBins_ + ((int)angle_low));

		const int lu = NumChannels * (((int)radius_low) * angleBins_ + ((int)angle_upp));
		const int lc = NumChannels * (((int)radius_low) * angleBins_ + ((int)angle_ctr));
		const int ll = NumChannels * (((int)radius_low) * angleBins_ + ((int)angle_low));

		const double wuu = wradius_upp * wangle_upp;
		const double wuc = wradius_upp;
		const double wul = wradius_upp * wangle_low;

		const double wcu = wangle_upp;
		const double wcc = 1.0;
		const double wcl = wangle_low;

		const double wlu = wradius_low * wangle_upp;
		const double wlc = wradius_low;
		const double wll = wradius_low * wangle_low;

		for( int c = 0; c < NumChannels; c++ ) {

			double dpositive = 0.0;
			double dcenter = 0.0;
			double dnegative = 0.0;

			if( v(c) >= centerRadii_(c) ) {
				dpositive = 1.0;
				dcenter = std::max( 0.0, 1.0 - (v(c) - centerRadii_(c)) * invMargins_(c) );
			}
			if( v(c) >= 0 && v(c) <= centerRadii_(c) ) {
				dcenter = 1.0;
				dpositive = std::max( 0.0, 1.0 - (centerRadii_(c) - v(c)) * invMargins_(c) );
			}
			if( v(c) <= 0 && v(c) >= -centerRadii_(c) ) {
				dcenter = 1.0;
				dnegative = std::max( 0.0, 1.0 - (v(c) + centerRadii_(c)) * invMargins_(c) );
			}
			if( v(c) <= -centerRadii_(c) ) {
				dcenter = std::max( 0.0, 1.0 - (-v(c)-centerRadii_(c)) * invMargins_(c) );
				dnegative = 1.0;
			}

			grid_positive_[ uu + c ] += dpositive * wuu;
			grid_positive_[ uc + c ] += dpositive * wuc;
			grid_positive_[ ul + c ] += dpositive * wul;
			grid_positive_[ cu + c ] += dpositive * wcu;
			grid_positive_[ cc + c ] += dpositive * wcc;
			grid_positive_[ cl + c ] += dpositive * wcl;
			grid_positive_[ lu + c ] += dpositive * wlu;
			grid_positive_[ lc + c ] += dpositive * wlc;
			grid_positive_[ ll + c ] += dpositive * wll;

			grid_center_[ uu + c ] += dcenter * wuu;
			grid_center_[ uc + c ] += dcenter * wuc;
			grid_center_[ ul + c ] += dcenter * wul;
			grid_center_[ cu + c ] += dcenter * wcu;
			grid_center_[ cc + c ] += dcenter * wcc;
			grid_center_[ cl + c ] += dcenter * wcl;
			grid_center_[ lu + c ] += dcenter * wlu;
			grid_center_[ lc + c ] += dcenter * wlc;
			grid_center_[ ll + c ] += dcenter * wll;

			grid_negative_[ uu + c ] += dnegative * wuu;
			grid_negative_[ uc + c ] += dnegative * wuc;
			grid_negative_[ ul + c ] += dnegative * wul;
			grid_negative_[ cu + c ] += dnegative * wcu;
			grid_negative_[ cc + c ] += dnegative * wcc;
			grid_negative_[ cl + c ] += dnegative * wcl;
			grid_negative_[ lu + c ] += dnegative * wlu;
			grid_negative_[ lc + c ] += dnegative * wlc;
			grid_negative_[ ll + c ] += dnegative * wll;


//			double pz = 0.5 + 0.5 * dpositive * dff;
//			double pocc = grid_positive_[ cc + c ];
//			grid_positive_[ cc + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//			pz = 0.5 + 0.5 * dpositive * dcf;
//			pocc = grid_positive_[ cf + c ];
//			grid_positive_[ cf + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//			pz = 0.5 + 0.5 * dpositive * dfc;
//			pocc = grid_positive_[ fc + c ];
//			grid_positive_[ fc + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//			pz = 0.5 + 0.5 * dpositive * dcc;
//			pocc = grid_positive_[ ff + c ];
//			grid_positive_[ ff + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//
//			pz = 0.5 + 0.5 * dcenter * dff;
//			pocc = grid_center_[ cc + c ];
//			grid_center_[ cc + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//			pz = 0.5 + 0.5 * dcenter * dcf;
//			pocc = grid_center_[ cf + c ];
//			grid_center_[ cf + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//			pz = 0.5 + 0.5 * dcenter * dfc;
//			pocc = grid_center_[ fc + c ];
//			grid_center_[ fc + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//			pz = 0.5 + 0.5 * dcenter * dcc;
//			pocc = grid_center_[ ff + c ];
//			grid_center_[ ff + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//
//			pz = 0.5 + 0.5 * dnegative * dff;
//			pocc = grid_negative_[ cc + c ];
//			grid_negative_[ cc + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//			pz = 0.5 + 0.5 * dnegative * dcf;
//			pocc = grid_negative_[ cf + c ];
//			grid_negative_[ cf + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//			pz = 0.5 + 0.5 * dnegative * dfc;
//			pocc = grid_negative_[ fc + c ];
//			grid_negative_[ fc + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
//
//			pz = 0.5 + 0.5 * dnegative * dcc;
//			pocc = grid_negative_[ ff + c ];
//			grid_negative_[ ff + c ] = pz * pocc / (pz * pocc + (1.0-pz) * (1.0-pocc));
		}


//		int angle = std::min( angleBins_-1, std::max( 0, (int)((M_PI + atan2( pos(1), pos(0) )) * invAngleResolution_) ) );
//		int radius = std::min( radiusBins_-1, std::max( 0, (int)( pos.norm() * invRadiusResolution_  ) ) );
//
//		for( int c = 0; c < NumChannels; c++ ) {
//			grid_[ radius * binsAC_ + angle * NumChannels + c ] += v(c);
//		}


	}


	friend std::ostream& operator<<( std::ostream& os, const ThreeLayeredPolarGrid2D<NumChannels>& grid ) {

		for( int r = 0; r < grid.radiusBins_; r++ ) {
			for( int a = 0; a < grid.angleBins_; a++ ) {
				// position of radius/angle bin
				double radius = grid.minRadius_ + ((double)r) * grid.radiusResolution_;
				double angle = grid.minAngle_ + ((double)a) * grid.angleResolution_;
				double x = radius * cos(angle);
				double y = radius * sin(angle);

				os << radius << " " << angle << " " << x << " " << y;

				for( int c = 0; c < NumChannels; c++ ) {
					os << " " << grid.grid_positive_[ r * grid.binsAC_ + a * NumChannels + c ] << " " << grid.grid_center_[ r * grid.binsAC_ + a * NumChannels + c ] << " " << grid.grid_negative_[ r * grid.binsAC_ + a * NumChannels + c ];
				}
				os << "\n";
			}
		}

		return os;
	}


	void binarizeRotated( std::bitset< SIGNATURE_ANGLES*SIGNATURE_RADII*SIGNATURE_ELEMENTS >& signature, double angle, double threshold ) const {

		// sub-angle accuracy: linearly interpolate between bins
		double angleOffset = angle * invAngleResolution_;
		double aomult = angleOffset / ((double)angleBins_);
		if( aomult < 0.0 )
			angleOffset += ceil(-aomult) * angleBins_;

		assert( angleOffset >= 0.0 );

		double df = angleOffset - floor(angleOffset);
		double dc = 1.0 - df;
		unsigned int angleOffsetBin = (unsigned int) floor(angleOffset);

		for( int r = 0; r < radiusBins_; r++ ) {
			for( int a = 0; a < angleBins_; a++ ) {

				int signature_idx = r*angleBins_ + a;

				// pick lower and upper bin
				unsigned int a_low = (a + angleOffsetBin) % ((unsigned int)angleBins_);
				unsigned int a_upp = (a + angleOffsetBin + 1) % ((unsigned int)angleBins_);

				unsigned int idx_low = NumChannels*(r*angleBins_ + a_low);
				unsigned int idx_upp = NumChannels*(r*angleBins_ + a_upp);

				double shapeP = dc * grid_positive_[idx_low] + df * grid_positive_[idx_upp];
				double shapeC = dc * grid_center_[idx_low] + df * grid_center_[idx_upp];
				double shapeN = dc * grid_negative_[idx_low] + df * grid_negative_[idx_upp];
				double LP = dc * grid_positive_[idx_low+1] + df * grid_positive_[idx_upp+1];
				double LC = dc * grid_center_[idx_low+1] + df * grid_center_[idx_upp+1];
				double LN = dc * grid_negative_[idx_low+1] + df * grid_negative_[idx_upp+1];
				double aP = dc * grid_positive_[idx_low+2] + df * grid_positive_[idx_upp+2];
				double aC = dc * grid_center_[idx_low+2] + df * grid_center_[idx_upp+2];
				double aN = dc * grid_negative_[idx_low+2] + df * grid_negative_[idx_upp+2];
				double bP = dc * grid_positive_[idx_low+3] + df * grid_positive_[idx_upp+3];
				double bC = dc * grid_center_[idx_low+3] + df * grid_center_[idx_upp+3];
				double bN = dc * grid_negative_[idx_low+3] + df * grid_negative_[idx_upp+3];

				if( shapeC > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+0 ] = 1;
				if( shapeP > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+1 ] = 1;
				if( shapeN > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+2 ] = 1;

				if( LC > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+3 ] = 1;
				if( LP > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+4 ] = 1;
				if( LN > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+5 ] = 1;

				if( aC > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+6 ] = 1;
				if( aP > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+7 ] = 1;
				if( aN > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+8 ] = 1;

				if( bC > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+9 ] = 1;
				if( bP > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+10 ] = 1;
				if( bN > threshold )
					signature[ signature_idx*SIGNATURE_ELEMENTS+11 ] = 1;
			}
		}

	}


	double* grid_positive_;
	double* grid_center_;
	double* grid_negative_;
	int angleBins_, radiusBins_, binsRAC_, binsAC_;
	double minRadius_, maxRadius_, minAngle_, maxAngle_;
	double angleResolution_, invAngleResolution_;
	double radiusResolution_, invRadiusResolution_;
	Eigen::Matrix< double, NumChannels, 1 > centerRadii_, margins_, invMargins_;


};

#endif /* GRIDS_H_ */



