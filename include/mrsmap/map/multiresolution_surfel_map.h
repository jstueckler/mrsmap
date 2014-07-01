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

#ifndef MULTIRESOLUTION_SURFEL_MAP_H_
#define MULTIRESOLUTION_SURFEL_MAP_H_

#include <Eigen/Core>
#include <Eigen/Eigen>

#include <vector>
#include <set>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/functional/hash.hpp>
//#include <boost/unordered_map.hpp>
#include <unordered_map>
#include <functional>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/vector_average.h>

#include <octreelib/spatialaggregate/octree.h>
#include <octreelib/algorithm/downsample.h>

#include <mrsmap/map/shapetexture_feature.h>

#include <gsl/gsl_rng.h>

#include <pcl/common/time.h>

#include <opencv2/opencv.hpp>

#include <tbb/tbb.h>
#include <tbb/concurrent_queue.h>

#include <flann/flann.h>


#define MAX_NUM_SURFELS 6

#define MIN_SURFEL_POINTS 10.0
#define MAX_SURFEL_POINTS 10000.0 //stop at this point count, since the sums may get numerically unstable



namespace mrsmap {

	// TODO (Jan): move optional fields to vectors indexed by surfel idx.

	class Surfel {
	public:
		Surfel() {
			clear();
		}

		~Surfel() {}

		inline void clear() {

			num_points_ = 0.0;
			mean_.setZero();
			cov_.setZero();

			up_to_date_ = false;
			applyUpdate_ = true;
			unevaluated_ = false;

			assocWeight_ = 1.f;

			idx_ = -1;

			reference_pose_set = false;

			seenThrough_ = false;

		}


		inline Surfel& operator+=(const Surfel& rhs) {

			if( rhs.num_points_ > 0 && num_points_ < MAX_SURFEL_POINTS ) {

				// numerically stable one-pass update scheme
				if( num_points_ <= std::numeric_limits<double>::epsilon() ) {
					cov_ = rhs.cov_;
					mean_ = rhs.mean_;
					num_points_ = rhs.num_points_;
				}
				else {
					const Eigen::Matrix< double, 6, 1 > deltaS = rhs.num_points_ * mean_ - num_points_ * rhs.mean_;
					cov_ += rhs.cov_ + 1.0 / (num_points_ * rhs.num_points_ * (rhs.num_points_ + num_points_)) * deltaS * deltaS.transpose();
					mean_ += rhs.mean_;
					num_points_ += rhs.num_points_;
				}

				first_view_dir_ = rhs.first_view_dir_;
				up_to_date_ = false;
			}

			return *this;
		}


		inline void add( const Eigen::Matrix< double, 6, 1 >& point ) {
			// numerically stable one-pass update scheme
			if( num_points_ < std::numeric_limits<double>::epsilon() ) {
				mean_ += point;
				num_points_ += 1.0;
				up_to_date_ = false;
			}
			else if( num_points_ < MAX_SURFEL_POINTS ) {
				const Eigen::Matrix< double, 6, 1 > deltaS = (mean_ - num_points_ * point);
				cov_ += 1.0 / (num_points_ * (num_points_ + 1.0)) * deltaS * deltaS.transpose();
				mean_ += point;
				num_points_ += 1.0;
				up_to_date_ = false;
			}
		}


		inline void evaluateNormal() {

			Eigen::Matrix< double, 3, 1> eigen_values_;
			Eigen::Matrix< double, 3, 3> eigen_vectors_;

			// eigen vectors are stored in the columns
			pcl::eigen33(Eigen::Matrix3d(cov_.block<3,3>(0,0)), eigen_vectors_, eigen_values_);

			normal_ = eigen_vectors_.col(0);
			if( normal_.dot( first_view_dir_ ) > 0.0 )
				normal_ *= -1.0;

		}

		inline void evaluate() {

			if( num_points_ >= MIN_SURFEL_POINTS ) {

				const double inv_num = 1.0 / num_points_;
				mean_ *= inv_num;
				cov_ /= (num_points_-1.0);


				// enforce symmetry..
				cov_(1,0) = cov_(0,1);
				cov_(2,0) = cov_(0,2);
				cov_(3,0) = cov_(0,3);
				cov_(4,0) = cov_(0,4);
				cov_(5,0) = cov_(0,5);
				cov_(2,1) = cov_(1,2);
				cov_(2,3) = cov_(3,2);
				cov_(2,4) = cov_(4,2);
				cov_(2,5) = cov_(5,2);
				cov_(3,1) = cov_(1,3);
				cov_(3,4) = cov_(4,3);
				cov_(3,5) = cov_(5,3);
				cov_(4,1) = cov_(1,4);
				cov_(4,5) = cov_(5,4);
				cov_(5,1) = cov_(1,5);

				double det = cov_.block<3,3>(0,0).determinant();

				if( det <= std::numeric_limits<double>::epsilon() ) {

					// TODO (Jan): make this optional in a special kind of surfel
					// pull out surfels in a separate header, templated surfel class, derived from a base class

//					cov_(0,0) += 0.000000001;
//					cov_(1,1) += 0.000000001;
//					cov_(2,2) += 0.000000001;

					mean_.setZero();
					cov_.setZero();

					num_points_ = 0;

					clear();
				}

			}


			up_to_date_ = true;
			unevaluated_ = false;

		}


		inline void unevaluate() {

			if( num_points_ > 0.0 ) {

				mean_ *= num_points_;
				cov_ *= (num_points_-1.0);

				unevaluated_ = true;

			}

		}

		// transforms from local surfel frame to map frame
		inline void updateReferencePose() {
			Eigen::Vector3d pos = mean_.block<3,1>(0,0);
			Eigen::AngleAxisd refRot( -acos ( normal_.dot( Eigen::Vector3d::UnitX () ) ),
									normal_.cross( Eigen::Vector3d::UnitX () ).normalized () );

			reference_pose_.block<3,1>(0,0) = pos;
			Eigen::Quaterniond q( refRot );
			reference_pose_(3,0) = q.x();
			reference_pose_(4,0) = q.y();
			reference_pose_(5,0) = q.z();
			reference_pose_(6,0) = q.w();

			reference_pose_set = true;
		}

	  Eigen::Matrix< double, 3, 1 > initial_view_dir_, first_view_dir_;

	  double num_points_;
	  Eigen::Matrix< double, 6, 1 > mean_;
	  Eigen::Matrix< double, 3, 1 > normal_;
	  Eigen::Matrix< double, 6, 6 > cov_;
	  bool up_to_date_, applyUpdate_;
	  bool unevaluated_;

	  // TODO (Jan): move to outside vector on idx_
	  float assocDist_;
	  float assocWeight_;

	  bool seenThrough_; // TODO (Jan) pull this out to a vector

	  int idx_;

	  Eigen::Matrix< double, 7, 1 > reference_pose_;
	  bool reference_pose_set;

	  ShapeTextureFeature simple_shape_texture_features_; // TODO (Jan): move to outside vector on idx_
	  ShapeTextureFeature agglomerated_shape_texture_features_;

	public:
	  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	};


	class PointFeature {
	public:
		PointFeature() {}

		PointFeature( unsigned int descriptorLength ) {
			has_depth_ = false;
		}
		~PointFeature() {}

		Eigen::Vector2d image_pos_;
		Eigen::Matrix2d image_cov_, image_assoc_cov_; // pos covariance
		Eigen::Vector3d origin_;
		Eigen::Quaterniond orientation_;
		Eigen::Vector4d pos_; // (x,y,d) relative to given transform (origin and orientation)
		Eigen::Matrix4d cov_; //, assoc_cov_; // pos covariance

		Eigen::Vector3d invzpos_; // (x,y,1/d) relative to given transform (origin and orientation), in inverse depth parametrization
		Eigen::Matrix3d invzcov_, invzinvcov_; //, assoc_invzcov_; // pos covariance, in inverse depth parametrization

		bool has_depth_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};



	class MultiResolutionSurfelMap
	{
	public:


		class Params {
		public:
			Params();

			~Params() {}

			float	depthNoiseFactor;
			float	pixelNoise;

			float	depthNoiseAssocFactor;
			float	pixelNoiseAssocFactor;

			bool	usePointFeatures;
			bool	debugPointFeatures;

			unsigned int	GridCols;
			unsigned int	GridRows;
			unsigned int	GridCellMax;
			unsigned int	numPointFeatures;

			double dist_dependency;


		    float surfelPairFeatureBinAngle_;
		    float surfelPairFeatureBinDist_;
		    float surfelPairSamplingRate_;
		    float surfelPairMaxDist_;
		    float surfelPairMaxDistResFactor_;
		    int surfelPairMinDepth_;
		    int surfelPairMaxDepth_;
		    bool surfelPairUseColor_;

		    bool parallel_;

		};



		// TODO (JAN): move out to surfel_pair header
		struct SurfelPairKey {
		    SurfelPairKey( ) : shape1_(0 ), shape2_(0 ),
		        shape3_(0 ), shape4_(0 ), color1_(0 ), color2_(0 ) {}

		    SurfelPairKey( uint64 shape1, uint64 shape2, uint64 shape3, uint64 shape4,
		                   uint64 color1, uint64 color2, uint64 color3 )
		        : shape1_(shape1 ), shape2_(shape2 ),
		            shape3_(shape3 ), shape4_(shape4 ), color1_(color1 ), color2_(color2 ),
		            color3_(color3 ) {}

		    bool operator==( const SurfelPairKey & other ) const {
		        return shape1_ == other.shape1_ && shape2_ == other.shape2_ &&
		                shape3_ == other.shape3_ && shape4_ == other.shape4_ &&
		                color1_ == other.color1_ && color2_ == other.color2_ &&
		                color3_ == other.color3_;
		    }

		    size_t hash() const {
		        std::size_t seed = 0;
		        boost::hash_combine(seed, shape1_);
		        boost::hash_combine(seed, shape2_);
		        boost::hash_combine(seed, shape3_);
		        boost::hash_combine(seed, shape4_);
		        boost::hash_combine(seed, color1_);
		        boost::hash_combine(seed, color2_);
		        boost::hash_combine(seed, color3_);
		        return seed;
		    }

		    uint64 shape1_;
		    uint64 shape2_;
		    uint64 shape3_;
		    uint64 shape4_;
		    uint64 color1_;
		    uint64 color2_;
		    uint64 color3_;
		};



		// TODO (JAN): move out to surfel_pair header
		/**
		 * Signature for a surfel pair
		 */
		class SurfelPairSignature {
		public:
		    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		    SurfelPairSignature() {
		    }

		    ~SurfelPairSignature() {
		    }

		    // distance, dot products of n1 to d, n2 to d, and n1 to n2
		    Eigen::Matrix< double, 4, 1 > shape_signature_;

		    // distance in L, alpha, beta color space
		    Eigen::Matrix< double, 3, 1 > color_signature_;
//		    Eigen::Matrix< double, 3, 1 > color_signature_src_;
//		    Eigen::Matrix< double, 3, 1 > color_signature_dst_;

		    // hash key
		    inline SurfelPairKey getKey( const float maxDist, const float bin_dist, const float bin_angle, const bool use_color = true ) {

		            const int bins_chrom = 5;
		            const int bins_lum = 3;
		            const int bins_angle = (int) 360 / bin_angle;

		            uint64 bin_s1 = (int)( shape_signature_(0) / bin_dist );
		            uint64 bin_s2 = std::max( 0, std::min( bins_angle-1, (int)(shape_signature_(1) * ((double)bins_angle)) ) );
		            uint64 bin_s3 = std::max( 0, std::min( bins_angle-1, (int)(shape_signature_(2) * ((double)bins_angle)) ) );
		            uint64 bin_s4 = std::max( 0, std::min( bins_angle-1, (int)(shape_signature_(3) * ((double)bins_angle)) ) );

//		            const double lumHighThreshold = 0.5 + LUMINANCE_BIN_THRESHOLD;
//		            const double lumLowThreshold = 0.5 - LUMINANCE_BIN_THRESHOLD;
//		            const double colorHighThreshold = 0.5 + COLOR_BIN_THRESHOLD;
//		            const double colorLowThreshold = 0.5 - COLOR_BIN_THRESHOLD;

		            uint64 bin_c1 = 0;
		            uint64 bin_c2 = 0;
		            uint64 bin_c3 = 0;

		            if ( use_color ) {
		                bin_c1 = std::max( 0, std::min( bins_lum-1, (int)(color_signature_(0) * ((double)bins_lum)) ) );
		                bin_c2 = std::max( 0, std::min( bins_chrom-1, (int)(color_signature_(1) * ((double)bins_chrom)) ) );
		                bin_c3 = std::max( 0, std::min( bins_chrom-1, (int)(color_signature_(2) * ((double)bins_chrom)) ) );
		            }

		//            key_ = 0;
		//            key_ |= bin_s1;
		//            key_ |= bin_s2 << 8;
		//            key_ |= bin_s3 << 16;
		//            key_ |= bin_s4 << 24;
		//            key_ |= bin_c1 << 32;
		//            key_ |= bin_c2 << 36;
		//            key_ |= bin_c3 << 40;
		//            key_ |= bin_c4 << 44;
		//            key_ |= bin_c5 << 48;
		//            key_ |= bin_c6 << 52;
		//        }

		            key_ = SurfelPairKey( bin_s1, bin_s2, bin_s3, bin_s4,
		                                  bin_c1, bin_c2, bin_c3 );

		        return key_;

		    }

		    float alpha_;
		    SurfelPairKey key_;

		};


		// TODO (JAN): move out to surfel_pair header
		/**
		 * Signature for a surfel pair
		 */
		class SurfelPair {
		public:
		    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		    SurfelPair( Surfel* src, Surfel* dst, const SurfelPairSignature& signature, float weight = 0.f )
		        : src_(src), dst_(dst), signature_( signature ),   weight_( weight ) {

			}

			~SurfelPair() {
			}

			Surfel* src_;
			Surfel* dst_;

		    SurfelPairSignature signature_;

		    float weight_;
		};



		// TODO (JAN): move out to node_value header
		class NodeValue {
		public:
			NodeValue() {
				initialize();
			}

			NodeValue( unsigned int v ) {
				initialize();
			}

			~NodeValue() {}

			inline void initialize() {

				idx_ = -1;
				associated_ = 0;
				assocWeight_ = 1.f;
				border_ = false;

				surfels_[0].initial_view_dir_ = Eigen::Vector3d( 1., 0., 0. );
				surfels_[1].initial_view_dir_ = Eigen::Vector3d( -1., 0., 0. );
				surfels_[2].initial_view_dir_ = Eigen::Vector3d( 0., 1., 0. );
				surfels_[3].initial_view_dir_ = Eigen::Vector3d( 0., -1., 0. );
				surfels_[4].initial_view_dir_ = Eigen::Vector3d( 0., 0., 1. );
				surfels_[5].initial_view_dir_ = Eigen::Vector3d( 0., 0., -1. );

				for( unsigned int i = 0; i < 6; i++ )
					surfels_[i].first_view_dir_ = surfels_[i].initial_view_dir_;

			}


			inline NodeValue& operator+=(const NodeValue& rhs) {

				// merge surfels
				for( unsigned int i = 0; i < 6; i++ ) {

					Surfel& surfel = surfels_[i];

					if( surfel.applyUpdate_ ) {
						if( surfel.up_to_date_ )
							surfel.clear();

						surfel += rhs.surfels_[i];
					}

				}

				return *this;
			}


			inline Surfel* getSurfel( const Eigen::Vector3d& viewDirection ) {

				Surfel* bestMatchSurfel = NULL;
				double bestMatchDist = -1.;

				for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {
					const double dist = viewDirection.dot( surfels_[i].initial_view_dir_ );
					if( dist > bestMatchDist ) {
						bestMatchSurfel = &surfels_[i];
						bestMatchDist = dist;
					}
				}

				return bestMatchSurfel;
			}


			inline void addSurfel( const Eigen::Vector3d& viewDirection, const Surfel& surfel ) {

				// find best matching surfel for the view direction
				Surfel* bestMatchSurfel = getSurfel( viewDirection );

				if( bestMatchSurfel->applyUpdate_ ) {
					if( bestMatchSurfel->up_to_date_ )
						bestMatchSurfel->clear();

					*bestMatchSurfel += surfel;
				}

			}


			inline void evaluateNormals( spatialaggregate::OcTreeNode<float, NodeValue>* node ) {
				for( unsigned int i = 0; i < 6; i++ ) {
					if( !surfels_[i].up_to_date_ || surfels_[i].unevaluated_ ) {

						Surfel surfel = surfels_[i];
						for( unsigned int n = 0; n < 27; n++ ) {
							if(node->neighbors_[n] && node->neighbors_[n] != node ) {
								surfel += node->neighbors_[n]->value_.surfels_[i];
							}
						}

						surfel.first_view_dir_ = surfels_[i].first_view_dir_;
						surfel.evaluate();
						surfel.evaluateNormal();

						surfels_[i].normal_ = surfel.normal_;

					}
				}
			}


			inline void evaluateSurfels() {
				for( unsigned int i = 0; i < 6; i++ ) {
					if( !surfels_[i].up_to_date_ || surfels_[i].unevaluated_ ) {
						surfels_[i].evaluate();
					}
				}
			}

			inline void unevaluateSurfels() {
				for( unsigned int i = 0; i < 6; i++ ) {
					if( surfels_[i].up_to_date_ ) {
						surfels_[i].unevaluate();
					}
				}
			}



			Surfel surfels_[6];

			// TODO: move these out to a vector on idx_
			char associated_; // -1: disabled, 0: not associated, 1: associated, 2: not associated but neighbor of associated node
			spatialaggregate::OcTreeNode<float, NodeValue>* association_;
			char assocSurfelIdx_, assocSurfelDstIdx_;
			float assocWeight_;

			int idx_;

			bool border_;

			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		};


		class ImagePreAllocator {
		public:
			ImagePreAllocator();

			~ImagePreAllocator();

			struct Info {
				Info() : value( NULL ) {}
				Info( NodeValue* v, const spatialaggregate::OcTreeKey< float, NodeValue >& k, unsigned int d )
				: value(v), key(k), depth(d) {}

				NodeValue* value;
				spatialaggregate::OcTreeKey< float, NodeValue > key;
				unsigned int depth;
			};

			void prepare( unsigned int w, unsigned int h, bool buildNodeImage );

			spatialaggregate::DynamicAllocator< NodeValue > imageNodeAllocator_;
			uint64_t* imgKeys;
			NodeValue** valueMap;
			std::vector< Info > infoList;
			tbb::concurrent_vector< std::vector< Info > > parallelInfoList;
			unsigned int width, height;
			spatialaggregate::OcTreeNode<float, NodeValue>** node_image_;
			std::set< spatialaggregate::OcTreeNode<float, NodeValue>* > node_set_;

			boost::mutex mutex_;

		};


		MultiResolutionSurfelMap( float minResolution, float radius, boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > > allocator = boost::make_shared< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > >() );

		~MultiResolutionSurfelMap();


		void extents( Eigen::Matrix< double, 3, 1 >& mean, Eigen::Matrix< double, 3, 3 >& cov );


		struct NodeSurfel {
			spatialaggregate::OcTreeNode< float, NodeValue >* node;
			Surfel* surfel;
		};


		void addPoints( const boost::shared_ptr< const pcl::PointCloud<pcl::PointXYZRGB> >& cloud, const boost::shared_ptr< const std::vector< int > >& indices );

		void addPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void addImage( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, bool smoothViewDir = true, bool buildNodeImage = false );

		void addDisplacementImage( const pcl::PointCloud<pcl::PointXYZRGB>& cloud_pos,  const pcl::PointCloud<pcl::PointXYZRGB>& cloud_disp, bool smoothViewDir = true, bool buildNodeImage = false );

		void addImagePointFeatures( const cv::Mat& img, const pcl::PointCloud< pcl::PointXYZRGB >& cloud );

		void getImage( cv::Mat& img, const Eigen::Vector3d& viewPosition );

		static inline bool splitCriterion( spatialaggregate::OcTreeNode< float, NodeValue >* oldLeaf, spatialaggregate::OcTreeNode< float, NodeValue >* newLeaf );

		void findImageBorderPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::vector< int >& indices );

		void findVirtualBorderPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::vector< int >& indices );

		void findForegroundBorderPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::vector< int >& indices );

		void findContourPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::vector< int >& indices );

		void clearAtPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void markNoUpdateAtPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void clearUpdateSurfelsAtPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void markBorderAtPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices );

		void markBorderFromViewpoint( const Eigen::Vector3d& viewpoint );

		static inline void clearBorderFlagFunction( spatialaggregate::OcTreeNode< float, NodeValue >* current, spatialaggregate::OcTreeNode< float, NodeValue >* next, void* data );
		void clearBorderFlag();

		void markUpdateAllSurfels();
		static inline void markUpdateAllSurfelsFunction( spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data );

		void evaluateSurfels();
		void unevaluateSurfels();

		bool pointInForeground( const Eigen::Vector3f& position, const cv::Mat& image_depth, const cv::Point2f imagePoint, float scale, float jumpThreshold );

		void setApplyUpdate( bool v );

		void setUpToDate( bool v );

		void clearUnstableSurfels();


		void buildShapeTextureFeatures();


		void clearAssociatedFlag();
		void distributeAssociatedFlag();

		void clearAssociationDist();

		void clearSeenThroughFlag();

		void clearAssociations();
		static inline void clearAssociationsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);


		std::vector< unsigned int > findInliers( const std::vector< unsigned int >& indices, const pcl::PointCloud<pcl::PointXYZRGB>& cloud, int maxDepth );


		inline bool buildSurfelPair( SurfelPairSignature & signature, const Surfel& src, const Surfel& dst );
		int buildSurfelPairsForSurfel( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, Surfel* srcSurfel, int surfelIdx, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* > & nodes, std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > & pairs, float & maxDist, float samplingRate = 1.f );

	    void buildSurfelPairs();

	    void buildSurfelPairsHashmap();

	    void buildSurfelPairsOnDepthParallel( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >& nodes, int processDepth, float & maxDist );

	    void buildSurfelPairsHashmapOnDepth( int processDepth  );

	    void buildSamplingMap();



		void visualize3DColorDistribution( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool random = true );
		void visualize3DColorDistributionWithNormals( pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr, int depth, int viewDir, bool random = true, int numSamples = 100 );
		void visualize3DColorMeans( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir );

		void visualizeContours( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, const Eigen::Matrix4d& transform, int depth, int viewDir, bool random = true );

		void visualizeSimilarity( spatialaggregate::OcTreeNode< float, NodeValue >* referenceNode, pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool simple );

		void visualizeBorders( pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloudPtr, int depth, int viewDir, bool foreground );

		void visualizeNormals( pcl::PointCloud< pcl::PointXYZRGBNormal >::Ptr cloudPtr, int depth, int viewDir );


		static inline void evaluateNormalsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void evaluateSurfelsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void unevaluateSurfelsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void clearUnstableSurfelsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void setApplyUpdateFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void setUpToDateFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void clearAssociatedFlagFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void distributeAssociatedFlagFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void clearAssociationDistFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void clearSeenThroughFlagFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void buildSimpleShapeTextureFeatureFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void buildAgglomeratedShapeTextureFeatureFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualize3DColorDistributionFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualize3DColorDistributionWithNormalsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeContoursFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeSimilarityFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeBordersFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeNormalsFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);
		static inline void visualizeMeansFunction(spatialaggregate::OcTreeNode<float, NodeValue>* current, spatialaggregate::OcTreeNode<float, NodeValue>* next, void* data);

		void save( const std::string& filename );
		void load( const std::string& filename );


		void indexNodes( int minDepth, int maxDepth, bool includeBranchingNodes = true );
		void indexNodesRecursive( spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* node, int minDepth, int maxDepth, bool includeBranchingNodes );

		unsigned int numSurfels();
		unsigned int numSurfelsRecursive( spatialaggregate::OcTreeNode< float, NodeValue >* node );

		boost::shared_ptr< spatialaggregate::OcTree<float, NodeValue> > octree_;
		boost::shared_ptr< spatialaggregate::OcTreeNodeAllocator< float, NodeValue > > allocator_;
		boost::shared_ptr< ImagePreAllocator > imageAllocator_;

		Eigen::Matrix4d reference_pose_;


		float min_resolution_, max_range_;

		int last_pair_surfel_idx_;

		static gsl_rng* r;

//		std_msgs::Header header;

		pcl::StopWatch stopwatch_;

		std::vector< spatialaggregate::OcTreeNode<float, NodeValue>* > indexedNodes_;

		std::vector< PointFeature, Eigen::aligned_allocator< PointFeature > > features_;
		cv::Mat descriptors_;
		boost::shared_ptr< flann::Index< flann::HammingPopcnt< unsigned char > > > lsh_index_;


		typedef std::unordered_map< SurfelPairKey, std::vector< SurfelPair* > > SurfelPairHashmap;
		typedef std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > SurfelPairVector;

		// key is feature descriptor, value is index of surfel pairs list in all_surfel_pairs
		std::vector< SurfelPairHashmap > surfel_pair_list_map_;

		// list of surfel pairs for reference surfel
		std::vector< std::vector< SurfelPairVector > > all_surfel_pairs_;

		// list of reference surfels by depth
		std::vector< std::vector< Surfel* > > reference_surfels_;

		algorithm::OcTreeSamplingVectorMap< float, NodeValue > samplingMap_;

		float surfelMaxDist_;

		Params params_;

		cv::Mat img_rgb_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	};


};


namespace std {
	template<>
	class hash< mrsmap::MultiResolutionSurfelMap::SurfelPairKey >
		: public std::unary_function< mrsmap::MultiResolutionSurfelMap::SurfelPairKey, size_t >
	{
	public:
		size_t operator()( const mrsmap::MultiResolutionSurfelMap::SurfelPairKey& k ) const
		{
			return k.hash();
		}
	};
};

#endif /* MULTIRESOLUTION_SURFEL_MAP_H_ */

