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

#include <mrsmap/map/surfel.h>
#include <mrsmap/map/nodevalue.h>
#include <mrsmap/map/surfelpair.h>
#include <mrsmap/map/shapetexture_feature.h>

#include <gsl/gsl_rng.h>

#include <pcl/common/time.h>

#include <opencv2/opencv.hpp>

#include <tbb/tbb.h>
#include <tbb/concurrent_queue.h>

#include <flann/flann.h>


namespace mrsmap {


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

		typedef GNodeValue< GSurfel< 1 >, 6 > NodeValue;
		typedef GSurfel< 1 > Surfel;
		typedef GSurfelPair< 1 > SurfelPair;
		typedef std::unordered_map< SurfelPairKey, std::vector< SurfelPair* > > SurfelPairHashmap;
		typedef std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > SurfelPairVector;

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


		void addPoints( const boost::shared_ptr< const pcl::PointCloud<pcl::PointXYZRGB> >& cloud, const boost::shared_ptr< const std::vector< int > >& indices, bool smoothViewDir = false );

		void addPoints( const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const std::vector< int >& indices, bool smoothViewDir = false );

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
		int buildSurfelPairsForSurfel( spatialaggregate::OcTreeNode< float, NodeValue >* node, Surfel* srcSurfel, int surfelIdx, std::vector< spatialaggregate::OcTreeNode< float, NodeValue >* > & nodes, std::vector< SurfelPair, Eigen::aligned_allocator< SurfelPair > > & pairs, float & maxDist, float samplingRate = 1.f );

	    void buildSurfelPairs();

	    void buildSurfelPairsHashmap();

	    void buildSurfelPairsOnDepthParallel( std::vector< spatialaggregate::OcTreeNode< float, NodeValue >* >& nodes, int processDepth, float & maxDist );

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
		void indexNodesRecursive( spatialaggregate::OcTreeNode< float, NodeValue >* node, int minDepth, int maxDepth, bool includeBranchingNodes );

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




#endif /* MULTIRESOLUTION_SURFEL_MAP_H_ */

