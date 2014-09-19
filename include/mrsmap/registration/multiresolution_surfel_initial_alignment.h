/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, University of Bonn, Computer Science Institute VI
 *  Author: Joerg Stueckler, 03.08.2011
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

#ifndef MULTIRESOLUTION_SURFEL_INITIAL_ALIGNMENT_H_
#define MULTIRESOLUTION_SURFEL_INITIAL_ALIGNMENT_H_

#include "mrsmap/map/multiresolution_surfel_map.h"
#include "mrsmap/registration/multiresolution_surfel_registration.h"
#include "mrsmap/utilities/geometry.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
//#include <terminal_tools/time.h>

#define MIN_REF_SURFELS 100
#define MIN_SURFEL_PAIRS 100

namespace mrsmap {

class SurfelPairAssociation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SurfelPairAssociation( int src_ref_idx, int src_referred_idx, int dst_ref_idx, int dst_referred_idx, float alpha ) :
            src_ref_idx_( src_ref_idx ), src_referred_idx_( src_referred_idx ), dst_ref_idx_( dst_ref_idx ),  dst_referred_idx_( dst_referred_idx ), alpha_( alpha ) {
    }

    ~SurfelPairAssociation() {
    }

    int src_ref_idx_, src_referred_idx_, dst_ref_idx_, dst_referred_idx_;
    float alpha_;
};

class PoseCluster {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector< Geometry::Pose, Eigen::aligned_allocator< Geometry::Pose > > poses_;
    Geometry::Pose center_;
    double score_;

    void addPose( Geometry::Pose & pose, double score ) {
        poses_.push_back( pose );
        score_ += score;
        updateCenter();
    }

    void merge( PoseCluster & other ) {
        poses_.insert( poses_.end(), other.poses_.begin(), other.poses_.end() );
        score_ += other.score_;
        updateCenter();
    }

    void updateCenter() {
        Eigen::Vector3d translation = Eigen::Vector3d::Zero();
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Zero();

        for ( std::vector< Geometry::Pose, Eigen::aligned_allocator< Geometry::Pose > >::iterator currentPose =
              poses_.begin(); currentPose != poses_.end(); ++currentPose ) {
            translation += currentPose->position_;

            rotation += Eigen::Matrix3d( currentPose->orientation_ );
        }

        translation /= poses_.size();
        rotation /= poses_.size();
        Eigen::Matrix3d rotationTranspose = rotation.transpose();

        Eigen::JacobiSVD<Eigen::Matrix3d, 2> svd = rotationTranspose.jacobiSvd( Eigen::ComputeFullU | Eigen::ComputeFullV );
        Eigen::Matrix3d u = svd.matrixU();
        Eigen::Matrix3d v = svd.matrixV();

        if ( rotation.transpose().determinant() > 0.f ) {
            rotation = v * u.transpose().eval();
        } else {
            rotation = v * Eigen::DiagonalMatrix<double, 3>( 1.f, 1.f, -1.f ) * u.transpose().eval();
        }

        center_.position_ = translation;
        center_.orientation_ = Eigen::Quaterniond( rotation );
    }
};

class PoseWithScore {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseWithScore( Geometry::Pose& pose, float score ) {
        pose_ = pose;
        this->score_ = score;
    }

    PoseWithScore( const Eigen::Matrix<double, 8, 1> & pose ) {
        const Eigen::Vector7d p = pose.block<7,1>(0,0);
        pose_ = Geometry::Pose( p );
        this->score_ = pose(7);
    }

    Geometry::Pose pose_;
    float score_;
};

class PoseClustering {
public:
    PoseClustering() { }

    PoseClustering( std::vector<PoseWithScore, Eigen::aligned_allocator<PoseWithScore> >* poses, const float transThresh, const float rotThresh, Eigen::Vector4d & map_mean ) {
        poses_ = poses;        
        poseCloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr ( new pcl::PointCloud<pcl::PointXYZ>() );
        kdtree_ = pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr ( new pcl::KdTreeFLANN<pcl::PointXYZ>() );
        transThresh_ = transThresh;
        rotThresh_ = rotThresh;

        poseCloud_->reserve( poses->size() );

        for ( unsigned int p = 0; p< poses->size(); ++p ) {
            const PoseWithScore & pose =
                    (*poses)[p];
            const Eigen::Vector4d pos = pose.pose_.asMatrix4d().inverse().eval() * map_mean;
            pcl::PointXYZ point;
            point.x = pos(0);
            point.y = pos(1);
            point.z = pos(2);


            poseCloud_->points.push_back( point );
        }

        kdtree_->setInputCloud( poseCloud_ );
    }

    void getClusters( std::vector<pcl::PointIndices> & clusters ) {
        const unsigned int min_pts_per_cluster = 3;
        const unsigned int max_pts_per_cluster = 100000000;

        // Create a bool vector of processed point indices, and initialize it to false
        std::vector<bool> processed (poseCloud_->points.size (), false);

        std::vector<int> nn_indices;
        std::vector<float> nn_distances;
        // Process all points in the indices vector
        for (unsigned int i = 0; i < poseCloud_->points.size (); ++i)
        {
          if (processed[i])
            continue;

          std::vector<int> seed_queue;
          unsigned int sq_idx = 0;
          seed_queue.push_back (i);

          processed[i] = true;

          while (sq_idx < (seed_queue.size ()))
          {
              Eigen::Matrix3d sq_orientation = Eigen::Matrix3d( (*poses_)[seed_queue[sq_idx] ].pose_.orientation_ );
              sq_orientation.transposeInPlace();

            // Search for sq_idx
            if ( !kdtree_->radiusSearch (seed_queue[sq_idx], transThresh_, nn_indices, nn_distances) )
            {
              sq_idx++;
              continue;
            }

            for (size_t j = 1; j < nn_indices.size (); ++j)             // nn_indices[0] should be sq_idx
            {
              if (nn_indices[j] == -1 || processed[nn_indices[j]])        // Has this point been processed before ?
                continue;

              // check on orientation...
              Eigen::Matrix3d nn_orientation = Eigen::Matrix3d( (*poses_)[nn_indices[j]].pose_.orientation_ );
              Eigen::AngleAxisd diff( sq_orientation * nn_orientation );

              if ( fabsf(diff.angle()) < rotThresh_ ) {
                // Perform a simple Euclidean clustering
                seed_queue.push_back (nn_indices[j]);
                processed[nn_indices[j]] = true;
              }
            }

            sq_idx++;
          }

          // If this queue is satisfactory, add to the clusters
          if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
          {
            pcl::PointIndices r;
            r.indices.resize (seed_queue.size ());
            for (size_t j = 0; j < seed_queue.size (); ++j)
              r.indices[j] = seed_queue[j];

            // These two lines should not be needed: (can anyone confirm?) -FF
//            std::sort (r.indices.begin (), r.indices.end ());
//            r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());

            r.header = poseCloud_->header;
            clusters.push_back (r);   // We could avoid a copy by working directly in the vector
          }
        }
    }



    PoseWithScore getMeanPose( const pcl::PointIndices & indices ) {
        Eigen::Vector3d translation = Eigen::Vector3d::Zero();
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Zero();
        float totalScore = 0.f;

        for (std::vector<int>::const_iterator pit = indices.indices.begin(); pit != indices.indices.end(); pit++) {
            PoseWithScore pose = (*poses_)[*pit];
            translation += pose.pose_.position_;
            rotation += Eigen::Matrix3d( pose.pose_.orientation_ );
            totalScore += pose.score_;
        }

        translation /= indices.indices.size();
        rotation /= indices.indices.size();

        Geometry::Pose pose = Geometry::correctMeanPose( translation, rotation );
        return PoseWithScore( pose, totalScore);

    }

    void getCluster( const pcl::PointIndices & indices, Geometry::PoseCluster & cluster ) {
        for ( auto pit = indices.indices.begin(); pit != indices.indices.end(); pit++) {
            PoseWithScore pose = (*poses_)[*pit];
            cluster.add( pose.pose_, pose.score_ );
        }
        cluster.updateMean();
    }

    pcl::KdTree<pcl::PointXYZ>::Ptr kdtree_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr poseCloud_;
    std::vector<PoseWithScore, Eigen::aligned_allocator<PoseWithScore> >* poses_;
    float transThresh_;
    float rotThresh_;
};











typedef std::list<Eigen::Matrix<double, 8, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 8, 1> > > PoseList;
typedef std::vector<Eigen::Matrix<double, 8, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 8, 1> > > PoseVector;
typedef std::list<MultiResolutionSurfelMap::SurfelPair, Eigen::aligned_allocator<MultiResolutionSurfelMap::SurfelPair> > PairList;
typedef std::list<SurfelPairAssociation, Eigen::aligned_allocator<SurfelPairAssociation> > PairAssociationList;
typedef std::vector<SurfelPairAssociation, Eigen::aligned_allocator<SurfelPairAssociation> > PairAssociationVector;
typedef std::unordered_map<int, std::vector<unsigned int> > AccumulatorArray;

typedef std::unordered_map<int, PairAssociationList> VoteMap;


	class InitialAlignmentParams {
	public:
		InitialAlignmentParams()
            :  surfelSimilarVoteFactor_( 0.8 ), allSurfelsSimilarVoteFactor_( 0.8 ),
              minNumVotes_( 1 ), anglePrecision_( 15.f ), minNumPairs_( 100 ), clusterMaxTransDist_( 0.05f ),
              clusterMaxRotDist_( 0.1f ), minNumSurfelPairs_( 100 ), useFeatures_( 1 ), maxFeatureDist_( 0.1 ), maxNumClusters_( 5 )
		{

		}

        InitialAlignmentParams( float surfelSimilarVoteFactor, float allSurfelsSimilarVoteFactor, unsigned int minNumVotes, float anglePrecision, unsigned int minNumPairs, float clusterMaxTransDist,
                                float clusterMaxRotDist )
            :  surfelSimilarVoteFactor_( surfelSimilarVoteFactor ), allSurfelsSimilarVoteFactor_( allSurfelsSimilarVoteFactor ),
              minNumVotes_( minNumVotes ), anglePrecision_(anglePrecision), minNumPairs_( minNumPairs ), clusterMaxTransDist_( clusterMaxTransDist ),
              clusterMaxRotDist_( clusterMaxRotDist ), minNumSurfelPairs_( 100 ), useFeatures_( 1 ), maxFeatureDist_( 0.1 ), maxNumClusters_( 5 )
		{

		}

		~InitialAlignmentParams() {

		}

        float surfelSimilarVoteFactor_;
        float allSurfelsSimilarVoteFactor_;
        unsigned int minNumVotes_;
        float anglePrecision_;
        unsigned int minNumPairs_;
        float clusterMaxTransDist_;
        float clusterMaxRotDist_;
        unsigned int minNumSurfelPairs_;
        unsigned int useFeatures_;
        double maxFeatureDist_;
        unsigned int maxNumClusters_;

	};

    bool surfelPairVoting( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
                       PoseVector & poseVotesAllDepths,
                        std::vector< Geometry::PoseCluster >& clusters,
                        std::vector<MultiResolutionSurfelMap::Surfel*> & associatedRefSurfels,
                        Eigen::Vector4d & map_mean,
                        const InitialAlignmentParams& params = InitialAlignmentParams() );

};


#endif /* MULTIRESOLUTION_SURFEL_INITIAL_ALIGNMENT_H_ */


