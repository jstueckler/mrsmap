/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, University of Bonn, Computer Science Institute VI
 *  Author: Joerg Stueckler, 04.06.2011
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

#include "mrsmap/registration/multiresolution_surfel_initial_alignment.h"

#include "mrsmap/utilities/grids.h"

#include "mrsmap/utilities/logging.h"

//#include "mrsmap/visualization/visualization.h"
//#include "mrsmap/util/timer.h"

//#include <fstream>
#include <list>



#define MAX_CLUSTERS 64
#define MAX_VIEWDIR_DIST cos( 0.25 * M_PI + 0.125*M_PI )
#define SURFEL_MATCH_ANGLE_THRESHOLD  0.5
#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif



namespace mrsmap {

bool poseInWindow( const Eigen::Matrix<double, 8, 1>& p, const Eigen::Matrix<double, 8, 1>& q, float posRadius,
                   float rotRadius ) {
    return fabsf( p( 0 ) - q( 0 ) ) < posRadius && fabsf( p( 1 ) - q( 1 ) ) < posRadius
            && fabsf( p( 2 ) - q( 2 ) ) < posRadius && fabsf( p( 3 ) - q( 3 ) ) < rotRadius
            && fabsf( p( 4 ) - q( 4 ) ) < rotRadius && fabsf( p( 5 ) - q( 5 ) ) < rotRadius
            && fabsf( p( 6 ) - q( 6 ) ) < rotRadius;
}

bool poseInWindow2( const Eigen::Matrix<double, 8, 1>& p, const Eigen::Matrix<double, 8, 1>& q, float posRadius2,
                    float rotRadius2 ) {
    return ( p.block<3, 1>( 0, 0 ) - q.block<3, 1>( 0, 0 ) ).squaredNorm() < posRadius2
            && ( p.block<4, 1>( 3, 0 ) - q.block<4, 1>( 3, 0 ) ).squaredNorm() < rotRadius2;
}

bool comparePoses( const Eigen::Matrix<double, 8, 1>& p, const Eigen::Matrix<double, 8, 1>& q ) {
    return ( p( 7 ) > q( 7 ) );
}

inline float computeMedian ( std::vector<float> & values ) {
    //    values.sort();
    std::sort( values.begin(), values.end(),  std::greater<float>());
    int count = values.size();
    int middleIndex = count / 2;
    float middle = values[middleIndex];
    if ( (count & 1) == 0 ) {
        return (middle + values[middleIndex-1]) / 2.f;
    } else {
        return middle;
    }
}

int viewDirectionIndex( Eigen::Vector3d & viewDirection ) {
    if ( viewDirection(0) > 0.f )
        return 0;
    else if ( viewDirection(0) < 0.f )
        return 1;
    else if ( viewDirection(1) > 0.f )
        return 2;
    else if ( viewDirection(1) < 0.f )
        return 3;
    else if ( viewDirection(2) > 0.f )
        return 4;
    else if ( viewDirection(2) < 0.f )
        return 5;
    else return -1;
}

bool surfelPairVoting( MultiResolutionSurfelMap& source, MultiResolutionSurfelMap& target,
                    PoseVector & poseVotesAllDepths,
                    std::vector< Geometry::PoseCluster >& clusters,
                    std::vector<MultiResolutionSurfelMap::Surfel*> & associatedRefSurfels,
                    Eigen::Vector4d & map_mean,
                    const InitialAlignmentParams& params ) {

    // align coarse to fine, but do not align on too sparse (coarse) resolutions...
    // select best fit cluster in pose space after all resolutions have been processed.

    const float surfelSimilarVoteFactor = params.surfelSimilarVoteFactor_;
    const float allSurfelsSimilarVoteFactor = params.allSurfelsSimilarVoteFactor_;
    const unsigned int minNumVotes = params.minNumVotes_;
    const float anglePrecision = params.anglePrecision_;
    const unsigned int minNumPairs = params.minNumPairs_;
    const float clusterMaxTransDist = params.clusterMaxTransDist_;
    const float clusterMaxRotDist = params.clusterMaxRotDist_;
    const unsigned int minNumSurfelPairs = params.minNumSurfelPairs_;

    const gsl_rng_type * T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    gsl_rng* rng = gsl_rng_alloc( T );
    gsl_rng_set( rng, time(NULL) );

    int maxOverallVotes = 0;

    poseVotesAllDepths.clear();

    float pairTime = 0.f;
    float matchingTime = 0.f;
    float poseTime = 0.f;

    float maxVotesAllSurfels = 0.f;

    source.reference_surfels_.resize( 17 );

    int maxDepth = source.octree_->max_depth_;

//    LOG_STREAM("Scene map max depth: " << source.octree_->max_depth_ );

    float minResolution = source.octree_->volumeSizeForDepth( maxDepth );
    float maxResolution = minResolution * 4.f;

    minResolution *= 0.99f;
    maxResolution *= 1.01f;

    // go through the octree layers in source from coarse to fine resolutions
    for ( int d = 0; d <= maxDepth; ++d ) {

        const float processResolution = source.octree_->volumeSizeForDepth( d );

        if ( processResolution < minResolution || processResolution > maxResolution )
            continue;

//        if ( d != 10)
//            continue;

//        LOG_STREAM( "processing resolution: " << processResolution );

        if ( target.reference_surfels_[d].size() == 0 ) {
//            LOG_STREAM( "No target ref surfels." );
            continue;
        }

        if ( target.all_surfel_pairs_[d].size() == 0 ) {
//            LOG_STREAM( "No surfel pairs in target." );
            continue;
        }

        PoseVector poseVotes;

        std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* > &
                srcNodes = source.samplingMap_[d];

        // lists of matching surfels for each key
//        const std::vector<std::vector<SurfelPair, Eigen::aligned_allocator<SurfelPair> > > & ref_surfel_pairs =
//                target.all_surfel_pairs_[d];
        const std::vector< MultiResolutionSurfelMap::Surfel* > & ref_surfels = target.reference_surfels_[d];
        // index into key_surfel_pairs from key
        const std::unordered_map< SurfelPairKey, std::vector<MultiResolutionSurfelMap::SurfelPair* > > & key_map = target.surfel_pair_list_map_[d];

        std::vector< MultiResolutionSurfelMap::Surfel* > & src_ref_surfels = source.reference_surfels_[d];

        if ( target.surfel_pair_list_map_[d].size() == 0 ) {
//            LOG_STREAM( "No keys" );
            continue;
        }

        const float anglePrecisionInv = 1.f / anglePrecision;
        const float angleStep = anglePrecision * M_PI / 180.;
        const float angleStepHalf = 0.5f * angleStep;

        const int numAngleBins = (int)(360 * anglePrecisionInv);

        unsigned int numTgtRefSurfelsAtDepth = ref_surfels.size();

//        LOG_STREAM("Number of tgt ref surfels: " << numTgtRefSurfelsAtDepth);

        std::vector<std::vector<float> > accumulator( numTgtRefSurfelsAtDepth, std::vector<float> ( numAngleBins, 0.f ) );
        std::vector<std::vector< std::vector< float > > > angleAccumulator( numTgtRefSurfelsAtDepth,
                                                                            std::vector<std::vector< float > > ( numAngleBins ));

//        float samplingRate = source.params_.surfelPairSamplingRate_;
//        float maxDist = target.surfelMaxDist_;
        float maxDist = source.params_.surfelPairMaxDistResFactor_ * processResolution;

//        samplingRate *= 1.f / maxDist;

//        float samplingRate = std::max( 0.25f, std::min( 1.f, ( 10.f * processResolution ) / 1.f ) );

        float targetNodesForDepth = 1.f / processResolution * 25;
        targetNodesForDepth += std::max( 0.f, 1.f / processResolution * 50 - numTgtRefSurfelsAtDepth );
        float samplingRate = std::max( 0.2f, std::min( source.params_.surfelPairSamplingRate_, targetNodesForDepth / (float)srcNodes.size() ) );

        if( source.params_.surfelPairSamplingRate_ > 2.f )
        	samplingRate = 1.f;

//        LOG_STREAM("Sampling rate: " << samplingRate );

        src_ref_surfels.reserve( samplingRate * srcNodes.size() );

        unsigned int numSrcRefSurfels = 0;
        unsigned int numSrcPairs = 0;
        unsigned int numSrcMatches = 0;

        float maxVotesDepth = 0.f;
        float maxBinVotes = 0.f;

        int minNumVotesDepth = (int)( 1.f / ( 10.f * processResolution ) * minNumVotes );

        Eigen::Vector4f maxDistVec( maxDist, maxDist, maxDist, 0.f );

        for ( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* >::iterator srcNodeIt =
              srcNodes.begin(); srcNodeIt != srcNodes.end(); ++srcNodeIt ) {

            if ( gsl_rng_uniform( source.r ) < ( 1.f - samplingRate ) )
                continue;

//            timing::Timer pairTimer;

            float surfelMaxVotes = 0.f;

            spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue >* currentNode = *srcNodeIt;

            std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionSurfelMap::NodeValue>* > nodes;
            nodes.reserve( 100 );
            Eigen::Vector4f minPosition = currentNode->getCenterPosition() - maxDistVec;
            Eigen::Vector4f maxPosition = currentNode->getCenterPosition() + maxDistVec;

            source.octree_->getAllNodesInVolumeOnDepth( nodes, minPosition, maxPosition, d, false );

            const unsigned int currentSurfelIdx = 4;

            if( currentNode->value_.surfels_[currentSurfelIdx].num_points_ < MultiResolutionSurfelMap::Surfel::min_points_ )
                continue;

            MultiResolutionSurfelMap::Surfel* srcRefSurfel = &(currentNode->value_.surfels_[currentSurfelIdx]);

            if ( boost::math::isnan( srcRefSurfel->normal_(0) ) ||
                 boost::math::isnan( srcRefSurfel->normal_(1) ) ||
                 boost::math::isnan( srcRefSurfel->normal_(2) ) )
                continue;


            unsigned int refIdx = src_ref_surfels.size();
            src_ref_surfels.push_back( srcRefSurfel );
            srcRefSurfel->ref_idx_ = refIdx;

            numSrcRefSurfels++;

            std::vector< MultiResolutionSurfelMap::SurfelPair, Eigen::aligned_allocator< MultiResolutionSurfelMap::SurfelPair > > pairs;

            float dist = 0.f;
            source.buildSurfelPairsForSurfel( currentNode, srcRefSurfel, currentSurfelIdx,
                                              nodes, pairs, dist );

//            pairTime += pairTimer.check();

            numSrcPairs += pairs.size();

//            timing::Timer matchingTimer;

            unsigned int numMatches = 0;

            for ( auto  srcPairIt = pairs.begin(); srcPairIt != pairs.end(); ++srcPairIt ) {
                MultiResolutionSurfelMap::SurfelPair & srcPair = *srcPairIt;

                SurfelPairKey key = srcPair.signature_.getKey( target.surfelMaxDist_, target.params_.surfelPairFeatureBinDist_ * processResolution, target.params_.surfelPairFeatureBinAngle_, target.params_.surfelPairUseColor_ );

                //                LOG_STREAM("Key: " << key);

                // look up matching pairs in the target
                auto got =
                        key_map.find( key );

                if ( got == key_map.end() )
                    continue;

                // get all the pairs in the target for this key
                const std::vector<MultiResolutionSurfelMap::SurfelPair* > & matchPairs =
                        got->second;

                numMatches += matchPairs.size();
                float normalizer = 1000.f / (float)matchPairs.size();

                // iterate over the matching pairs
                for ( unsigned int j = 0; j < matchPairs.size(); ++j ) {


                    const MultiResolutionSurfelMap::SurfelPair& dstPair = *(matchPairs[j]);
                    const int tgtSurfelRefIdx = dstPair.src_->ref_idx_;

                    float featureDist = 1.f;

                    if ( params.useFeatures_ ) {
                        featureDist = srcPair.src_->agglomerated_shape_texture_features_.distance(
                                    dstPair.src_->agglomerated_shape_texture_features_ );

                        if ( featureDist > params.maxFeatureDist_ )
                            continue;
                    }


                    // compute alpha angle which aligns these pairs
                    const float alpha = Geometry::twoPiCut(srcPair.signature_.alpha_ - dstPair.signature_.alpha_);
                    const float alpha_2pi = alpha;
                    float alpha2 = alpha;

                    // bin the angle...
                    const float alphaBinned = alpha_2pi / angleStep;
                    int alphaBin1 = (int) alphaBinned;
                    if ( alphaBin1 >= numAngleBins )
                        alphaBin1 = numAngleBins - 1;

                    int alphaBin2 = alphaBin1 + 1;
                    bool bin2up = true;

                    // fractional binning - weight 2 nearest bins
                    float bin1diff = fabsf( alphaBinned - (float)alphaBin1 );
                    float bin2diff = fabsf( (float)alphaBin2 - alphaBinned );

                    if ( bin2up && alphaBin2 >= numAngleBins ) {
                        alphaBin2 = 0;
                        alpha2 -= M_2PI;
                    }
                    else if ( alphaBin2 < 0  ) {
                        alphaBin2 = numAngleBins-1;
                        alpha2 += M_2PI;
                    }

                    const float alphaBin1Weight = 1.f + ( 1.f - bin1diff );
                    const float alphaBin2Weight = 1.f + ( 1.f - bin2diff );

                    accumulator[tgtSurfelRefIdx][alphaBin1] += alphaBin1Weight * normalizer;
                    accumulator[tgtSurfelRefIdx][alphaBin2] += alphaBin2Weight * normalizer;

                    angleAccumulator[ tgtSurfelRefIdx ][alphaBin1].push_back( alpha );

                    angleAccumulator[ tgtSurfelRefIdx ][alphaBin2].push_back( alpha2 );

                    if ( accumulator[tgtSurfelRefIdx][alphaBin1] > surfelMaxVotes )
                        surfelMaxVotes = accumulator[tgtSurfelRefIdx][alphaBin1];
                    if ( accumulator[tgtSurfelRefIdx][alphaBin2] > surfelMaxVotes )
                        surfelMaxVotes = accumulator[tgtSurfelRefIdx][alphaBin2];

                } // matching surfel pairs loop
            } // src surfel pair loop

            numSrcMatches += numMatches;

            if ( surfelMaxVotes > maxVotesDepth )
                maxVotesDepth = surfelMaxVotes;

//            matchingTime += matchingTimer.check();

            if ( surfelMaxVotes == 0 ) {
                continue;
            }

            bool srcPoseSet = false;
            Eigen::Matrix4d srcRefPose = Eigen::Matrix4d::Identity();

//            timing::Timer poseTimer;


            int numPoses = 0;

            // loop around the accumulator to find the poses with most support for this surfel
            for ( auto targetSurfelIt = accumulator.begin();
                  targetSurfelIt != accumulator.end(); ++targetSurfelIt) {

                if ( targetSurfelIt->begin() == targetSurfelIt->end() ) {
//                    LOG_STREAM( "blah" );
                    continue;
                }

                bool dstPoseSet = false;
                Eigen::Matrix4d dstRefPose = Eigen::Matrix4d::Identity();

                const int tgtSurfelRefIdx = targetSurfelIt - accumulator.begin();
                MultiResolutionSurfelMap::Surfel* targetSurfel = ref_surfels[tgtSurfelRefIdx];

                std::vector<float> & votes = *targetSurfelIt;

                // loop around the angles to find the most supported for this target ref surfel
                for ( unsigned int j = 0; j < votes.size(); ++j ) {
                    // enough votes?
                    if ( ( votes[j] >= minNumVotesDepth ) && ( votes[j] > ( surfelSimilarVoteFactor * surfelMaxVotes ) ) ) {

                        if ( !srcPoseSet ) {
                            srcRefPose.block<3, 1>( 0, 3 ) = srcRefSurfel->reference_pose_.block< 3, 1 >( 0, 0 );
                            srcRefPose.block<3, 3>( 0, 0 ) = Eigen::Matrix3d(
                                        Eigen::Quaterniond( srcRefSurfel->reference_pose_( 6, 0 ), srcRefSurfel->reference_pose_( 3, 0 ),
                                                            srcRefSurfel->reference_pose_( 4, 0 ), srcRefSurfel->reference_pose_( 5, 0 ) ) );
                            // invert the transformation inv(T(s->g))
                            srcRefPose.block<3, 3>( 0, 0 ) = srcRefPose.block<3, 3>( 0, 0 ).transpose().eval();
                            srcRefPose.block<3, 1>( 0, 3 ) = -( srcRefPose.block<3, 3>( 0, 0 )
                                                                * srcRefPose.block<3, 1>( 0, 3 ) ).eval();

                            srcPoseSet = true;
                        }

                        if ( !dstPoseSet ) {
                            dstRefPose.block<3, 1>( 0, 3 ) = targetSurfel->reference_pose_.block<3,1>( 0, 0 );
                            dstRefPose.block<3, 3>( 0, 0 ) = Eigen::Matrix3d(
                                        Eigen::Quaterniond( targetSurfel->reference_pose_( 6, 0 ),
                                                            targetSurfel->reference_pose_( 3, 0 ),
                                                            targetSurfel->reference_pose_( 4, 0 ),
                                                            targetSurfel->reference_pose_( 5, 0 ) ) );
                            dstPoseSet = true;
                        }

                        // get the median angle for this bin
                        double angle = (double) computeMedian( angleAccumulator[tgtSurfelRefIdx][j] );

                        // construct the rotation around x axis...
                        Eigen::Matrix4d rotX = Eigen::Matrix4d::Identity();
                        rotX.block<3,3>( 0,0 ) = Eigen::Matrix3d( Eigen::AngleAxisd( angle, Eigen::Vector3d::UnitX() ) );

                        // compute the full pose
                        Eigen::Matrix4d poseTransform = dstRefPose * rotX * srcRefPose;
                        Eigen::Matrix3d rotation = poseTransform.block<3,3>(0,0);

                        Eigen::Vector3d dir_match_src = rotation * srcRefSurfel->initial_view_dir_;

                        bool poseValid = true;

                        const double dist = dir_match_src.dot( targetSurfel->initial_view_dir_ );
                        if( dist < SURFEL_MATCH_ANGLE_THRESHOLD ) {
                            poseValid = false;
                        }

                        if (  poseValid ) {
                            Eigen::Matrix<double, 8, 1> pose;
                            pose.block<3, 1>( 0, 0 ) = poseTransform.block<3, 1>( 0, 3 );
                            Eigen::Quaterniond q( Eigen::Matrix3d( poseTransform.block<3, 3>( 0, 0 ) ) );

                            if ( q.w() < 0 ) {
                                q = q.coeffs() * -1;
                            }

                            pose( 3, 0 ) = q.x();
                            pose( 4, 0 ) = q.y();
                            pose( 5, 0 ) = q.z();
                            pose( 6, 0 ) = q.w();
                            pose( 7, 0 ) = votes[j];
                            poseVotes.push_back( pose );

                            numPoses++;

                            if ( votes[j] > maxOverallVotes )
                                maxOverallVotes = votes[j];
                        }
                    }

                    // reset for the next surfel
                    votes[j] = 0;
                    angleAccumulator[ tgtSurfelRefIdx ][j].clear();

                } // angle accumulator loop

            } // target surfel accumulator loop

            if ( numPoses > 0 )
                associatedRefSurfels.push_back( srcRefSurfel );

//            poseTime += poseTimer.check();
        } // src surfel loop

//        LOG_STREAM( "Num src ref surfels: " << numSrcRefSurfels );
//        LOG_STREAM( "Num src pairs: " << numSrcPairs );
//        LOG_STREAM( "Num src matches: " << numSrcMatches );
//
//        LOG_STREAM( "Max votes at depth: " << maxVotesDepth );
        if ( maxVotesDepth > maxVotesAllSurfels )
            maxVotesAllSurfels = maxVotesDepth;

        if( poseVotes.size() > 0 ) {

            std::sort( poseVotes.begin(), poseVotes.end(),
                       []( const Eigen::Matrix<double, 8, 1> & x, const Eigen::Matrix<double, 8, 1> & y ) { return x(7) > y(7); } );

            float voteThreshold = allSurfelsSimilarVoteFactor * maxVotesDepth;
//            LOG_STREAM( "Threshold: " << voteThreshold );

            PoseVector::iterator thresh = poseVotes.begin();
            while ( thresh != poseVotes.end() ) {
                if ( (*thresh)(7) >= voteThreshold ) {
                    ++thresh;
                }
                else
                    break;
            }


//            LOG_STREAM( "Poses added: " << thresh - poseVotes.begin() - 1 );
            poseVotesAllDepths.insert( poseVotesAllDepths.end(), poseVotes.begin(), thresh);

        }
    }

//    LOG_STREAM( "Computing pairs took [" << pairTime << "]" );
//    LOG_STREAM( "Matching pairs took ["<< matchingTime << "]" );
//    LOG_STREAM( "Finding poses took ["<< poseTime  << "]" );
//
//    LOG_STREAM( "Number of poses: " << poseVotesAllDepths.size() );

    // TODO: Mean-shift
//    float voteThreshold2 = allSurfelsSimilarVoteFactor * maxVotesAllSurfels;
//
//    LOG_STREAM( "vote threshold2: " << voteThreshold2 );

    unsigned int maxNumClusters = params.maxNumClusters_;


    if ( poseVotesAllDepths.size() == 0 )
        return false;

    std::sort( poseVotesAllDepths.begin(), poseVotesAllDepths.end(),
               []( const Eigen::Matrix<double, 8, 1> & x, const Eigen::Matrix<double, 8, 1> & y ) { return x(7) > y(7); } );

    std::vector<PoseWithScore, Eigen::aligned_allocator<PoseWithScore> > poses;
    poses.reserve( poseVotesAllDepths.size() );

    for ( auto poseIt = poseVotesAllDepths.begin(); poseIt != poseVotesAllDepths.end(); ++poseIt ) {
        Eigen::Matrix<double, 7, 1> p = poseIt->block<7,1>(0,0);
        Geometry::Pose currentPose( p );
        double score = (*poseIt)(7);

//        if ( score >= voteThreshold2 )
            poses.push_back( PoseWithScore( currentPose, score ) );
    }

//    LOG_STREAM( "Poses going into clustering: " << poses.size() );

    PoseClustering clustering( &poses, clusterMaxTransDist, clusterMaxRotDist, map_mean );
    std::vector<pcl::PointIndices> indices;
    clustering.getClusters( indices );

//    LOG_STREAM( "Num clusters: " << indices.size() );

    if ( indices.size() == 0 )
        return false;


    for ( auto it = indices.begin(); it != indices.end(); ++it ) {
        if ( it->indices.size() == 0 )
            continue;

        Geometry::PoseCluster cluster;
        clustering.getCluster( *it, cluster );
        clusters.push_back( cluster );
    }


	std::sort( clusters.begin(), clusters.end(),
			   [](const Geometry::PoseCluster & x, const Geometry::PoseCluster & y){return x.score() > y.score(); });

	while ( clusters.size() > maxNumClusters ) {
		clusters.erase( clusters.end()-1 );
	}

//	LOG_STREAM( "Number of clusters: " << clusters.size() );
	for ( auto clusterIt = clusters.begin(); clusterIt != clusters.end(); ++clusterIt ) {
//	    LOG_STREAM( "Cluster, score :" << clusterIt->score() << "; centre: " << clusterIt->mean() );
	}


	return true;
}

}

