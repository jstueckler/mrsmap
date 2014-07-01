#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <math.h>

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <mrsmap/utilities/eigen_extensions.h>

namespace Geometry {

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

#ifndef M_3PI
#define M_3PI 9.424777960769379715387930149838509
#endif

static inline float piCut( const float & a ) {
    float angle = a;
    while( angle > M_PI )
            angle -= 2.f * M_PI;
    while( angle <= -M_PI )
            angle += 2.f * M_PI;
    return angle;
}

static inline float twoPiCut( const float & a ) {
    float angle = a;
    angle = piCut( angle );

    if ( angle < 0 ) {
        angle += (float)M_2PI;
     }
    return angle;
}


static inline float wrapTo2Pi( float angle ) {
    bool negative = angle < 0;
    angle = fmod(angle, (float)(2*M_PI) );
    if (negative)
        angle += (float)(2*M_PI);
    return angle;
}

class Pose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Pose() {
        position_ = Eigen::Vector3d::Zero();
        orientation_ = Eigen::Quaterniond::Identity();
    }

    Pose( Eigen::Vector7d pose ) {
        position_ = pose.head<3>();
        orientation_ = Eigen::Quaterniond( pose.tail<4>() );
    }

    Pose( Eigen::Vector3d position, Eigen::Quaterniond orientation ) {
        position_ = position;
        orientation_ = orientation;
    }

    Pose( Eigen::Matrix4d transform ) {
        position_ = transform.block<3, 1>( 0, 3 );
        orientation_ = Eigen::Quaterniond( transform.block<3, 3>( 0, 0 ) );
    }

    ~Pose() {}

    inline Eigen::Vector7d asVector() const {
        Eigen::Vector7d pose;
        pose.head<3>() = position_;
        pose.tail<4>() = orientation_.coeffs();
        return pose;
    }

    inline Eigen::Matrix4d asMatrix4d() const {
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3, 1>( 0, 3 ) = position_;
        transform.block<3, 3>( 0, 0 ) = Eigen::Matrix3d( orientation_ );
        return transform;
    }

    friend std::ostream& operator<< (std::ostream& stream, const Pose& pose) {
        stream << pose.position_ << "; " << pose.orientation_.coeffs();
        return stream;
    }

    Eigen::Vector3d position_;
    Eigen::Quaterniond orientation_;
};

class PoseWithCovariance {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseWithCovariance() {
        mean_ = Pose();
        covariance_ = Eigen::Matrix<double, 6, 6>::Identity();
    }

    PoseWithCovariance( Pose & mean, Eigen::Matrix<double, 6, 6> & covariance ) {
        mean_ = mean;
        covariance_ = covariance;
    }

    ~PoseWithCovariance() {}

    friend std::ostream& operator<< (std::ostream& stream, const PoseWithCovariance& pose) {
        stream << pose.mean_ << "\n " << pose.covariance_;
        return stream;
    }

    Pose mean_;
    Eigen::Matrix<double, 6, 6> covariance_;
};

class PoseAndVelocity {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseAndVelocity() {
        velocity_ = Eigen::Vector6d::Zero();
        delta_t_ = 1.0;
    }

    PoseAndVelocity( Eigen::Vector3d position, Eigen::Quaterniond orientation,
                     Eigen::Vector6d velocity ) {
        pose_.position_ = position;
        pose_.orientation_ = orientation;
        velocity_ = velocity;
        delta_t_ = 1.0;
    }

    ~PoseAndVelocity() {}

    friend std::ostream& operator<< (std::ostream& stream, const PoseAndVelocity& pose) {
        stream << pose.pose_ << "; " << pose.velocity_;
        return stream;
    }

    PoseWithCovariance poseWithCovariance_;
    Pose pose_;    
    Pose pose_tminus2_;
    Pose prevPose_;
    Eigen::Vector6d velocity_;
    double delta_t_;
};

// computes corrected mean transform
static inline Pose correctMeanPose( Eigen::Vector3d & translation, Eigen::Matrix3d & rotation ) {
    Pose mean;
    Eigen::Matrix3d rotationTranspose = rotation.transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d, 2> svd = rotationTranspose.jacobiSvd( Eigen::ComputeFullU | Eigen::ComputeFullV );
    Eigen::Matrix3d u = svd.matrixU();
    Eigen::Matrix3d v = svd.matrixV();

    if ( rotation.transpose().determinant() > 0.f ) {
        rotation = v * u.transpose().eval();
    } else {
        rotation = v * Eigen::DiagonalMatrix<double, 3>( 1.f, 1.f, -1.f ) * u.transpose().eval();
    }

    mean.position_ = translation;
    mean.orientation_ = Eigen::Quaterniond( rotation );

    return mean;
}

template <typename T_state>
class ClusterItem {
public:

    ClusterItem ( ) : score_( 0.f ) {}
    ClusterItem( T_state state, float score = 0.f ) : state_( state ), score_( score ) {}

    T_state state_;
    float score_;
};

template < typename T_state >
class Cluster {
public:
    typedef typename std::vector< ClusterItem< T_state > > ClusterItemsList;

    virtual ~Cluster()  {}

    virtual T_state & updateMean() {
        return mean_;
    }

    void updateScore() {
        float score = 0;
        for( typename std::vector< ClusterItem< T_state > >::iterator it =
             members_.begin(); it != members_.end(); ++it ) {
            score += it->score_;
        }

        score_ = score;
    }

    Cluster() : score_( 0.f ) {}
    Cluster( T_state state, float score = 0.f ) {
        mean_ = state;
        score_ = score;
    }


    virtual void add( T_state state, float score = 0.f ) {
        members_.push_back( ClusterItem<T_state>( state, score ) );
//        updateMean();
        score_ += score;
//        updateScore();
    }

    virtual void merge( Cluster<T_state>& other ) {
        members_.insert( members_.end(), other.members_.begin(), other.members_.end() );
        updateMean();
        updateScore();
    }

    virtual const ClusterItemsList & members() const { return members_; }

    virtual const T_state & mean() const { return mean_; }

    virtual unsigned int size() const { return members_.size(); }

    virtual float score() const { return score_ > 0.f ? score_ : size(); }


    T_state mean_;
    float score_;
protected:
    ClusterItemsList members_;
};

class PoseCluster : public Cluster< Pose > {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    

    PoseCluster() {
        score_ = 0.f;
    }

    PoseCluster( Pose state, float score = 0.f ) {
        mean_ = state;
        score_ = score;
    }

    virtual ~PoseCluster()  {}


    virtual Pose & updateMean() {
        Eigen::Vector3d translation = Eigen::Vector3d::Zero();
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Zero();

        for ( ClusterItemsList::iterator currentItem =
              members_.begin(); currentItem != members_.end(); ++currentItem ) {
            translation += currentItem->state_.position_;

            rotation += Eigen::Matrix3d( currentItem->state_.orientation_ );
        }

        translation /= members_.size();
        rotation /= members_.size();

        mean_ = correctMeanPose( translation, rotation );
        return mean_;
    }


    double numAssocs;
    double score_match_;
    double score_min_[16];
    double score_max_[16];
    double score_sum_[16];

};

class PoseWithVelocityCluster : public Cluster< PoseAndVelocity > {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseWithVelocityCluster( PoseAndVelocity state, float score = 0.f ) {
        mean_ = state;
        score_ = score;
    }

    virtual ~PoseWithVelocityCluster()  {}

    virtual PoseAndVelocity & updateMean() {
        Eigen::Vector3d translation = Eigen::Vector3d::Zero();
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Zero();
        Eigen::Vector6d velocity = Eigen::Vector6d::Zero();

        for ( ClusterItemsList::iterator currentItem =
              members_.begin(); currentItem != members_.end(); ++currentItem ) {
            translation += currentItem->state_.pose_.position_;

            rotation += Eigen::Matrix3d( currentItem->state_.pose_.orientation_ );
            velocity += currentItem->state_.velocity_;
        }

        translation /= members_.size();
        rotation /= members_.size();
        velocity /= members_.size();

        mean_.pose_ = correctMeanPose( translation, rotation );
        mean_.velocity_ = velocity;
        return mean_;
    }
};

struct ClusterMemberFunctorParams {
};

template <typename T_state>
class ClusterMemberFunctor {
public:
    ClusterMemberFunctor() {}
};

struct PoseClusterMemberFunctorParams : ClusterMemberFunctorParams {
    PoseClusterMemberFunctorParams( float maxTransDist = 0.f, float maxRotDist = 0.f)
        : maxTransDist_( maxTransDist ), maxRotDist_( maxRotDist ) {}
public:
    float maxTransDist_;
    float maxRotDist_;
};

struct PoseWithVelocityClusterMemberFunctorParams : public PoseClusterMemberFunctorParams {
public:
    float maxTransVel_;
    float maxRotVel_;
};

class PoseClusterMemberFunctor : public ClusterMemberFunctor<Pose> {
public:
    PoseClusterMemberFunctor() : cluster_( 0 ) {}

    PoseClusterMemberFunctor( PoseCluster* cluster, const PoseClusterMemberFunctorParams & params =
            PoseClusterMemberFunctorParams() ) :
        cluster_( cluster ), params_( params ) {}

    bool operator()(const Pose & state, float & transDist, float & rotDist ) {
        float bestTransDist = std::numeric_limits<float>::max();
        float bestRotDist = std::numeric_limits<float>::max();
        for ( auto it : cluster_->members() ) {

            Eigen::Matrix4d delta = state.asMatrix4d() * it.state_.asMatrix4d().inverse();
            transDist = delta.block<3,1>( 0, 3 ).norm();
            rotDist = fabs(Geometry::piCut( Eigen::AngleAxisd( delta.block<3,3>( 0,0 ) ).angle() ) );

            bestTransDist = std::min( bestTransDist, transDist );
            bestRotDist = std::min( bestRotDist, rotDist );
        }

        return ( bestTransDist < params_.maxTransDist_ && bestRotDist < params_.maxRotDist_ );
    }

    PoseCluster* cluster_;
    PoseClusterMemberFunctorParams params_;
};

class PoseWithVelocityClusterMemberFunctor : public ClusterMemberFunctor<PoseAndVelocity> {
public:
    PoseWithVelocityClusterMemberFunctor() : cluster_( 0 ) {}

    PoseWithVelocityClusterMemberFunctor( PoseWithVelocityCluster* cluster, const PoseWithVelocityClusterMemberFunctorParams & params =
            PoseWithVelocityClusterMemberFunctorParams() ) :
        cluster_( cluster ), params_( params ) {}

    bool operator()(const PoseAndVelocity & state, float & transDist, float & rotDist, float & transVelDist, float & rotVelDist ) {
        Eigen::Matrix4d delta = state.pose_.asMatrix4d() * cluster_->mean().pose_.asMatrix4d().inverse();
        transDist = delta.block<3,1>( 0, 3 ).norm();
        rotDist = fabs(Geometry::piCut( Eigen::AngleAxisd( delta.block<3,3>( 0,0 ) ).angle() ) );

        transVelDist = ( state.velocity_.head(3) - cluster_->mean().velocity_.head(3) ).norm();
        rotVelDist = ( state.velocity_.tail(3) - cluster_->mean().velocity_.tail(3) ).norm();

        return ( transDist < params_.maxTransDist_ && rotDist < params_.maxRotDist_ && transVelDist < params_.maxTransVel_
                 && rotVelDist < params_.maxRotVel_);
    }

    PoseWithVelocityCluster* cluster_;
    PoseWithVelocityClusterMemberFunctorParams params_;
};

static inline void poseFirstDerivatives( const Pose&  pose, Eigen::Matrix4d & dt_tx, Eigen::Matrix4d & dt_ty,
    Eigen::Matrix4d & dt_tz, Eigen::Matrix4d & dR_qx, Eigen::Matrix4d & dR_qy, Eigen::Matrix4d & dR_qz ) {
    const double qx = pose.orientation_.x();
    const double qy = pose.orientation_.y();
    const double qz = pose.orientation_.z();
    const double qw = pose.orientation_.w();
    const double inv_qw = 1.0 / qw;
    dt_tx.setZero();
    dt_ty.setZero();
    dt_tz.setZero();
    dR_qx.setZero();
    dR_qy.setZero();
    dR_qz.setZero();

    dt_tx(0,3) = 1.f;
    dt_ty(1,3) = 1.f;
    dt_tz(2,3) = 1.f;

    dR_qx( 0, 0 ) = 0.0;
    dR_qx( 0, 1 ) = 2.0 * ( ( qx * qz ) * inv_qw + qy );
    dR_qx( 0, 2 ) = 2.0 * ( qz - ( qx * qy ) * inv_qw );
    dR_qx( 1, 0 ) = 2.0 * ( qy - ( qx * qz ) * inv_qw );
    dR_qx( 1, 1 ) = -4.0 * qx;
    dR_qx( 1, 2 ) = 2.0 * ( qx * qx * inv_qw - qw );
    dR_qx( 2, 0 ) = 2.0 * ( ( qx * qy ) * inv_qw + qz );
    dR_qx( 2, 1 ) = 2.0 * ( qw - qx * qx * inv_qw );
    dR_qx( 2, 2 ) = -4.0 * qx;

    dR_qy( 0, 0 ) = -4.0 * qy;
    dR_qy( 0, 1 ) = 2.0 * ( ( qy * qz ) * inv_qw + qx );
    dR_qy( 0, 2 ) = 2.0 * ( qw - qy * qy * inv_qw );
    dR_qy( 1, 0 ) = 2.0 * ( qx - ( qy * qz ) * inv_qw );
    dR_qy( 1, 1 ) = 0.0;
    dR_qy( 1, 2 ) = 2.0 * ( ( qx * qy ) * inv_qw + qz );
    dR_qy( 2, 0 ) = 2.0 * ( qy * qy * inv_qw - qw );
    dR_qy( 2, 1 ) = 2.0 * ( qz - ( qx * qy ) * inv_qw );
    dR_qy( 2, 2 ) = -4.0 * qy;

    dR_qz( 0, 0 ) = -4.0 * qz;
    dR_qz( 0, 1 ) = 2.0 * ( qz * qz * inv_qw - qw );
    dR_qz( 0, 2 ) = 2.0 * ( qx - ( qy * qz ) * inv_qw );
    dR_qz( 1, 0 ) = 2.0 * ( qw - qz * qz * inv_qw );
    dR_qz( 1, 1 ) = -4.0 * qz;
    dR_qz( 1, 2 ) = 2.0 * ( ( qx * qz ) * inv_qw + qy );
    dR_qz( 2, 0 ) = 2.0 * ( ( qy * qz ) * inv_qw + qx );
    dR_qz( 2, 1 ) = 2.0 * ( qy - ( qx * qz ) * inv_qw );
    dR_qz( 2, 2 ) = 0.0;
}

// see "Iterative Estimation of Rotation and Translation using the Quaternion"
static inline void jacobianWrtPose( const Pose & pose, const Eigen::Vector4d & p, Eigen::Matrix<double, 3, 6> & J ) {
    Eigen::Matrix3d rotation( pose.orientation_ );
    Eigen::Vector3d p_rot = rotation * p.head(3);
    Eigen::Matrix3d p_rot_ss = Eigen::skewSymmetricMatrix( p_rot );

    J.block<3,3>( 0,0 ) = p_rot_ss.transpose();
    J.block<3,3>( 0,3 ) = Eigen::Matrix3d::Identity();
//    Eigen::Matrix4d dt_tx, dt_ty, dt_tz;
//    Eigen::Matrix4d dR_qx, dR_qy, dR_qz;
//    J.setZero();

//    poseFirstDerivatives( pose, dt_tx, dt_ty, dt_tz, dR_qx, dR_qy, dR_qz );

//    J.block<4,1>(0,0) = dt_tx * p;
//    J.block<4,1>(0,1) = dt_ty * p;
//    J.block<4,1>(0,2) = dt_tz * p;
//    J.block<4,1>(0,3) = dR_qx * p;
//    J.block<4,1>(0,4) = dR_qy * p;
//    J.block<4,1>(0,5) = dR_qz * p;
}

  class Quaternion {
  public:
//    // from: http://planning.cs.uiuc.edu/node198.html
//    static Eigen::Quaterniond Random( gsl_rng* rng ) {
//      Eigen::Vector3d unit( gsl_rng_uniform(rng), gsl_rng_uniform(rng), gsl_rng_uniform(rng) );
//      double rootu1 = sqrt(unit[0]);
//      double root1minusu1 = sqrt(1-unit[0]);
//      double twopiu2 = 2.0 * M_PI * unit[1];
//      double twopiu3 = 2.0 * M_PI * unit[2];
//
//      Eigen::Quaterniond random(root1minusu1*sin(twopiu2), root1minusu1*cos(twopiu2),
//                         rootu1 * sin(twopiu3), rootu1 * cos(twopiu3) );
//      return random;
//
//    }

    static Eigen::Quaterniond FromEulerAngles(double theta_z, double theta_y, double theta_x) {
      Eigen::Quaterniond q;
      double cos_z_div2 = cos(0.5f*theta_z);
      double cos_y_div2 = cos(0.5f*theta_y);
      double cos_x_div2 = cos(0.5f*theta_x);

      double sin_z_div2 = sin(0.5f*theta_z);
      double sin_y_div2 = sin(0.5f*theta_y);
      double sin_x_div2 = sin(0.5f*theta_x);

      q.w() = cos_z_div2*cos_y_div2*cos_x_div2 + sin_z_div2*sin_y_div2*sin_x_div2;
      q.x() = cos_z_div2*cos_y_div2*sin_x_div2 - sin_z_div2*sin_y_div2*cos_x_div2;
      q.y() = cos_z_div2*sin_y_div2*cos_x_div2 + sin_z_div2*cos_y_div2*sin_x_div2;
      q.z() = sin_z_div2*cos_y_div2*cos_x_div2 - cos_z_div2*sin_y_div2*sin_x_div2;

//      std::cout << "Rotation: " << q.coeffs();
      q.normalize();
      return q;
    }

    static Eigen::Quaterniond FromYawPitchRoll(double yaw, double pitch, double roll) {
        // roll axis is z
        Eigen::Quaterniond qroll( cos(0.5*roll), 0, 0, sin(0.5*roll) );
        // pitch is x
        Eigen::Quaterniond qpitch( cos(0.5*pitch), sin(0.5*pitch), 0, 0);
        // yaw is y
        Eigen::Quaterniond qyaw( cos(0.5*yaw), 0, sin(0.5*yaw), 0);

        Eigen::Quaterniond q = qyaw * ( qpitch * qroll );
        return q;
    }

    static Eigen::Quaterniond FromEulerAngles(Eigen::Vector3d eulerzyx) {
      return FromEulerAngles(eulerzyx[0], eulerzyx[1], eulerzyx[2]);
    }

    static Eigen::Quaterniond FromYawPitchRoll(Eigen::Vector3d ypr) {
      return FromYawPitchRoll( ypr[0], ypr[1], ypr[2] );
    }
  };

  class Conversion {
  public:
    // takes a 7-dimensional vector consisting of an XYZ pose and a quaternion representing the rotation
    static inline void poseToTransform(const Eigen::Matrix<double, 7, 1> & pose, Eigen::Matrix4d & transform) {
        transform.setIdentity();
        transform.block<3, 1>( 0, 3 ) = pose.block<3, 1>( 0, 0 );
        transform.block<3, 3>( 0, 0 ) = Eigen::Matrix3d(
                    Eigen::Quaterniond( pose( 6, 0 ), pose( 3, 0 ), pose( 4, 0 ), pose( 5, 0 ) ) );
    }

    static inline void poseToTransform(const Eigen::Matrix<double, 3, 1> & pose, const Eigen::Quaterniond & orientation,
                                       Eigen::Matrix4d & transform ) {
        transform.setIdentity();
        transform.block<3, 1>( 0, 3 ) = pose;
        transform.block<3, 3>( 0, 0 ) = Eigen::Matrix3d( orientation );
    }

    static inline void transformToPose( const Eigen::Matrix4d & transform, Eigen::Matrix<double, 7, 1> & pose ) {
        pose.block<3, 1>( 0, 0 ) = transform.block<3, 1>( 0, 3 );
        pose.block<4, 1>( 3, 0 ) = Eigen::Quaterniond( transform.block<3, 3>( 0, 0 ) ).coeffs();
    }

    static inline void transformToPose( const Eigen::Matrix4d & transform, Eigen::Matrix<double, 3, 1> & pose,
                                        Eigen::Quaterniond & orientation) {
        pose = transform.block<3, 1>( 0, 3 );
        orientation = Eigen::Quaterniond( transform.block<3, 3>( 0, 0 ) );
    }

    static void poseToTransform(const Eigen::Matrix<double, 7, 1> & pose, Eigen::Affine3d & transform) {
      Eigen::Quaterniond rotation( pose.block<4, 1>(3, 0) );
      Eigen::Translation3d translation ( pose.block<3, 1>(0, 0) );
      transform = translation * rotation;
    }

    static void poseToTransform( const Eigen::Matrix<double, 3, 1> & pose, const Eigen::Quaterniond & rotation, Eigen::Affine3d & transform ) {
//      Eigen::Quaterniond rotation( pose.block<4, 1>(3, 0) );
      Eigen::Translation3d translation ( pose.block<3, 1>(0, 0) );
      transform = translation * rotation;
    }
  };
}

#endif
