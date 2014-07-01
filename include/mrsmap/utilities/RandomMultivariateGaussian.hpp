/* Adapted from http://lost-found-wandering.blogspot.de/2011/05/sampling-from-multivariate-normal-in-c.html */

#ifndef __RANDOM_MULTIVARIATEGAUSSIAN_HPP
#define __RANDOM_MULTIVARIATEGAUSSIAN_HPP

#include <Eigen/Cholesky>
#include <gsl/gsl_rng.h>
#include <Eigen/Core>
#include <math.h>

namespace random_numbers {

template<typename Scalar, int size>
class RandomMultivariateGaussian {
public:
    RandomMultivariateGaussian( ) {
        init = true;
        ready = false;
    }

    RandomMultivariateGaussian( gsl_rng* rng ) {
        m_rng = rng;
        init = true;
        ready = false;
    }

    void setRng( gsl_rng* rng ) {
        m_rng = rng;
        init = true;
    }

    void update( const Eigen::Matrix<Scalar, size, 1> & mean, const Eigen::Matrix<Scalar, size, size > & covariance ) {
        covariance_ = covariance;
        mean_ = mean;

        llt_.compute( covariance );
        l_ = llt_.matrixL();
        ready = true;
    }

    Eigen::Matrix<Scalar, size, 1> getSample() {
        assert( init && ready );

        Eigen::Matrix<Scalar, size, 1> sample;
        for ( int i=0; i<size; ++i ) {
            sample(i) = gsl_ran_gaussian( m_rng, 1.0 );
        }

        return propagate( sample );
    }

    Eigen::Matrix<Scalar, size, 1> propagate( const Eigen::Matrix<Scalar, size, 1>& sample ) {
        assert( init && ready );

        return mean_ + l_ * sample;
    }

protected:
    bool init, ready;
    Eigen::Matrix<Scalar, size, size > covariance_;
    Eigen::Matrix<Scalar, size, 1> mean_;
    Eigen::Matrix<Scalar, size, size > l_;
    Eigen::LLT<Eigen::Matrix<Scalar, size, size> > llt_;
    gsl_rng* m_rng;
};

}


//#include <Eigen/Dense>

//#include <math.h>

//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/normal_distribution.hpp>
//#include <boost/random/variate_generator.hpp>

///**
//    We find the eigen-decomposition of the covariance matrix.
//    We create a vector of normal samples scaled by the eigenvalues.
//    We rotate the vector by the eigenvectors.
//    We add the mean.
//*/
//template<typename _Scalar, int _size>
//class EigenMultivariateNormal
//{
//    boost::mt19937 rng;    // The uniform pseudo-random algorithm
//    boost::normal_distribution<_Scalar> norm;  // The gaussian combinator
//    boost::variate_generator<boost::mt19937&,boost::normal_distribution<_Scalar> >
//    randN; // The 0-mean unit-variance normal generator

//    Eigen::Matrix<_Scalar,_size,_size> rot;
//    Eigen::Matrix<_Scalar,_size,1> scl;

//    Eigen::Matrix<_Scalar,_size,1> mean;

//public:
//    EigenMultivariateNormal() : randN( rng, norm ) { }

//      EigenMultivariateNormal(const Eigen::Matrix<_Scalar,_size,1>& meanVec,
//        const Eigen::Matrix<_Scalar,_size,_size>& covarMat)
//      : randN(rng,norm)
//    {
//        setCovar(covarMat);
//        setMean(meanVec);
//    }

//    void setCovar(const Eigen::Matrix<_Scalar,_size,_size>& covarMat)
//    {
//        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<_Scalar,_size,_size> >
//                eigenSolver(covarMat);
//        rot = eigenSolver.eigenvectors();
//        scl = eigenSolver.eigenvalues();
//        for (int ii=0;ii<_size;++ii) {
//            scl(ii,0) = sqrt(scl(ii,0));
//        }
//    }

//    void setMean(const Eigen::Matrix<_Scalar,_size,1>& meanVec)
//    {
//        mean = meanVec;
//    }

//    void nextSample(Eigen::Matrix<_Scalar,_size,1>& sampleVec)
//    {
//        for (int ii=0;ii<_size;++ii) {
//            sampleVec(ii,0) = randN() * scl(ii,0);
//        }

//        sampleVec = rot*sampleVec + mean;
//    }

#endif
