#ifndef EIGEN_EXTENSIONS_H_
#define EIGEN_EXTENSIONS_H_

#include <Eigen/Core>
#include <Eigen/Eigen>

#include <cmath>


namespace Eigen {
    typedef Matrix< float, 6, 1 > Vector6f;
    typedef Matrix< double, 6, 1 > Vector6d;
    typedef Matrix< float, 7, 1 > Vector7f;
    typedef Matrix< double, 7, 1 > Vector7d;
    typedef Matrix< float, 13, 1 > Vector13f;
    typedef Matrix< double, 13, 1 >	Vector13d;
    typedef Matrix<double, 6, 6> Matrix6d;
    typedef Matrix<float, 6, 6> Matrix6f;


// \note Method from TooN (C) Tom Drummond (GNU GPL)
inline void rodrigues_so3_exp(const Eigen::Vector3d & w, const double A, const double B, Eigen::Matrix3d & R)
{
    {
        const double wx2 = w(0)*w(0);
        const double wy2 = w(1)*w(1);
        const double wz2 = w(2)*w(2);
        R(0,0) = 1.0 - B*(wy2 + wz2);
        R(1,1) = 1.0 - B*(wx2 + wz2);
        R(2,2) = 1.0 - B*(wx2 + wy2);
    }
    {
        const double a = A*w(2);
        const double b = B*(w(0)*w(1));
        R(0,1) = b - a;
        R(1,0) = b + a;
    }
    {
        const double a = A*w(1);
        const double b = B*(w(0)*w(2));
        R(0,2) = b + a;
        R(2,0) = b - a;

    }
    {
        const double a = A*w(0);
        const double b = B*(w(1)*w(2));
        R(1,2) = b - a;
        R(2,1) = b + a;
    }
}

// Method from TooN (C) Tom Drummond (GNU GPL)
inline Eigen::Matrix3d exp_rotation(const Eigen::Vector3d & w)
{
      static const double one_6th = 1.0/6.0;
      static const double one_20th = 1.0/20.0;

      const double theta_sq = w.squaredNorm(); //w*w;
      const double theta = std::sqrt(theta_sq);
      double A, B;
      //Use a Taylor series expansion near zero. This is required for
      //accuracy, since sin t / t and (1-cos t)/t^2 are both 0/0.
      if (theta_sq < 1e-8) {
              A = 1.0 - one_6th * theta_sq;
              B = 0.5;
      } else {
              if (theta_sq < 1e-6) {
                      B = 0.5 - 0.25 * one_6th * theta_sq;
                      A = 1.0 - theta_sq * one_6th*(1.0 - one_20th * theta_sq);
              } else {
                      const double inv_theta = 1.0/theta;
                      A = std::sin(theta) * inv_theta;
                      B = (1 - std::cos(theta)) * (inv_theta * inv_theta);
              }
      }

      Matrix3d result;
      rodrigues_so3_exp(w, A, B, result);
      return result;
}


// Method from TooN (C) Tom Drummond (GNU GPL)
inline Eigen::Vector3d ln_rotation( Eigen::Matrix3d & rotation ) {
    Eigen::Vector3d result;

    const double cos_angle = (rotation.trace() - 1.0) * 0.5;
    result(0) = (rotation(2,1)-rotation(1,2))*0.5;
    result(1) = (rotation(0,2)-rotation(2,0))*0.5;
    result(2) = (rotation(1,0)-rotation(0,1))*0.5;

    double sin_angle_abs = result.norm(); //std::sqrt(result*result);
    if (cos_angle > M_SQRT1_2)
    {            // (0 - Pi/4( use asin
        if(sin_angle_abs > 0){
            result *= std::asin(sin_angle_abs) / sin_angle_abs;
        }
    }
    else if( cos_angle > -M_SQRT1_2)
    {    // (Pi/4 - 3Pi/4( use acos, but antisymmetric part
        double angle = std::acos(cos_angle);
        result *= angle / sin_angle_abs;
    }
    else
    {  // rest use symmetric part
        // antisymmetric part vanishes, but still large rotation, need information from symmetric part
        const double angle = M_PI - std::asin(sin_angle_abs);
        const double
                d0 = rotation(0,0) - cos_angle,
                d1 = rotation(1,1) - cos_angle,
                d2 = rotation(2,2) - cos_angle;
        Eigen::Vector3d r2;
        if(fabs(d0) > fabs(d1) && fabs(d0) > fabs(d2))
        { // first is largest, fill with first column
            r2(0) = d0;
            r2(1) = (rotation(1,0)+rotation(0,1))*0.5;
            r2(2) = (rotation(0,2)+rotation(2,0))*0.5;
        }
        else if(fabs(d1) > fabs(d2))
        {                           // second is largest, fill with second column
            r2(0) = (rotation(1,0)+rotation(0,1))*0.5;
            r2(1) = d1;
            r2(2) = (rotation(2,1)+rotation(1,2))*0.5;
        }
        else
        {                                                           // third is largest, fill with third column
            r2(0) = (rotation(0,2)+rotation(2,0))*0.5;
            r2(1) = (rotation(2,1)+rotation(1,2))*0.5;
            r2(2) = d2;
        }
        // flip, if we point in the wrong direction!
        if( r2.dot(result) < 0)
            r2 *= -1;
        result = r2;
        result *= (angle/r2.norm());
    }
    return result;
}

// Method from TooN (C) Tom Drummond (GNU GPL)
inline Vector6d ln_affine( const Eigen::Matrix4d & m )
{
    Vector3d t = m.block<3,1>(0,3);
    Matrix3d r = m.block<3,3>(0,0);

    Vector3d rot = ln_rotation( r );
    const double theta =  rot.norm(); //std::sqrt(rot*rot);

    double shtot = 0.5;
    if(theta > 0.00001)
        shtot = std::sin(theta*0.5)/theta;

    // now do the rotation
    Vector3d rot_half = rot;
    rot_half*=-0.5;
    const Matrix3d halfrotator = exp_rotation( rot_half );

    Vector3d rottrans = halfrotator * t;

    if(theta > 0.001)
    {
        rottrans -= rot * ( t.dot(rot) * (1-2*shtot) / rot.squaredNorm() ); //(rot*rot));
    }
    else
    {
        rottrans -= rot * ( t.dot(rot)/24);
    }

    rottrans *= 1.0/(2 * shtot);

    Vector6d result;
    result.block<3,1>(0,0) = rottrans;
    result.block<3,1>(3,0) = rot;
    return result;

//    for (int i=0;i<3;i++) result(i) = rottrans(i);
//    for (int i=0;i<3;i++) result(3+i) = rot(i);
}

// Method from TooN (C) Tom Drummond (GNU GPL)
inline Matrix4d exp_affine(const Vector6d & mu)
{
        static const double one_6th = 1.0/6.0;
        static const double one_20th = 1.0/20.0;

        // Resulting XYZ coords:
        Vector3d res_xyz;

        Vector3d mu_xyz = mu.block<3,1>(0,0);
//        for (int i=0;i<3;i++) mu_xyz(i) = mu(i);

        Vector3d w = mu.block<3,1>(3,0);
//        for (int i=0;i<3;i++) w(i) = mu(3+i);

        const double theta_sq = w.squaredNorm(); // w*w;
        const double theta = std::sqrt(theta_sq);
        double A, B;

//        CArrayDouble<3> cross;
//        mrpt::math::crossProduct3D(w, mu_xyz, cross );

        Vector3d cross = w.cross( mu_xyz );

        if (theta_sq < 1e-8)
        {
                A = 1.0 - one_6th * theta_sq;
                B = 0.5;
                res_xyz(0) = mu_xyz(0) + 0.5 * cross(0);
                res_xyz(1) = mu_xyz(1) + 0.5 * cross(1);
                res_xyz(2) = mu_xyz(2) + 0.5 * cross(2);
        }
        else
        {
                double C;
                if (theta_sq < 1e-6)
                {
                        C = one_6th*(1.0 - one_20th * theta_sq);
                        A = 1.0 - theta_sq * C;
                        B = 0.5 - 0.25 * one_6th * theta_sq;
                }
                else
                {
                        const double inv_theta = 1.0/theta;
                        A = std::sin(theta) * inv_theta;
                        B = (1 - std::cos(theta)) * (inv_theta * inv_theta);
                        C = (1 - A) * (inv_theta * inv_theta);
                }

//                CArrayDouble<3> w_cross;        // = w^cross
//                mrpt::math::crossProduct3D(w, cross, w_cross );

                Vector3d w_cross = w.cross( cross );

                //result.get_translation() = mu_xyz + B * cross + C * (w ^ cross);
                res_xyz(0) = mu_xyz(0) + B * cross(0) + C * w_cross(0);
                res_xyz(1) = mu_xyz(1) + B * cross(1) + C * w_cross(1);
                res_xyz(2) = mu_xyz(2) + B * cross(2) + C * w_cross(2);
        }

        // 3x3 rotation part:
        Matrix3d res_ROT;
        rodrigues_so3_exp(w, A, B, res_ROT);

        Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
        result.block<3,3>(0,0) = res_ROT;
        result.block<3,1>(0,3) = res_xyz;

        return result;

//        return CPose3D(res_ROT, res_xyz);
}



/* crossover point to Taylor Series approximation.  Figuring 16
 * decimal digits of precision for doubles, the Taylor approximation
 * should be indistinguishable (to machine precision) for angles
 * of magnitude less than 1e-4. To be conservative, we add on three
 * additional orders of magnitude.  */
const double MIN_ANGLE = 1e-7;

/* Angle beyond which we perform dynamic reparameterization of a 3 DOF EM */
const double CUTOFF_ANGLE = M_PI;

inline void V3Scale(const Eigen::Vector3d & v, const double s1, Eigen::Vector3d & prod)
{
    prod(0) = v(0) * s1;
    prod(1) = v(1) * s1;
    prod(2) = v(2) * s1;
}

/* -----------------------------------------------------------------
 * 'Check_Parameterization' To escape the vanishing derivatives at
 * shells of 2PI rotations, we reparameterize to a rotation of (2PI -
 * theta) about the opposite axis when we get too close to 2PI
 * -----------------------------------------------------------------*/
inline int Check_Parameterization(Eigen::Vector3d & v, double *theta)
{
    int     rep = 0;
    *theta = v.norm();

    if (*theta > CUTOFF_ANGLE){
    double scl = *theta;
    if (*theta > 2*M_PI){	/* first get theta into range 0..2PI */
        *theta = fmod(*theta, 2*M_PI);
        scl = *theta/scl;
        V3Scale(v, scl, v);
        rep = 1;
    }
    if (*theta > CUTOFF_ANGLE){
        scl = *theta;
        *theta = 2*M_PI - *theta;
        scl = 1.0 - 2*M_PI/scl;
        V3Scale(v, scl, v);
        rep = 1;
    }
    }

    return rep;
}

/* -----------------------------------------------------------------
 * 'EM_To_Q' Convert a 3 DOF EM vector 'v' into its corresponding
 * quaternion 'q'. If 'reparam' is non-zero, perform dynamic
 * reparameterization, if necessary, storing the reparameterized EM in
 * 'v' and returning 1.  Returns 0 if no reparameterization was
 * performed.
 * -----------------------------------------------------------------*/
inline int EM_To_Q(Eigen::Vector3d& v, Eigen::Quaterniond & q, int reparam)
{
    int      rep=0;
    double   cosp, sinp, theta;

    if (reparam)
      rep = Check_Parameterization(v, &theta);
    else
        theta = v.norm();

    cosp = std::cos(.5*theta);
    sinp = std::sin(.5*theta);

    q.w() = cosp;
    Eigen::Vector3d qv;
    if (theta < MIN_ANGLE)
      V3Scale(v, .5 - theta*theta/48.0, qv);	/* Taylor Series for sinc */
    else
      V3Scale(v, sinp/theta, qv);

    q.x() = qv(0);
    q.y() = qv(1);
    q.z() = qv(2);
    return rep;
}





/* -----------------------------------------------------------------
 * 'Partial_Q_Partial_3V' Partial derivative of quaternion wrt i'th
 * component of EM vector 'v'
 * -----------------------------------------------------------------*/
inline void Partial_Q_Partial_3V(const Eigen::Vector3d & v, int i, Eigen::Vector4d& dq_dvi)
{
    double   theta = v.norm();
    double   cosp = std::cos(.5*theta);
    double   sinp = std::sin(.5*theta);

    assert(i>=0 && i<3);

    /* This is an efficient implementation of the derivatives given
     * in Appendix A of the paper with common subexpressions factored out */
    if (theta < MIN_ANGLE){
        const int i2 = (i+1)%3, i3 = (i+2)%3;
        double Tsinc = 0.5 - theta*theta/48.0;
        double vTerm = v(i) * (theta*theta/40.0 - 1.0) / 24.0;

        dq_dvi(i)  = v[i]* vTerm + Tsinc;
        dq_dvi(i2) = v[i2]*vTerm;
        dq_dvi(i3) = v[i3]*vTerm;
        dq_dvi(3) = -.5*v[i]*Tsinc;
    }
    else{
        const int i2 = (i+1)%3, i3 = (i+2)%3;
        const double  ang = 1.0/theta, ang2 = ang*ang*v[i], sang = sinp*ang;
        const double  cterm = ang2*(.5*cosp - sang);

        dq_dvi(i)  = cterm*v[i] + sang;
        dq_dvi(i2) = cterm*v[i2];
        dq_dvi(i3) = cterm*v[i3];
        dq_dvi(3) = -.5*v[i]*sang;
    }
}

// compute matrix of partial derivatives of Q wrt V
inline void JacobianQ_3V( const Eigen::Vector3d & v, Eigen::Matrix<double, 4, 3> & J ) {
    Eigen::Vector4d dq_dvx, dq_dvy, dq_dvz;
    Partial_Q_Partial_3V( v, 0, dq_dvx );
    Partial_Q_Partial_3V( v, 1, dq_dvy );
    Partial_Q_Partial_3V( v, 2, dq_dvz );
    J.block<4,1>(0,0) = dq_dvx;
    J.block<4,1>(0,1) = dq_dvy;
    J.block<4,1>(0,2) = dq_dvz;
}


inline Eigen::Matrix3d skewSymmetricMatrix( const Eigen::Vector3d & v ) {
    Matrix3d result = Eigen::Matrix3d::Zero();
    result(0,1) = -v(2);
    result(0,2) = v(1);
    result(1,0) = v(2);
    result(1,2) = -v(0);
    result(2,0) = -v(1);
    result(2,1) = v(0);

    return result;
}

inline Eigen::Vector3d skewSymmetricToVector( const Eigen::Matrix3d & m ) {
    Vector3d result;
    result(0) = m(2,1);
    result(1) = m(0,2);
    result(2) = m(1,0);

    return result;
}



}


#endif
