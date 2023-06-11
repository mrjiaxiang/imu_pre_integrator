#pragma once

#include "lidar_localization/graph_optimizer/g2o/edge/edge_prvag_imu_pre_integration.hpp"
#include "lidar_localization/pre_integrator/pre_integrator.hpp"
#include "lidar_localization/sensor_data/imu_data.hpp"

#include <sophus/so3.hpp>

namespace lidar_localization {
class IMUPreIntegrator : public PreIntegrator {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  public:
    static const int DIM_STATE = 15;
    typedef Eigen::Matrix<double, DIM_STATE, DIM_STATE> MatrixP;
    typedef Eigen::Matrix<double, DIM_STATE, DIM_STATE> MatrixJ;

    struct IMUPreIntegration {
        double T_;

        Eigen::Vector3d g_;

        Eigen::Vector3d alpha_ij_;
        Sophus::SO3d theta_ij_;
        Eigen::Vector3d beta_ij_;
        Eigen::Vector3d b_a_i_;
        Eigen::Vector3d b_g_i_;

        // 信息矩阵
        MatrixP P_;
        // c. Jacobian for update caused by bias:
        MatrixJ J_;

        double GetT() const { return T_; }

        Eigen::Vector3d GetGravity() const { return g_; }

        Eigen::MatrixXd GetInformation() const { return P_.inverse(); }

        Eigen::MatrixXd GetJacobian() const { return J_; }

        Vector15d GetMeasurement(void) const {
            Vector15d measurement = Vector15d::Zero();

            measurement.block<3, 1>(g2o::EdgePRVAGIMUPreIntegration::INDEX_P,
                                    0) = alpha_ij_;
            measurement.block<3, 1>(g2o::EdgePRVAGIMUPreIntegration::INDEX_R,
                                    0) = theta_ij_.log();
            measurement.block<3, 1>(g2o::EdgePRVAGIMUPreIntegration::INDEX_V,
                                    0) = beta_ij_;

            return measurement;
        }
    };

    IMUPreIntegrator();

    bool Init(const IMUData &init_imu_data);

    bool Update(const IMUData &imu_data);

    bool Reset(const IMUData &init_imu_data,
               IMUPreIntegration &imu_pre_integration);

  private:
    static const int DIM_NOISE = 18;

    static const int INDEX_ALPHA = 0;
    static const int INDEX_THETA = 3;
    static const int INDEX_BETA = 6;
    static const int INDEX_B_A = 9;
    static const int INDEX_B_G = 12;

    static const int INDEX_M_ACC_PREV = 0;
    static const int INDEX_M_GYR_PREV = 3;
    static const int INDEX_M_ACC_CURR = 6;
    static const int INDEX_M_GYR_CURR = 9;
    static const int INDEX_R_ACC_PREV = 12;
    static const int INDEX_R_GYR_PREV = 15;

    typedef Eigen::Matrix<double, DIM_STATE, DIM_STATE> MatrixF;
    typedef Eigen::Matrix<double, DIM_STATE, DIM_NOISE> MatrixB;
    typedef Eigen::Matrix<double, DIM_NOISE, DIM_NOISE> MatrixQ;

    std::deque<IMUData> imu_data_buff_;

    // hyper-params:
    // a. earth constants:
    struct {
        double GRAVITY_MAGNITUDE;
    } EARTH;
    // b. prior state covariance, process & measurement noise:
    struct {
        struct {
            double ACCEL;
            double GYRO;
        } RANDOM_WALK;
        struct {
            double ACCEL;
            double GYRO;
        } MEASUREMENT;
    } COV;

    struct {
        Eigen::Vector3d g_;
        Eigen::Vector3d alpha_ij_;
        Sophus::SO3d theta_ij_;
        Eigen::Vector3d beta_ij_;
        Eigen::Vector3d b_a_i_;
        Eigen::Vector3d b_g_i_;
    } state;

    // state covariance
    MatrixP P_ = MatrixP::Zero();

    // jacobian
    MatrixJ J_ = MatrixJ::Identity();

    // process noise
    MatrixQ Q_ = MatrixQ::Zero();

    // process equation
    MatrixF F_ = MatrixF::Zero();
    MatrixB B_ = MatrixB::Zero();

    void ResetState(const IMUData &init_imu_data);

    void UpdateState();
};
} // namespace lidar_localization