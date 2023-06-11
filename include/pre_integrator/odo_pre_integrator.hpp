#pragma once

#include "lidar_localization/graph_optimizer/g2o/edge/edge_pravg_odo_pre_integration.hpp"
#include "lidar_localization/pre_integrator/pre_integrator.hpp"
#include "lidar_localization/sensor_data/velocity_data.hpp"

#include <sophus/so3.hpp>

namespace lidar_localization {
class OdoPreIntegrator : public PreIntegrator {
  public:
    static const int DIM_STATE = 6;
    typedef Eigen::Matrix<double, DIM_STATE, DIM_STATE> MatrixP;

    struct OdoPreIntegration {
        // 相对旋转鱼平移
        Eigen::Vector3d alpha_ij_;
        Sophus::SO3d theta_ij_;

        MatrixP P_;

        g2o::EdgePRVAGOdoPreIntegration::Measurement GetMeasurement() const {
            g2o::EdgePRVAGOdoPreIntegration::Measurement measurement =
                g2o::EdgePRVAGOdoPreIntegration::Measurement::Zero();

            measurement.block<3, 1>(g2o::EdgePRVAGOdoPreIntegration::
                                        EdgePRVAGOdoPreIntegration::INDEX_P,
                                    0) = alpha_ij_;
            measurement.block<3, 1>(g2o::EdgePRVAGOdoPreIntegration::
                                        EdgePRVAGOdoPreIntegration::INDEX_R,
                                    0) = theta_ij_.log();

            return measurement;
        }

        Eigen::MatrixXd GetInformation() const { return P_.inverse(); }
    };

    OdoPreIntegrator();

    bool Init(const VelocityData &init_velocity_data);

    bool Update(const VelocityData &velocity_data);

    bool Reset(const VelocityData &init_velocity_data,
               OdoPreIntegration &odo_pre_integration);

  private:
    static const int DIM_NOISE = 12;

    static const int INDEX_ALPHA = 0;
    static const int INDEX_THETA = 3;

    static const int INDEX_M_V_PREV = 0;
    static const int INDEX_M_W_PREV = 0;
    static const int INDEX_M_V_CURR = 6;
    static const int INDEX_M_W_CURR = 9;

    typedef Eigen::Matrix<double, DIM_STATE, DIM_STATE> MatrixF;
    typedef Eigen::Matrix<double, DIM_STATE, DIM_NOISE> MatrixB;
    typedef Eigen::Matrix<double, DIM_NOISE, DIM_NOISE> MatrixQ;

    std::deque<VelocityData> odo_data_buff_;

    // a. prior state covariance, process & measurement noise:
    struct {
        struct {
            double V;
            double W;
        } MEASUREMENT;
    } COV;

    struct {
        Eigen::Vector3d alpha_ij_;
        Sophus::SO3d theta_ij_;
    } state;

    MatrixP P_ = MatrixP::Zero();
    MatrixQ Q_ = MatrixQ::Zero();
    MatrixF F_ = MatrixF::Zero();
    MatrixB B_ = MatrixB::Zero();

    void ResetState(const VelocityData &init_velocity_data);

    void UpdateState();
};
} // namespace lidar_localization