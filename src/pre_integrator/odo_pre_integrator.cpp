#include "lidar_localization/pre_integrator/odo_pre_integrator.hpp"
#include "lidar_localization/global_defination/global_defination.h"

#include <glog/logging.h>

namespace lidar_localization {
OdoPreIntegrator::OdoPreIntegrator() {
    COV.MEASUREMENT.V = 0;
    COV.MEASUREMENT.W = 0;

    LOG(INFO) << std::endl
              << "Odo Pre-Integration params:" << std::endl
              << "\tprocess noise:" << std::endl
              << "\t\tmeasurement:" << std::endl
              << "\t\t\tv.: " << COV.MEASUREMENT.V << std::endl
              << "\t\t\tw.: " << COV.MEASUREMENT.W << std::endl
              << std::endl;

    Q_.block<3, 3>(INDEX_M_V_PREV, INDEX_M_V_PREV) =
        Q_.block<3, 3>(INDEX_M_V_CURR, INDEX_M_V_CURR) =
            COV.MEASUREMENT.V * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(INDEX_M_W_PREV, INDEX_M_W_PREV) =
        Q_.block<3, 3>(INDEX_M_W_CURR, INDEX_M_W_CURR) =
            COV.MEASUREMENT.W * Eigen::Matrix3d::Identity();

    B_.block<3, 3>(INDEX_THETA, INDEX_M_W_PREV) =
        B_.block<3, 3>(INDEX_THETA, INDEX_M_W_CURR) =
            0.50 * Eigen::Matrix3d::Identity();
}

bool OdoPreIntegrator::Init(const VelocityData &init_velocity_data) {
    ResetState(init_velocity_data);

    is_inited_ = true;

    return true;
}

void OdoPreIntegrator::ResetState(const VelocityData &init_velocity_data) {
    time_ = init_velocity_data.time;

    state.alpha_ij_ = Eigen::Vector3d::Zero();
    state.theta_ij_ = Sophus::SO3d();

    P_ = MatrixP::Zero();
    odo_data_buff_.clear();
    odo_data_buff_.push_back(init_velocity_data);
}

bool OdoPreIntegrator::Reset(const VelocityData &init_velocity_data,
                             OdoPreIntegration &odo_pre_integration) {
    Update(init_velocity_data);

    odo_pre_integration.alpha_ij_ = state.alpha_ij_;
    odo_pre_integration.theta_ij_ = state.theta_ij_;

    odo_pre_integration.P_ = P_;

    ResetState(init_velocity_data);

    return true;
}

bool OdoPreIntegrator::Update(const VelocityData &velocity_data) {
    if (odo_data_buff_.front().time < velocity_data.time) {
        odo_data_buff_.push_back(velocity_data);

        UpdateState();

        odo_data_buff_.pop_front();
    }
    return true;
}

void OdoPreIntegrator::UpdateState() {
    static double T = 0.0;

    static Eigen::Vector3d w_mid = Eigen::Vector3d::Zero();
    static Eigen::Vector3d v_mid = Eigen::Vector3d::Zero();

    static Sophus::SO3d prev_theta_ij = Sophus::SO3d();
    static Sophus::SO3d curr_theta_ij = Sophus::SO3d();
    static Sophus::SO3d d_theta_ij = Sophus::SO3d();

    static Eigen::Matrix3d dR_inv = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d prev_R = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d curr_R = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d prev_R_v_hat = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d curr_R_v_hat = Eigen::Matrix3d::Zero();

    const VelocityData &prev_odo_data = odo_data_buff_.at(0);
    const VelocityData &curr_odo_data = odo_data_buff_.at(1);

    T = curr_odo_data.time - prev_odo_data.time;

    const Eigen::Vector3d prev_w(prev_odo_data.angular_velocity.x,
                                 prev_odo_data.angular_velocity.y,
                                 prev_odo_data.angular_velocity.z);
    const Eigen::Vector3d curr_w(curr_odo_data.angular_velocity.x,
                                 curr_odo_data.angular_velocity.y,
                                 curr_odo_data.angular_velocity.z);
    const Eigen::Vector3d prev_v(curr_odo_data.linear_velocity.x,
                                 prev_odo_data.linear_velocity.y,
                                 prev_odo_data.linear_velocity.z);
    const Eigen::Vector3d curr_v(curr_odo_data.linear_velocity.x,
                                 curr_odo_data.linear_velocity.y,
                                 curr_odo_data.linear_velocity.z);

    w_mid = 0.5 * (prev_w + curr_w);

    prev_theta_ij = state.theta_ij_;
    d_theta_ij = Sophus::SO3d::exp(w_mid * T);
    state.theta_ij_ = state.theta_ij_ * d_theta_ij;
    curr_theta_ij = state.theta_ij_;

    v_mid = 0.5 * (prev_theta_ij * prev_v + curr_theta_ij * curr_v);
    state.alpha_ij_ += v_mid * T;

    dR_inv = d_theta_ij.inverse().matrix();
    prev_R = prev_theta_ij.matrix();
    curr_R = curr_theta_ij.matrix();

    prev_R_v_hat = prev_R * Sophus::SO3d::hat(prev_v);
    curr_R_v_hat = curr_R * Sophus::SO3d::hat(curr_v);

    F_.block<3, 3>(INDEX_ALPHA, INDEX_THETA) =
        -0.5 * (prev_R_v_hat + curr_R_v_hat * dR_inv);

    F_.block<3, 3>(INDEX_THETA, INDEX_THETA) = -Sophus::SO3d::hat(w_mid);

    B_.block<3, 3>(INDEX_ALPHA, INDEX_M_V_PREV) = 0.5 * prev_R;
    B_.block<3, 3>(INDEX_ALPHA, INDEX_M_V_CURR) = +0.50 * curr_R;

    B_.block<3, 3>(INDEX_ALPHA, INDEX_M_W_PREV) =
        B_.block<3, 3>(INDEX_ALPHA, INDEX_M_W_CURR) = -0.25 * T * curr_R_v_hat;

    MatrixF F = MatrixF::Identity() + T * F_;
    MatrixB B = T * B_;

    P_ = F * P_ * F.transpose() + B * Q_ * B.transpose();
}

} // namespace lidar_localization