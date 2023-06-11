#include "lidar_localization/pre_integrator/imu_pre_integrator.hpp"

#include <glog/logging.h>

namespace lidar_localization {

IMUPreIntegrator::IMUPreIntegrator() {

    LOG(INFO) << std::endl
              << "IMU Pre-Integration params:" << std::endl
              << "\tgravity magnitude: " << EARTH.GRAVITY_MAGNITUDE << std::endl
              << std::endl
              << "\tprocess noise:" << std::endl
              << "\t\tmeasurement:" << std::endl
              << "\t\t\taccel.: " << COV.MEASUREMENT.ACCEL << std::endl
              << "\t\t\tgyro.: " << COV.MEASUREMENT.GYRO << std::endl
              << "\t\trandom_walk:" << std::endl
              << "\t\t\taccel.: " << COV.RANDOM_WALK.ACCEL << std::endl
              << "\t\t\tgyro.: " << COV.RANDOM_WALK.GYRO << std::endl
              << std::endl;
    state.g_ = Eigen::Vector3d(0.0, 0.0, EARTH.GRAVITY_MAGNITUDE);

    Q_.block<3, 3>(INDEX_M_ACC_PREV, INDEX_M_ACC_PREV) =
        Q_.block<3, 3>(INDEX_M_ACC_CURR, INDEX_M_ACC_CURR) =
            COV.MEASUREMENT.ACCEL * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(INDEX_M_GYR_PREV, INDEX_M_GYR_PREV) =
        Q_.block<3, 3>(INDEX_M_GYR_CURR, INDEX_M_GYR_CURR) =
            COV.MEASUREMENT.GYRO * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(INDEX_R_ACC_PREV, INDEX_R_ACC_PREV) =
        COV.RANDOM_WALK.ACCEL * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(INDEX_R_GYR_PREV, INDEX_R_GYR_PREV) =
        COV.RANDOM_WALK.GYRO * Eigen::Matrix3d::Identity();

    F_.block<3, 3>(INDEX_ALPHA, INDEX_BETA) = Eigen::Matrix3d::Identity();
    F_.block<3, 3>(INDEX_THETA, INDEX_B_G) = -Eigen::Matrix3d::Identity();

    B_.block<3, 3>(INDEX_THETA, INDEX_M_GYR_PREV) =
        B_.block<3, 3>(INDEX_THETA, INDEX_M_GYR_CURR) =
            0.50 * Eigen::Matrix3d::Identity();
    B_.block<3, 3>(INDEX_B_A, INDEX_R_ACC_PREV) =
        B_.block<3, 3>(INDEX_B_G, INDEX_R_GYR_PREV) =
            Eigen::Matrix3d::Identity();
}

bool IMUPreIntegrator::Init(const IMUData &init_imu_data) {
    ResetState(init_imu_data);
    is_inited_ = true;
    return true;
}

void IMUPreIntegrator::ResetState(const IMUData &init_imu_data) {
    time_ = init_imu_data.time;

    state.alpha_ij_ = Eigen::Vector3d::Zero();
    state.theta_ij_ = Sophus::SO3d();
    state.beta_ij_ = Eigen::Vector3d::Zero();
    state.b_a_i_ =
        Eigen::Vector3d(init_imu_data.accel_bias.x, init_imu_data.accel_bias.y,
                        init_imu_data.accel_bias.z);
    state.b_g_i_ =
        Eigen::Vector3d(init_imu_data.gyro_bias.x, init_imu_data.gyro_bias.y,
                        init_imu_data.gyro_bias.z);

    // reset state covariance:
    P_ = MatrixP::Zero();

    // reset jacobian
    J_ = MatrixJ::Identity();

    // reset buffer
    imu_data_buff_.clear();
    imu_data_buff_.push_back(init_imu_data);
}

bool IMUPreIntegrator::Reset(const IMUData &init_imu_data,
                             IMUPreIntegration &imu_pre_integration) {
    // one last update:
    Update(init_imu_data);

    // set output IMU pre-integration:
    imu_pre_integration.T_ = init_imu_data.time - time_;

    // set gravity constant:
    imu_pre_integration.g_ = state.g_;

    // set measurement:
    imu_pre_integration.alpha_ij_ = state.alpha_ij_;
    imu_pre_integration.theta_ij_ = state.theta_ij_;
    imu_pre_integration.beta_ij_ = state.beta_ij_;
    imu_pre_integration.b_a_i_ = state.b_a_i_;
    imu_pre_integration.b_g_i_ = state.b_g_i_;
    // set information:
    imu_pre_integration.P_ = P_;
    // set Jacobian:
    imu_pre_integration.J_ = J_;

    // reset:
    ResetState(init_imu_data);

    return true;
}

bool IMUPreIntegrator::Update(const IMUData &imu_data) {
    // 新来的imu
    if (imu_data_buff_.front().time < imu_data.time) {
        imu_data_buff_.push_back(imu_data);
        UpdateState();

        imu_data_buff_.pop_front();
    }

    return true;
}

void IMUPreIntegrator::UpdateState() {
    static double T = 0.0;
    static Eigen::Vector3d w_mid = Eigen::Vector3d::Zero();
    static Eigen::Vector3d a_mid = Eigen::Vector3d::Zero();

    static Sophus::SO3d prev_theta_ij = Sophus::SO3d();
    static Sophus::SO3d curr_theta_ij = Sophus::SO3d();
    static Sophus::SO3d d_theta_ij = Sophus::SO3d();

    static Eigen::Matrix3d dR_inv = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d prev_R = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d curr_R = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d prev_R_a_hat = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d curr_R_a_hat = Eigen::Matrix3d::Zero();

    const IMUData &prev_imu_data = imu_data_buff_.at(0);
    const IMUData &curr_imu_data = imu_data_buff_.at(1);

    T = curr_imu_data.time - prev_imu_data.time;

    const Eigen::Vector3d prev_w(
        prev_imu_data.angular_velocity.x - state.b_g_i_.x(),
        prev_imu_data.angular_velocity.y - state.b_g_i_.y(),
        prev_imu_data.angular_velocity.z - state.b_g_i_.z());

    const Eigen::Vector3d curr_w(
        curr_imu_data.angular_velocity.x - state.b_g_i_.x(),
        curr_imu_data.angular_velocity.y - state.b_g_i_.y(),
        curr_imu_data.angular_velocity.z - state.b_g_i_.z());

    const Eigen::Vector3d prev_a(
        prev_imu_data.linear_acceleration.x - state.b_a_i_.x(),
        prev_imu_data.linear_acceleration.y - state.b_a_i_.y(),
        prev_imu_data.linear_acceleration.z - state.b_a_i_.z());

    const Eigen::Vector3d curr_a(
        curr_imu_data.linear_acceleration.x - state.b_a_i_.x(),
        curr_imu_data.linear_acceleration.y - state.b_a_i_.y(),
        curr_imu_data.linear_acceleration.z - state.b_a_i_.z());

    w_mid = 0.5 * (prev_w + curr_w);
    prev_theta_ij = state.theta_ij_;
    d_theta_ij = Sophus::SO3d::exp(w_mid * T);
    state.theta_ij_ = state.theta_ij_ * d_theta_ij; // 更新ij
    curr_theta_ij = state.theta_ij_;

    a_mid = 0.5 * (prev_theta_ij * prev_a + curr_theta_ij * curr_a);
    state.alpha_ij_ += state.beta_ij_ * T + 0.5 * a_mid * T * T;
    state.beta_ij_ += a_mid * T;

    dR_inv = d_theta_ij.inverse().matrix();
    prev_R = prev_theta_ij.matrix();
    curr_R = curr_theta_ij.matrix();
    prev_R_a_hat = prev_R * Sophus::SO3d::hat(prev_a);
    curr_R_a_hat = curr_R * Sophus::SO3d::hat(curr_a);

    // F
    F_ = MatrixF::Identity();
    // ALPHA rows
    F_.block<3, 3>(INDEX_ALPHA, INDEX_THETA) =
        -0.25 * T * T *
        (prev_R_a_hat + curr_R_a_hat * (Eigen::Matrix3d::Identity() -
                                        Sophus::SO3d::hat(w_mid) * T));
    F_.block<3, 3>(INDEX_ALPHA, INDEX_BETA) = Eigen::Matrix3d::Identity() * T;
    F_.block<3, 3>(INDEX_ALPHA, INDEX_B_A) = -0.25 * (prev_R + curr_R) * T * T;
    F_.block<3, 3>(INDEX_ALPHA, INDEX_B_G) = 0.25 * T * T * T * curr_R_a_hat;

    // theta rows
    F_.block<3, 3>(INDEX_THETA, INDEX_THETA) =
        Eigen::Matrix3d::Identity() - Sophus::SO3d::hat(w_mid) * T;
    F_.block<3, 3>(INDEX_THETA, INDEX_B_G) = -Eigen::Matrix3d::Identity() * T;

    // beta rows
    F_.block<3, 3>(INDEX_BETA, INDEX_THETA) =
        -0.5 * T *
        (prev_R_a_hat + curr_R_a_hat * (Eigen::Matrix3d::Identity() -
                                        Sophus::SO3d::hat(w_mid) * T));
    F_.block<3, 3>(INDEX_BETA, INDEX_B_A) = -0.5 * (prev_R + curr_R) * T;
    F_.block<3, 3>(INDEX_BETA, INDEX_B_G) = 0.5 * T * T * curr_R_a_hat;

    // B
    B_ = MatrixB::Zero();

    // alpha
    B_.block<3, 3>(INDEX_ALPHA, INDEX_M_ACC_PREV) = 0.25 * prev_R * T * T;
    B_.block<3, 3>(INDEX_ALPHA, INDEX_M_GYR_PREV) =
        -0.125 * T * T * T * curr_R_a_hat;
    B_.block<3, 3>(INDEX_ALPHA, INDEX_M_ACC_CURR) = 0.25 * curr_R * T * T;
    B_.block<3, 3>(INDEX_ALPHA, INDEX_M_GYR_CURR) =
        -0.125 * T * T * T * curr_R_a_hat;
    // theta
    B_.block<3, 3>(INDEX_THETA, INDEX_M_GYR_PREV) =
        0.5 * Eigen::Matrix3d::Identity() * T;
    B_.block<3, 3>(INDEX_THETA, INDEX_M_GYR_CURR) =
        0.5 * Eigen::Matrix3d::Identity() * T;
    // beta
    B_.block<3, 3>(INDEX_BETA, INDEX_M_ACC_PREV) = 0.5 * prev_R * T;
    B_.block<3, 3>(INDEX_BETA, INDEX_M_GYR_PREV) = -0.25 * T * T * curr_R_a_hat;
    B_.block<3, 3>(INDEX_BETA, INDEX_M_ACC_CURR) = 0.5 * curr_R * T;
    B_.block<3, 3>(INDEX_BETA, INDEX_M_GYR_CURR) = -0.25 * T * T * curr_R_a_hat;
    // ba
    B_.block<3, 3>(INDEX_B_A, INDEX_R_ACC_PREV) =
        Eigen::Matrix3d::Identity() * T;
    // bg
    B_.block<3, 3>(INDEX_B_G, INDEX_R_GYR_PREV) =
        Eigen::Matrix3d::Identity() * T;

    // update P_
    P_ = F_ * P_ * F_.transpose() + B_ * Q_ * B_.transpose();

    // update Jacobian
    J_ = F_ * J_;
}

} // namespace lidar_localization