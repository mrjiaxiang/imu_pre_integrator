#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace lidar_localization {
class PreIntegrator {
  public:
    bool IsInited(void) const { return is_inited_; }

    double GetTime() const { return time_; }

  private:
    PreIntegrator() {}

    bool is_inited_{false};

    double time_;
};
} // namespace lidar_localization