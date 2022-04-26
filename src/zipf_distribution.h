#pragma once

#include <limits>
#include <random>

template <typename T>
class ZipfDistribution {
 public:
  // Zipf distribution in the range [1,max]
  // The distribution follows the power-law 1/(n)^s
  ZipfDistribution(T max = std::numeric_limits<T>::max(), const double exp = 1.0)
      : max_(max),
        exp_(exp),
        om_exp(1.0 - exp),
        spole(abs(om_exp) < epsilon),
        rv_exp(spole ? 0.0 : 1.0 / om_exp),
        H_x1(H(1.5) - h(1.0)),
        H_n(H(max_ + 0.5)),
        cut(1.0 - H_inv(H(1.5) - h(1.0))),
        dist(H_x1, H_n) {}

  T operator()(std::mt19937& random_generator) {
    while (true) {
      const double u = dist(random_generator);
      const double x = H_inv(u);
      const T k = static_cast<T>(x);
      if (k - x <= cut) return k;
      if (u >= H(k + 0.5) - h(k)) return k;
    }
  }

  double exp() const { return exp_; }
  T min() const { return 1; }
  T max() const { return max_; }

 private:
  // (exp(x) - 1) / x
  static double expxm1bx(const double x) {
    if (std::abs(x) > epsilon) return std::expm1(x) / x;
    return (1.0 + x / 2.0 * (1.0 + x / 3.0 * (1.0 + x / 4.0)));
  }

  // log(1 + x) / x
  static double log1pxbx(const double x) {
    if (std::abs(x) > epsilon) return std::log1p(x) / x;
    return 1.0 - x * ((1 / 2.0) - x * ((1 / 3.0) - x * (1 / 4.0)));
  }
  // The hat function h(x) = 1/(x)^s
  double h(const double x) { return std::pow(x, -exp_); }

  /**
   * H(x) is an integral of h(x).
   *     H(x) = [(x)^(1-exp) - (1)^(1-exp)] / (1-exp)
   * and if exp==1 then
   *     H(x) = log(x) - log(1)
   */
  double H(const double x) {
    if (!spole) return std::pow(x, om_exp) / om_exp;

    double log_x = std::log(x);
    return log_x * expxm1bx(om_exp * log_x);
  }

  /**
   * The inverse function of H(x).
   *    H^{-1}(y) = [(1-exp)y + 1^{1-exp}]^{1/(1-exp)}
   * Same convergence issues as above; two regimes.
   *
   * For exp far away from 1.0 use the paper version
   *    H^{-1}(y) = (y(1-exp))^{1/(1-exp)}
   */
  double H_inv(const double y) {
    if (!spole) return std::pow(y * om_exp, rv_exp);

    return std::exp(y * log1pxbx(om_exp * y));
  }

  T max_;
  double exp_;
  double om_exp;
  bool spole;
  double rv_exp;
  double H_x1;
  double H_n;
  double cut;
  std::uniform_real_distribution<double> dist;
  static constexpr double epsilon = 2e-5;
};