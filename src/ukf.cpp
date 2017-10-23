#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

const float EPSILON = 0.001;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // std_a_ = 30;
  std_a_ = 2.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // std_yawdd_ = 30;
  std_yawdd_ = M_PI/6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;
  n_sig_ = 2 * n_aug_ + 1;

  lambda_ = 3 - n_aug_;

  x_ = VectorXd(n_x_);
  P_ = MatrixXd(n_x_, n_x_);

  Xsig_aug_ = MatrixXd(n_aug_, n_sig_);

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // set weights
  weights_ = VectorXd(n_sig_);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < n_sig_; i ++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  R_ = MatrixXd(2, 2);
  R_ << std_laspx_ * std_laspx_, 0,
        0,  std_laspy_ * std_laspy_;

  NIS_radar_ = 0.0;
  NIS_laser_ = 0.0;

  P_ << 1,  0,  0,  0,  0,
        0,  1,  0,  0,  0,
        0,  0,  10,  0,  0,
        0,  0,  0,  50,  0,
        0,  0,  0,  0,  50;

  H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

  I_ = MatrixXd::Identity(n_x_, n_x_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if(!is_initialized_){
    x_ << 0, 0, 0, 0, 0;
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      float rho = meas_package.raw_measurements_[0];
      rho = std::max(rho, EPSILON);
      
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];

      x_ << rho * cos(phi),
            rho * sin(phi),
            rho_dot,
            0,
            0;

    }else if(meas_package.sensor_type_ == MeasurementPackage::LASER){
      float px = meas_package.raw_measurements_(0);
      float py = meas_package.raw_measurements_[1];
      if ((abs(px) < EPSILON) && (abs(py) < EPSILON)){
        px = EPSILON;
        py = EPSILON;
      }

      x_ << px,
            py,
            0,
            0,
            0;
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  UKF::Prediction(dt);

  if(use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UKF::UpdateRadar(meas_package);
  }else if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UKF::UpdateLidar(meas_package);
  }
  // exit(0);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  // std::cout << "Prediction called, delta_t = " << delta_t << std::endl;

  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  UKF::GenerateSigmaPoints(&Xsig_aug);

  // std::cout << "Created Augmented Sigma Points = " << std::endl << Xsig_aug << std::endl;

  UKF::SigmaPointPrediction(delta_t, Xsig_aug);

  // std::cout << "Predicted Sigma Points = " << std::endl << Xsig_aug << std::endl;

  UKF::PredictMeanAndCovariance();

  // Xsig_pred_ = Xsig_pred;
}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // Create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  // Calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  // Create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_aug_, n_sig_);
 
  // Create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // Create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  // Set first column of sigma point matrix
  Xsig.col(0)  = x_aug;

  // Set remaining sigma points
  for (int i = 0; i < n_aug_; i++){
    Xsig.col(i + 1)           = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig.col(i + 1 + n_aug_)  = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  *Xsig_out = Xsig;

}

void UKF::SigmaPointPrediction(double delta_t, MatrixXd Xsig_aug) {
  // predict sigma points
  for (int i = 0; i< n_sig_; i++){

    // extract values for better readability
    double p_x      = Xsig_aug(0, i);
    double p_y      = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if(fabs(yawd) > EPSILON) {
        px_p = p_x + v / yawd * ( sin(yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v / yawd * ( cos(yaw) - cos(yaw + yawd * delta_t));
    }else{
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p    = v;
    double yaw_p  = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p    = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p    = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p     = v_p + nu_a * delta_t;

    yaw_p   = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p  = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

double UKF::NormalizeAngle(double angle) {
  while (angle >  M_PI) angle -= 2.0*M_PI;
  while (angle < -M_PI) angle += 2.0*M_PI;
  return angle;
}

void UKF::PredictMeanAndCovariance(){

  // Predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // Predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    x_diff(3) = UKF::NormalizeAngle(x_diff(3));
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // Predict Lidar measurement

  int n_z = 2;
  MatrixXd Zsig   = MatrixXd(n_z, n_sig_);
  VectorXd z_pred = VectorXd(n_z);
  MatrixXd S      = MatrixXd(n_z, n_z);

  z_pred.fill(0.0);
  S.fill(0.0);

  for(int i=0; i < n_sig_; i++){
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);
    z_pred += weights_(i) * Zsig.col(i);
  }

  for(int i = 0; i < n_sig_; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  S += R_;

  // Update State

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for(int i = 0; i < n_sig_; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = UKF::NormalizeAngle(x_diff(3));

    VectorXd z_diff = Zsig.col(i) - z_pred;
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // Residual of measurement to prediction
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // Update state and covariance
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  // NIS Update
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // Predict Radar measurement

  int n_z = 3;
  MatrixXd Zsig   = MatrixXd(n_z, n_sig_);
  VectorXd z_pred = VectorXd(n_z);
  MatrixXd S      = MatrixXd(n_z, n_z);

  z_pred.fill(0.0);
  S.fill(0.0);

  // Transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {
    double p_x  = Xsig_pred_(0, i);
    double p_y  = Xsig_pred_(1, i);
    double v    = Xsig_pred_(2, i);
    double yaw  = Xsig_pred_(3, i);
    double v1   = cos(yaw) * v;
    double v2   = sin(yaw) * v;
    if (fabs(p_x) < EPSILON) {
      p_x = EPSILON;
    }
    if (fabs(p_y) < EPSILON) {
      p_y = EPSILON;
    }

    // measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         //r
    Zsig(1, i) = atan2(p_y, p_x);                                     //phi
    // Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
    if(Zsig(0,i) > EPSILON){
      Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
      // Zsig(2,i) = (p_x*vx + p_y*vy)/sqrt(p_x*p_x + p_y*p_y);
    }else{
      Zsig(2, i) = 0.0;
    }

    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  for(int i = 0; i < n_sig_; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = NormalizeAngle(z_diff(1));
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  S += R_radar_;

  // Update State

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for(int i = 0; i < n_sig_; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = NormalizeAngle(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // Residual of measurement to prediction
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  z_diff(1) = NormalizeAngle(z_diff(1));

  // Update state and covariance
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  // NIS Update
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
