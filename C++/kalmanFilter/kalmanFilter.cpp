#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include <iostream>
#include <chrono>
#include <random>
#include <math.h>

using namespace Eigen;
using namespace std;


class KF{
public:

  float dt;
  float initial_x;
  float initial_v;
  float accel_variance;
  
  MatrixXf x = MatrixXf(2,1);
  MatrixXf P = MatrixXf(2,2);
  MatrixXf G = MatrixXf(2,1);
  MatrixXf F = MatrixXf(2,2);
  MatrixXf H = MatrixXf(1,2);
  MatrixXf z = MatrixXf(1,1);
  MatrixXf R = MatrixXf(1,1);
  MatrixXf y = MatrixXf(1,1);
  MatrixXf S = MatrixXf(1,1);
  MatrixXf K = MatrixXf(2,1);
  MatrixXf I = MatrixXf(2,2);



  KF(float initial_x, float initial_v, float accel_variance){

    x << initial_x,initial_v;

    P << 1,0,
         0,1;
    this->accel_variance = accel_variance;

    I << 1,0,0,1;

  }

  void predict(float dt){
    MatrixXf F = MatrixXf(2,2);
    F << 1,dt,
         0,1;

    MatrixXf new_x =  F*x;
    G << 0.5*dt*dt,dt;
    MatrixXf new_P = F*P*F.transpose() + G*G.transpose()*accel_variance;

    x = new_x;
    P = new_P;
  }


  void update(float meas_value, float meas_variance){
    H << 1,0;
    z << meas_value;
    R << meas_variance;
    y = z - H*x;
    S = H*P*H.transpose() + R;
    K = P*H.transpose()*S.inverse();

    x = x + K*y;
    P = (I-K*H)*P;

  }

  void print(){
    cout << x << endl;
    cout << P << endl;
  }

};





 
int main()
{
  
  float real_x = 0;
  float real_v = 0.9;
  float dt = 0.1;

  KF kalman(10,2,0.9);

  vector<float> kalman_position;
  vector<float> kalman_velocity;
  vector<float> measured_position;

  int NUM_STEPS = 1000;

  for (int i=0; i<NUM_STEPS; i++){
    kalman_position.push_back(kalman.x(0,0));
    kalman_velocity.push_back(kalman.x(1,0));

    real_x = real_x + dt*real_v;

    kalman.predict(dt);

    float meas_variance = 0.5;
    // normally distributed random error:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<float> distribution (0.0,1.0);
    float error = distribution(generator);

    float meas_value = real_x + error*sqrt(meas_variance);
    measured_position.push_back(meas_value);

    kalman.update(meas_value,meas_variance);
  }
  

  for (int i=0; i<50; i++){
    cout << "Measured Value:" << measured_position[i] << endl;
    cout << "Kalman filter output: " << kalman_position[i] << endl;
    cout << "******************" << endl;
  }




    

  return 0;
}