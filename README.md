# Deep learning calibration of industrial robots

Robot position accuracy plays a very important role in advanced industrial applications, nowadays, most of the industrial robots have excellent repeatability, however, it still always remain some absolute position error that are due to non geometric calibration parameters that are hard to model and identify. The present project studies a method to reduce the absolute position error of robots using conventional identification procedures as well as neural networks.

In order to increase the robot accuracy, we use deep learning based methods to identify the non-geometric error sources such as link compliance, gear backlash, and others, which are difficult to model correctly and completely. The algorithm is tested on simulation with the UR-10 robot

<!-- ![UR10](/images/UR10.png) -->
<p align="center">
  <img src="https://github.com/samysellami/Deep_Sim2Real/blob/master/images/UR10.png" />
</p>

# Download & Setup Instructions

-   1 - Clone project: git clone https://github.com/samysellami/Deep_Sim2Real
-   2 - cd Deep_Sim2Real/
-   3 - Create virtual environment: virtualenv myenv
-   4 - source myenv\bin\activate
-   5 - pip install -e .

# Training and testing the model

-   1 - To train the model run:
    ```
    deep-calibration run --config calib_config.ini --algo sac --load-best
    ```
-   2 - To test the resulting model run:
    ```
    deep-calibration train --config calib_config.ini --algo sac
    ```
