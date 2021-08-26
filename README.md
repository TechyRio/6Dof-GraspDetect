# Grasp Pose Detection (GPD)
    主要参考Andreas ten Pas的工作，针对未知物体的抓取检测，通过改进GPD抓取算法，在6DOF自由度下识别物体的抓取位姿。抓取检测精度提高2%，且参数降为原来的一半，使用OpenVino工具将算法部署到X86架构中，检测速度从8s减小到1s以内，满足实时抓取需求。


## 1) 依赖

1. [PCL 1.9 or newer](http://pointclouds.org/)
2. [Eigen 3.0 or newer](https://eigen.tuxfamily.org)
3. [OpenCV 3.4 or newer](https://opencv.org)


## 2)安装

在**Ubuntu 18.04**下测试通过。

1. 安装 PCL、Eigen 、  安装 OpenCV 3.4 

2. 编译包:

   ```
   cd gpd
   mkdir build && cd build
   cmake ..
   make -j
   ```

## 11)参考文献

[1] Andreas ten Pas, Marcus Gualtieri, Kate Saenko, and Robert Platt. [**Grasp
Pose Detection in Point Clouds**](http://arxiv.org/abs/1706.09911). The
International Journal of Robotics Research, Vol 36, Issue 13-14, pp. 1455-1473.
October 2017.

[2] Marcus Gualtieri, Andreas ten Pas, Kate Saenko, and Robert Platt. [**High
precision grasp pose detection in dense
clutter**](http://arxiv.org/abs/1603.01564). IROS 2016, pp. 598-605.

