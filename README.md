# elas-road-reconstruct
this is a No-ROS version dense reconstruction project, modified from [stereo_dense_reconstruction](https://github.com/umass-amrl/stereo_dense_reconstruction) project, thanks the original author.

## Run
* requirements: PCL and Opencv

* change data path in `src/elas_reconstruction.cpp` as your path

```shell
cd /path/to/project
mkdir build && cd build
cmake ..
make -j8
```

## Visualization 

### disparity map

* opencv SGBM
  
![SGBM]("./screenshot/disparity image_screenshot_01.07.2023.png")

* ELAS
  
![SGBM]("./screenshot/elas image_screenshot_01.07.2023.png")


### dense recon

![Kitti dense recon]("./screenshot/screenshot-1688196678.png")
![Kitti dense recon2]("./screenshot/screenshot-1688224255.png")


