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
