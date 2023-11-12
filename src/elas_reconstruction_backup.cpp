#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "elas.h"
#include "popt_pp.h"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace cv;
using namespace pcl;
using namespace std;

// 1. 加载双目图像及其参数
// 2. 处理生成视差图
// 3. 视差图转化为PCL点云
// 4. 点云三角化与可视化

Mat XR, XT, Q, P1, P2;
Mat R1, R2, K1, K2, D1, D2, R;
Mat lmapx, lmapy, rmapx, rmapy;
Vec3d T;

FileStorage calib_file;
int debug = 0;
Size out_img_size;
Size calib_img_size;

Mat composeRotationCamToRobot(float x, float y, float z) {
  Mat X = Mat::eye(3, 3, CV_64FC1);
  Mat Y = Mat::eye(3, 3, CV_64FC1);
  Mat Z = Mat::eye(3, 3, CV_64FC1);
  
  X.at<double>(1,1) = cos(x);
  X.at<double>(1,2) = -sin(x);
  X.at<double>(2,1) = sin(x);
  X.at<double>(2,2) = cos(x);

  Y.at<double>(0,0) = cos(y);
  Y.at<double>(0,2) = sin(y);
  Y.at<double>(2,0) = -sin(y);
  Y.at<double>(2,2) = cos(y);

  Z.at<double>(0,0) = cos(z);
  Z.at<double>(0,1) = -sin(z);
  Z.at<double>(1,0) = sin(z);
  Z.at<double>(1,1) = cos(z);
  
  return Z*Y*X;
}

Mat composeTranslationCamToRobot(float x, float y, float z) {
  return (Mat_<double>(3,1) << x, y, z);
}

void generatePointCloud(){

  // Load input file into a PointCloud<T> with an appropriate type
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCLPointCloud2 cloud_blob;
  pcl::io::loadPCDFile ("..\\..\\source\\table_scene_lms400_downsampled.pcd", cloud_blob);
  pcl::fromPCLPointCloud2(cloud_blob, *cloud);
  //* the data should be available in cloud
 
  // Normal estimation*
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;//设置法线估计对象
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);//存储估计的法线
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);//定义kd树指针
  tree->setInputCloud (cloud);//用cloud构造tree对象
  n.setInputCloud (cloud);//为法线估计对象设置输入点云
  n.setSearchMethod (tree);//设置搜索方法
  n.setKSearch (20);//设置k邻域搜素的搜索范围
  n.compute (*normals);//估计法线
  //* normals should not contain the point normals + surface curvatures
 
  // Concatenate the XYZ and normal fields*
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);//
  pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);//连接字段，cloud_with_normals存储有向点云
  //* cloud_with_normals = cloud + normals
 
  // Create search tree*
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);//定义搜索树对象
  tree2->setInputCloud (cloud_with_normals);//利用有向点云构造tree
 
  // Initialize objects
  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;//定义三角化对象
  pcl::PolygonMesh triangles;//存储最终三角化的网络模型
 
  // Set the maximum distance between connected points (maximum edge length)
  gp3.setSearchRadius (0.025);         //设置搜索半径radius，来确定三角化时k一邻近的球半径。
 
  // Set typical values for the parameters
  gp3.setMu (2.5);                     //设置样本点到最近邻域距离的乘积系数 mu 来获得每个样本点的最大搜索距离，这样使得算法自适应点云密度的变化
  gp3.setMaximumNearestNeighbors (100);//设置样本点最多可以搜索的邻域数目100 。
  gp3.setMaximumSurfaceAngle(M_PI/4);  //45 degrees，设置连接时的最大角度 eps_angle ，当某点法线相对于采样点的法线偏离角度超过该最大角度时，连接时就不考虑该点。
  gp3.setMinimumAngle(M_PI/18);        //10 degrees，设置三角化后三角形的最小角，参数 minimum_angle 为最小角的值。
  gp3.setMaximumAngle(2*M_PI/3);       //120 degrees，设置三角化后三角形的最大角，参数 maximum_angle 为最大角的值。
  gp3.setNormalConsistency(false);     //设置一个标志 consistent ，来保证法线朝向一致，如果设置为 true 则会使得算法保持法线方向一致，如果为 false 算法则不会进行法线一致性检查。
 
  // Get result
  gp3.setInputCloud (cloud_with_normals);//设置输入点云为有向点云
  gp3.setSearchMethod (tree2);           //设置搜索方式tree2
  gp3.reconstruct (triangles);           //重建提取三角化
 // std::cout << triangles;
  // Additional vertex information
  std::vector<int> parts = gp3.getPartIDs();//获得重建后每点的 ID, Parts 从 0 开始编号， a-1 表示未连接的点。
  /*
  获得重建后每点的状态，取值为 FREE 、 FRINGE 、 BOUNDARY 、 COMPLETED 、 NONE 常量，
  其中 NONE 表示未定义， 
  FREE 表示该点没有在 三 角化后的拓扑内，为自由点， 
  COMPLETED 表示该点在三角化后的拓扑内，并且邻域都有拓扑点，
  BOUNDARY 表示该点在三角化后的拓扑边缘， 
  FRINGE 表示该点在 三 角化后的拓扑内，其连接会产生重叠边。
  */
  std::vector<int> states = gp3.getPointStates();
 
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPolygonMesh(triangles,"my");
 
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
   // 主循环
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
  // Finish
  return (0);

}


void publishPointCloud(Mat& img_left, Mat& dmap) {
  if (debug == 1) {
    XR = composeRotationCamToRobot(config.PHI_X,config.PHI_Y,config.PHI_Z);
    XT = composeTranslationCamToRobot(config.TRANS_X,config.TRANS_Y,config.TRANS_Z);
    cout << "Rotation matrix: " << XR << endl;
    cout << "Translation matrix: " << XT << endl;
  }
  Mat V = Mat(4, 1, CV_64FC1);
  Mat pos = Mat(4, 1, CV_64FC1);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      int d = dmap.at<uchar>(j,i);
      // if low disparity, then ignore
      if (d < 2) {
        continue;
      }
      // V is the vector to be multiplied to Q to get
      // the 3D homogenous coordinates of the image point
      V.at<double>(0,0) = (double)(i);
      V.at<double>(1,0) = (double)(j);
      V.at<double>(2,0) = (double)d;
      V.at<double>(3,0) = 1.;
      pos = Q * V; // 3D homogeneous coordinate
      double X = pos.at<double>(0,0) / pos.at<double>(3,0);
      double Y = pos.at<double>(1,0) / pos.at<double>(3,0);
      double Z = pos.at<double>(2,0) / pos.at<double>(3,0);
      Mat point3d_cam = Mat(3, 1, CV_64FC1);
      point3d_cam.at<double>(0,0) = X;
      point3d_cam.at<double>(1,0) = Y;
      point3d_cam.at<double>(2,0) = Z;
      // transform 3D point from camera frame to robot frame
      Mat point3d_robot = XR * point3d_cam + XT;

      for (int i = 0; i < cloud->points.size(); ++i) # points 数量未知
      {
        cloud->points[i].x = point3d_robot.at<double>(0,0);
        cloud->points[i].y = point3d_robot.at<double>(0,0);
        cloud->points[i].z = point3d_robot.at<double>(0,0);
      }

      
      int32_t red, blue, green;
      red = img_left.at<Vec3b>(j,i)[2];
      green = img_left.at<Vec3b>(j,i)[1];
      blue = img_left.at<Vec3b>(j,i)[0];
      int32_t rgb = (red << 16 | green << 8 | blue);
      ch.values.push_back(*reinterpret_cast<float*>(&rgb));
    }
  }
  if (!dmap.empty()) {
    Mat disp_img =  dmap;
  }
}

Mat generateDisparityMap(Mat& left, Mat& right) {
  if (left.empty() || right.empty()) 
    return left;
  const Size imsize = left.size();
  const int32_t dims[3] = {imsize.width, imsize.height, imsize.width};
  Mat leftdpf = Mat::zeros(imsize, CV_32F);
  Mat rightdpf = Mat::zeros(imsize, CV_32F);

  Elas::parameters param(Elas::MIDDLEBURY);
  param.postprocess_only_left = true;
  Elas elas(param);
  elas.process(left.data, right.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
  Mat dmap = Mat(out_img_size, CV_8UC1, Scalar(0));
  leftdpf.convertTo(dmap, CV_8U, 1.);
  return dmap;
}

void imgCallback(const Mat& img_left, const Mat& img_right) {
  Mat tmpL = img_left;
  Mat tmpR = img_right;
  if (tmpL.empty() || tmpR.empty())
    return;
  
  Mat img_left, img_right, img_left_color;
  remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR);
  remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
  
  cvtColor(img_left, img_left_color, CV_GRAY2BGR);
  
  Mat dmap = generateDisparityMap(img_left, img_right);
  publishPointCloud(img_left_color, dmap);
  
  imshow("LEFT", img_left);
  imshow("RIGHT", img_right);
  imshow("DISP", dmap);
  waitKey(30);
}

void findRectificationMap(FileStorage& calib_file, Size finalSize) {
  Rect validRoi[2];
  cout << "Starting rectification" << endl;
  stereoRectify(K1, D1, K2, D2, calib_img_size, R, Mat(T), R1, R2, P1, P2, Q, 
                CV_CALIB_ZERO_DISPARITY, 0, finalSize, &validRoi[0], &validRoi[1]);
  cv::initUndistortRectifyMap(K1, D1, R1, P1, finalSize, CV_32F, lmapx, lmapy);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, finalSize, CV_32F, rmapx, rmapy);
  cout << "Done rectification" << endl;
}

void paramsCallback(stereo_dense_reconstruction::CamToRobotCalibParamsConfig &conf, uint32_t level) {
  config = conf;
}

int main(int argc, char** argv) {
  
  const char* left_img_topic;
  const char* right_img_topic;
  const char* calib_file_name;
  int calib_width, calib_height, out_width, out_height;
  
  static struct poptOption options[] = {
    { "left_topic",'l',POPT_ARG_STRING,&left_img_topic,0,"Left image topic name","STR" },
    { "right_topic",'r',POPT_ARG_STRING,&right_img_topic,0,"Right image topic name","STR" },
    { "calib_file",'c',POPT_ARG_STRING,&calib_file_name,0,"Stereo calibration file name","STR" },
    { "calib_width",'w',POPT_ARG_INT,&calib_width,0,"Calibration image width","NUM" },
    { "calib_height",'h',POPT_ARG_INT,&calib_height,0,"Calibration image height","NUM" },
    { "out_width",'u',POPT_ARG_INT,&out_width,0,"Rectified image width","NUM" },
    { "out_height",'v',POPT_ARG_INT,&out_height,0,"Rectified image height","NUM" },
    { "debug",'d',POPT_ARG_INT,&debug,0,"Set d=1 for cam to robot frame calibration","NUM" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}
  
  calib_img_size = Size(calib_width, calib_height);
  out_img_size = Size(out_width, out_height);
  
  calib_file = FileStorage(calib_file_name, FileStorage::READ);
  calib_file["K1"] >> K1;
  calib_file["K2"] >> K2;
  calib_file["D1"] >> D1;
  calib_file["D2"] >> D2;
  calib_file["R"] >> R;
  calib_file["T"] >> T;
  calib_file["XR"] >> XR;
  calib_file["XT"] >> XT;
  
  findRectificationMap(calib_file, out_img_size);
  
  return 0;
}