//
// Created by hoho on 2023/8/26.
//

#include "elas.h"

#include <vector>
#include <thread>
#include <iostream>
#include <ctime>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/viz3d.hpp>

#include <pcl/point_cloud.h>
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



void showPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud)
{
    pcl::visualization::PCLVisualizer visualizer("showcloud");
    visualizer.addPointCloud(pointcloud);
    visualizer.spin();
}

void showPointCloudVector(const std::vector<cv::Vec3f> &cloudpos, const std::vector<cv::Vec3b> &cloudcol)
{

    cv::viz::Viz3d window("showcloud");
    cv::viz::WCloud cloud_widget(cloudpos, cloudcol);
    window.showWidget("pointcloud", cloud_widget);
    window.spin();
}

void showMesh(const pcl::PolygonMesh triangles)
{
    // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPolygonMesh(triangles,"my");

    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    // 主循环
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        std::this_thread::sleep_for(chrono::milliseconds(1000));
        // boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    // Finish
}

pcl::PolygonMesh pointCloudTriangulation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud) {

    // Load input file into a PointCloud<T> with an appropriate type
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::copyPointCloud(*pointcloud, *cloud);

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
    // Additional vertex information
//	std::vector<int> parts = gp3.getPartIDs();//获得重建后每点的 ID, Parts 从 0 开始编号， a-1 表示未连接的点。
//	/*
//	获得重建后每点的状态，取值为 FREE 、 FRINGE 、 BOUNDARY 、 COMPLETED 、 NONE 常量，
//	其中 NONE 表示未定义，
//	FREE 表示该点没有在 三 角化后的拓扑内，为自由点，
//	COMPLETED 表示该点在三角化后的拓扑内，并且邻域都有拓扑点，
//	BOUNDARY 表示该点在三角化后的拓扑边缘，
//	FRINGE 表示该点在 三 角化后的拓扑内，其连接会产生重叠边。
//	*/
//	std::vector<int> states = gp3.getPointStates();
//
//
//	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZ>);
//	cloud_smoothed->points.resize(pointcloud->size());
//	for (size_t i = 0; i < pointcloud->points.size(); i++) {
//	     cloud_smoothed->points[i].x = pointcloud->points[i].x;
//	     cloud_smoothed->points[i].y = pointcloud->points[i].y;
//	     cloud_smoothed->points[i].z = pointcloud->points[i].z;
//	}
//	pcl::copyPointCloud(*pointcloud, *cloud_smoothed);
//
//	// 法线估计
//    pcl::NormalEstimation<PointXYZ,pcl::Normal> normalEstimation;             //创建法线估计的对象
//    normalEstimation.setInputCloud(cloud_smoothed);                         //输入点云
//    pcl::search::KdTree<PointXYZ>::Ptr tree(new pcl::search::KdTree<PointXYZ>);// 创建用于最近邻搜索的KD-Tree
//    normalEstimation.setSearchMethod(tree);
//    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>); // 定义输出的点云法线
//    // K近邻确定方法，使用k个最近点，或者确定一个以r为半径的圆内的点集来确定都可以，两者选1即可
//    normalEstimation.setKSearch(10);// 使用当前点周围最近的10个点
//    //normalEstimation.setRadiusSearch(0.03);            //对于每一个点都用半径为3cm的近邻搜索方式
//    normalEstimation.compute(*normals);               //计算法线
//    // 输出法线
//    std::cout<<"normals: "<<normals->size()<<", "<<"normals fields: "<<pcl::getFieldsList(*normals)<<std::endl;
//    // pcl::io::savePCDFileASCII("normals.pcd",*normals);
//
//	// 将点云位姿、颜色、法线信息连接到一起
//    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
//    pcl::concatenateFields(*cloud_smoothed, *normals, *cloud_with_normals);
//    // pcl::io::savePCDFileASCII("cloud_with_normals.pcd",*cloud_with_normals);
//
//	// 贪心投影三角化
//    //定义搜索树对象
//    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
//    tree2->setInputCloud(cloud_with_normals);
//
//    // 三角化
//    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;   // 定义三角化对象
//    pcl::PolygonMesh triangles; //存储最终三角化的网络模型

    // 设置三角化参数
//    gp3.setSearchRadius(0.5);  //设置搜索时的半径，也就是KNN的球半径
//    gp3.setMu (2.5);  //设置样本点搜索其近邻点的最远距离为2.5倍（典型值2.5-3），这样使得算法自适应点云密度的变化
//    gp3.setMaximumNearestNeighbors (100);    //设置样本点最多可搜索的邻域个数，典型值是50-100
//
//    gp3.setMinimumAngle(M_PI/18); // 设置三角化后得到的三角形内角的最小的角度为10°
//    gp3.setMaximumAngle(2*M_PI/3); // 设置三角化后得到的三角形内角的最大角度为120°
//
//    gp3.setMaximumSurfaceAngle(M_PI/4); // 设置某点法线方向偏离样本点法线的最大角度45°，如果超过，连接时不考虑该点
//    gp3.setNormalConsistency(false);  //设置该参数为true保证法线朝向一致，设置为false的话不会进行法线一致性检查
//
//    gp3.setInputCloud (cloud_with_normals);     //设置输入点云为有向点云
//    gp3.setSearchMethod (tree2);   //设置搜索方式
//    gp3.reconstruct (triangles);  //重建提取三角化


    return triangles;

}

int main()
{
    // 初始化图像与相机内参
    double f = 718.856, cx = 607.1928, cy = 185.2157; //相机内参
    double b = 0.573; //基线长度

    double LCamParam[] = {7.070493000000e+02,7.070493000000e+02,6.040814000000e+02,1.805066000000e+02};
    double RCamParam[] = {7.215377000000e+02,7.215377000000e+02,6.095593000000e+02,1.728540000000e+02};

    double Lfx = LCamParam[0];
    double Lfy = LCamParam[1];
    double Lcx = LCamParam[2];
    double Lcy = LCamParam[3];
    double Lk1 = LCamParam[4];
    double Lk2 = LCamParam[5];
    double Lp1 = LCamParam[6];
    double Lp2 = LCamParam[7];

    double Rfx = RCamParam[0];
    double Rfy = RCamParam[1];
    double Rcx = RCamParam[2];
    double Rcy = RCamParam[3];
    double Rk1 = RCamParam[4];
    double Rk2 = RCamParam[5];
    double Rp1 = RCamParam[6];
    double Rp2 = RCamParam[7];

    cv::Mat LcameraMatrix = (Mat_<double>(3, 3) << Lfx, 0, Lcx, 0, Lfy, Lcy, 0, 0, 1);
    cv::Mat LdistCoeffs = (Mat_<float>(4, 1) << Lk1, Lk2, Lp1, Lp2);
    cv::Mat RcameraMatrix = (Mat_<double>(3, 3) << Rfx, 0, Rcx, 0, Rfy, Rcy, 0, 0, 1);
    cv::Mat RdistCoeffs = (Mat_<float>(4, 1) << Rk1, Rk2, Rp1, Rp2);

    cv::Mat left_rgb = cv::imread("/home/hoho/SLAM-Recon/Ground-Reconstruction/dataset/kitti/image_2/000001-old.png");
    cv::Mat left = cv::imread("/home/hoho/SLAM-Recon/Ground-Reconstruction/dataset/kitti/image_2/000001-old.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread("/home/hoho/SLAM-Recon/Ground-Reconstruction/dataset/kitti/image_3/000001.png", cv::IMREAD_GRAYSCALE);

    if (left.empty() || right.empty())
        return -1;

    Mat left_img, right_img;
    left_img = left;
    right_img = right;
//	undistort(left, left_img, LcameraMatrix, LdistCoeffs);
//	undistort(right, right_img, RcameraMatrix, RdistCoeffs);
//	cv::imshow("left_img image", left_img);
//    cv::imshow("right_img image", left_img);
    // cv::imwrite("opencv_sgbm.png", disparity / 96);
    cv::waitKey(0);
    cv::destroyAllWindows();

    /************************************ OPencv 重建*************************************************/
    // Opencv 双目重建
    //下面是书上说的来自网络上的神奇参数
    // cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
    // cv::Mat disparity_sgbm, disparity;
    // sgbm->compute(left_img, right_img, disparity_sgbm); //计算两帧图像的视差
    // disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0);
    // return disparity

    // PCL 方法读取点云
//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//	for (int v = 0; v < left_img.rows; v++)
//	{
//		for (int u = 0; u < right_img.cols; u++)
//		{
//			if (disparity.at<float>(v, u) <= 10 || disparity.at<float>(v, u) >= 96) continue;
//			pcl::PointXYZRGB point;
//			double x = (u - cx) / f;
//			double y = (v - cy) / f;
//			double depth = f * b / (disparity.at<float>(v, u));
//			point.x = x * depth;
//			point.y = y * depth;
//			point.z = depth;
//			point.b = left_img.at<cv::Vec3b>(v, u)[0];
//			point.g = left_img.at<cv::Vec3b>(v, u)[1];
//			point.r = left_img.at<cv::Vec3b>(v, u)[2];
//			pointcloud->push_back(point);
//		}
//	}
//
//	std::cout << "show pointcloud !" <<std::endl;
//	showPointCloud(pointcloud);
//	cv::imshow("disparity image", disparity / 96);
//	// cv::imwrite("opencv_sgbm.png", disparity / 96);
//	cv::waitKey(0);
//	cv::destroyAllWindows();

    /************************************ ELAS 重建*************************************************/

    // ELAS 双目重建
    const Size imsize = left_img.size();
    const int32_t dims[3] = {imsize.width, imsize.height, imsize.width};
    Mat leftdpf = Mat::zeros(imsize, CV_32F);
    Mat rightdpf = Mat::zeros(imsize, CV_32F);

    Elas::parameters param(Elas::MIDDLEBURY);
    param.postprocess_only_left = true;
    Elas elas(param);
    elas.process(left_img.data, right_img.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
    Mat dmap = Mat(imsize, CV_32F, Scalar(0));
    leftdpf.convertTo(dmap, CV_32F, 1.);
    // return dmap;


    std::cout << "show pointcloud !" <<std::endl;
    //showPointCloud(pointcloud);
//	 cv::imshow("elas image", dmap / 96);
//	 //cv::imwrite("elas.png", dmap / 96);
//	 cv::waitKey(0);
//	 cv::destroyAllWindows();

    // PCL 方法读取点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int v = 0; v < left_img.rows; v++)
    {
        for (int u = 0; u < right_img.cols; u++)
        {
            if (dmap.at<float>(v, u) <= 10 || dmap.at<float>(v, u) >= 96) continue;
            pcl::PointXYZRGB point;
            double x = (u - cx) / f;
            double y = (v - cy) / f;
            double depth = f * b / (dmap.at<float>(v, u));
            point.x = x * depth;
            point.y = y * depth;
            point.z = depth;
            point.b = left_rgb.at<cv::Vec3b>(v, u)[0];
            point.g = left_rgb.at<cv::Vec3b>(v, u)[1];
            point.r = left_rgb.at<cv::Vec3b>(v, u)[2];
            pointcloud->push_back(point);
        }
    }

    auto start = chrono::steady_clock::now();
    std::cout << "start trianglation operation !" << endl;

    pcl::PolygonMesh triangles = pointCloudTriangulation(pointcloud);

    auto end = chrono::steady_clock::now();

    auto time_diff = end - start;
    auto duration = chrono::duration_cast<chrono::seconds>(time_diff);
    cout << "Operation cost : " << duration.count() << "s" << endl;

    std::cout << "show triangles mesh !" <<std::endl;

    showMesh(triangles);

    // vector 方法读取点云
    /*std::vector<cv::Vec3f> pointcloud;
    std::vector<cv::Vec3b> pointcolor;
    for (int v = 0; v < left_img.rows; v++)
    {
        for (int u = 0; u < left_img.cols; u++)
        {
            if (disparity.at<float>(v, u) <= 10 || disparity.at<float>(v, u) >= 96) continue;
            cv::Vec3f pos;
            cv::Vec3b col;

            double x = (u - cx) / f;
            double y = (v - cy) / f;
            double depth = f * b / (disparity.at<float>(v, u));
            pos[0] = x * depth;
            pos[1] = y * depth;
            pos[2] = depth;
            col = left_img.at<cv::Vec3b>(v, u) / 255;
            pointcloud.emplace_back(pos);
            pointcolor.emplace_back(col);
        }
    }
    showPointCloudVector(pointcloud, pointcolor);*/


//	showPointCloud(pointcloud);
//	pcl::PolygonMesh triangles = pointCloudTriangulation(pointcloud);
//	showMesh(triangles);



    return 0;
}
