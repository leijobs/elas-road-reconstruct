#include "elas.h"

#include <vector>
#include <thread>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/viz/viz3d.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
// #include <boost/thread/thread.hpp>

using namespace cv;
using namespace pcl;
using namespace std;

// 1. load stereo camera and get disparity map
// 2. disparity map into point cloud
// 3. point cloud triangulation and vis


void showPointCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor(0.0,0.0,0.0,0);
	viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();
	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);
		std::this_thread::sleep_for(chrono::milliseconds(1000));
	}
}


void showPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pointcloud)
{
	//创建3D窗口并添加点云
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("PCLVisualizer", false));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pointcloud); //using PCL rgb or rgba
	viewer->addPointCloud<pcl::PointXYZRGB>(pointcloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();


    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        std::this_thread::sleep_for(chrono::milliseconds(1000));
    }
}

void showMesh(const pcl::PolygonMesh triangles)
{
	std::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
	viewer->addPolygonMesh(triangles,"my");
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters ();

	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);
		std::this_thread::sleep_for(chrono::milliseconds(1000));
	}
}

pcl::PolygonMesh pointCloudTriangulation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud) {

    // Load input file into a PointCloud<T> with an appropriate type
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::copyPointCloud(*pointcloud, *cloud);

    // Normal estimation*
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;//
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);//save for normal vector
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);// create tree for point cloud
    tree->setInputCloud (cloud);//convert point cloud into tree
    n.setInputCloud (cloud);//get point cloud for normal compute
    n.setSearchMethod (tree);
    n.setKSearch (20);//set KNN search range limit
    n.compute (*normals);//estimate normal vector
    //* normals should not contain the point normals + surface curvatures

    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);//
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);//cloud_with_normals
    //* cloud_with_normals = cloud + normals

    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);//create kdtree
    tree2->setInputCloud (cloud_with_normals);//set tree

    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;//define triangulation
    pcl::PolygonMesh triangles;//set mesh for triangles

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.025);         //set radius

    // Set typical values for the parameters
    gp3.setMu (2.5);                     //mu for max search distance
    gp3.setMaximumNearestNeighbors (100);//max neighbor 100 。
    gp3.setMaximumSurfaceAngle(M_PI/4);  //45 degrees，max surface angle
    gp3.setMinimumAngle(M_PI/18);        //10 degrees， minimum_angle
    gp3.setMaximumAngle(2*M_PI/3);       //120 degrees， maximum_angle
    gp3.setNormalConsistency(false);     //set consistent ，whether to check normal directions

    // Get result
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);           //reconstruct

    return triangles;
}


int main()
{
    // work model
    int model; // 1：elas 2:opencv
    cout << "0 : test pcd" << " 1 : elas, " << " 2: opencv" << endl;
    cin >> model;
    // 初始化图像与相机内参
    double f = 718.856, cx = 607.1928, cy = 185.2157; //相机内参
    double b = 0.573; //基线长度

	string test_pcd = "/home/hoho/JustDoit/stereo_dense_reconstruction/dataset/table.pcd";

    Mat left_rgb = imread("/home/hoho/JustDoit/stereo_dense_reconstruction/dataset/kitti/image_2/000001-old.png");
    Mat left = imread("/home/hoho/JustDoit/stereo_dense_reconstruction/dataset/kitti/image_2/000001-old.png", cv::IMREAD_GRAYSCALE);
    Mat right = imread("/home/hoho/JustDoit/stereo_dense_reconstruction/dataset/kitti/image_3/000001.png", cv::IMREAD_GRAYSCALE);

    if (left.empty() || right.empty()){
		cout << "input image is empty, check input " << endl;
		return -1;
	}


    Mat left_img = left;
    Mat right_img = right;
    cout << "processing .." << endl;

	/************************************ Opencv reconstruct *************************************************/

	switch(model) {
		case 1: {
			//	default parameters
			cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
			cv::Mat disparity_sgbm, disparity;
			sgbm->compute(left_img, right_img, disparity_sgbm); //get disparity
			disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0);

			cv::imshow("disparity image", disparity / 96);
			// cv::imwrite("opencv_sgbm.png", disparity / 96);
			cv::waitKey(0);
			cv::destroyAllWindows();

			// PCL read point cloud
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			for (int v = 0; v < left_img.rows; v++) {
				for (int u = 0; u < right_img.cols; u++) {
					if (disparity.at<float>(v, u) <= 10 || disparity.at<float>(v, u) >= 96) continue;
					pcl::PointXYZRGB point;
					double x = (u - cx) / f;
					double y = (v - cy) / f;
					double depth = f * b / (disparity.at<float>(v, u));
					point.x = x * depth;
					point.y = y * depth;
					point.z = depth;
					point.b = left_rgb.at<cv::Vec3b>(v, u)[0];
					point.g = left_rgb.at<cv::Vec3b>(v, u)[1];
					point.r = left_rgb.at<cv::Vec3b>(v, u)[2];
					pointcloud->push_back(point);
				}
			}

			std::cout << "show pointcloud !" << std::endl;
			showPointCloud(pointcloud);

		}
		case 2: {
			// ELAS stereo recon
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

			std::cout << "show pointcloud !" << std::endl;
			//showPointCloud(pointcloud);
			cv::imshow("elas image", dmap / 96);
			//cv::imwrite("elas.png", dmap / 96);
			cv::waitKey(0);
			cv::destroyAllWindows();

			// PCL read point cloud
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			for (int v = 0; v < left_img.rows; v++) {
				for (int u = 0; u < right_img.cols; u++) {
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

			std::cout << "show triangles mesh !" << std::endl;

			showMesh(triangles);
		}
		default: {
			std::cout << "test show pointcloud !" << std::endl;

			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
					new pcl::PointCloud<pcl::PointXYZ>); // create PointCloud<PointXYZ> shared pointer
			pcl::io::loadPCDFile(test_pcd, *cloud); // load point cloud

			showPointCloud(cloud); // vis

		}
	}
	return 0;
}
