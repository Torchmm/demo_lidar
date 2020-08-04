#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>

#include "cameraParameters.h"
#include "pointDefinition.h"

using namespace std;
using namespace cv;

bool systemInited = false;
double timeCur, timeLast;

const int imagePixelNum = imageHeight * imageWidth;
CvSize imgSize = cvSize(imageWidth, imageHeight);

IplImage *imageCur = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
IplImage *imageLast = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);

int showCount = 0;
const int showSkipNum = 2;
const int showDSRate = 2;
CvSize showSize = cvSize(imageWidth / showDSRate, imageHeight / showDSRate);

IplImage *imageShow = cvCreateImage(showSize, IPL_DEPTH_8U, 1);
IplImage *harrisLast = cvCreateImage(showSize, IPL_DEPTH_32F, 1);

CvMat kMat = cvMat(3, 3, CV_64FC1, kImage);
CvMat dMat = cvMat(4, 1, CV_64FC1, dImage);

IplImage *mapx, *mapy;

//整个图像划分为12x8 = 96个小区域，每个区域最多2个特征，一共最多96x2 = 192个特征
// 划分区域使得特征点分布均匀
const int maxFeatureNumPerSubregion = 2;
const int xSubregionNum = 12;
const int ySubregionNum = 8;
const int totalSubregionNum = xSubregionNum * ySubregionNum; //96
const int MAXFEATURENUM = maxFeatureNumPerSubregion * totalSubregionNum;// 12*8*2=192

const int xBoundary = 20;
const int yBoundary = 20;
const double subregionWidth = (double)(imageWidth - 2 * xBoundary) / (double)xSubregionNum;
const double subregionHeight = (double)(imageHeight - 2 * yBoundary) / (double)ySubregionNum;

const double maxTrackDis = 100; // 跟踪到的点距离不超过100个像素
const int winSize = 15;

IplImage *imageEig, *imageTmp, *pyrCur, *pyrLast;

CvPoint2D32f *featuresCur = new CvPoint2D32f[2 * MAXFEATURENUM]; // 保留了上一帧和当前帧光流法追踪的特征点
CvPoint2D32f *featuresLast = new CvPoint2D32f[2 * MAXFEATURENUM]; // 保留了上帧光流法和上一帧harris检测的特征点
char featuresFound[2 * MAXFEATURENUM];//
float featuresError[2 * MAXFEATURENUM];

int featuresIndFromStart = 0; // 这个值一值在累加
int featuresInd[2 * MAXFEATURENUM] = {0};// 储存了光流法追踪到的特征点和下一帧来时检测到的特征点索引

int totalFeatureNum = 0;
//int subregionFeatureNum[2 * totalSubregionNum] = {0};// 子区域特征点数，2*总的子区域数
int subregionFeatureNum[totalSubregionNum] = {0};// 子区域特征点数，2*总的子区域数

pcl::PointCloud<ImagePoint>::Ptr imagePointsCur(new pcl::PointCloud<ImagePoint>());
pcl::PointCloud<ImagePoint>::Ptr imagePointsLast(new pcl::PointCloud<ImagePoint>());

ros::Publisher *imagePointsLastPubPointer;
ros::Publisher *imageShowPubPointer;
cv_bridge::CvImage bridge;

void imageDataHandler(const sensor_msgs::Image::ConstPtr& imageData) 
{
  timeLast = timeCur;
  timeCur = imageData->header.stamp.toSec() - 0.1163;

  IplImage *imageTemp = imageLast;
  imageLast = imageCur;
  imageCur = imageTemp;

  for (int i = 0; i < imagePixelNum; i++) {
    imageCur->imageData[i] = (char)imageData->data[i]; //char 的范围是-128~127
  }

  IplImage *t = cvCloneImage(imageCur); // 744x480

  /*
   * 图像重映射
   * t 输入图像
   * imageCur 输出图像
   * mapx x坐标对应的重映射
   * mapy y坐标对应的重映射
   */

  cvRemap(t, imageCur, mapx, mapy);//矫正畸变图像
//     cvShowImage("t",t);
//    cvShowImage("ss",imageCur);
//    cvWaitKey(0);

  //cvEqualizeHist(imageCur, imageCur);

  // 释放t的内存
  cvReleaseImage(&t);

  cvResize(imageLast, imageShow);

  /*
   * 检测图像的哈里斯角点，判断出某一点是不是图像的角点
   * imageshow 输入图像
   * harrisLast 存储哈里斯角点检测responces的图像，与输入图像一样大
   * 3 邻域大小
   */
  cvCornerHarris(imageShow, harrisLast, 3);

//    cvShowImage("as",harrisLast);
//    cvShowImage("aSSs",imageLast);
//    cvShowImage("ASDA",imageShow);
//    cvWaitKey(0);
  CvPoint2D32f *featuresTemp = featuresLast;
  featuresLast = featuresCur;
  featuresCur = featuresTemp;

  pcl::PointCloud<ImagePoint>::Ptr imagePointsTemp = imagePointsLast;
  imagePointsLast = imagePointsCur;
  imagePointsCur = imagePointsTemp;
  imagePointsCur->clear();

  if (!systemInited) {
    systemInited = true;
    return;
  }

//    for (int l = 0; l < totalFeatureNum; ++l) {
//        //cout << "x: " <<featuresLast[l].x << "y: " <<featuresLast[l].y <<  endl;
//        cvDrawCircle(imageLast,cvPoint(featuresLast[l].x ,featuresLast[l].y),2,(255,0,0),-1);
//    }
//    cvShowImage("ssd",imageLast);
//    cvWaitKey(0);

  //对每个划分出来的区域查找特征点,光流找到的特征点
  int recordFeatureNum = totalFeatureNum;

  for (int i = 0; i < ySubregionNum; i++)
  {
    for (int j = 0; j < xSubregionNum; j++)
    {
      int ind = xSubregionNum * i + j;
      int numToFind = maxFeatureNumPerSubregion - subregionFeatureNum[ind];

      if (numToFind > 0) {
          // 划分的小区域
        int subregionLeft = xBoundary + (int)(subregionWidth * j);
        int subregionTop = yBoundary + (int)(subregionHeight * i);
        CvRect subregion = cvRect(subregionLeft, subregionTop, (int)subregionWidth, (int)subregionHeight);

        // 基于给定的矩形设置图像的ROI,显示image图像是只显示ROI标识的一部分，即改变了指针image，
        // 但是它仍旧保留有原来图像的信息，在执行这一句cvResetImageROI(image);之后，image指示原来的图像信息。

        cvSetImageROI(imageLast, subregion);

        /*
         * 确定上一帧每个划分小区域的的强角点
         * imageLast 输入图像，8-位或者浮点32-比特，单通道
         * imageEig 临时浮点32-位图像，尺寸与输入图像一致
         * imageTmp 另外一个临时图像，格式尺寸和imageEig一致
         * featureLast+totalFeatureNum
         * numToFind 角点数
         * 0.1 质量等级
         * 5.0 最小距离，像素单位
         * Null 图像mask
         * 3 块大小blocksize
         * 1 true 使用哈尔检测算法
         * 0.04 哈尔算法参数，一般默认0.04
         */

        cvGoodFeaturesToTrack(imageLast, imageEig, imageTmp, featuresLast + totalFeatureNum,
                              &numToFind, 0.1, 5.0, NULL, 3, 1, 0.04);

        int numFound = 0;
        for(int k = 0; k < numToFind; k++)
        {
            //修正找到的角点的图像坐标，从局部子区域坐标修正到整个图像的坐标.
            // totalFeatureNum是上一帧光流法找到的特征点.
          featuresLast[totalFeatureNum + k].x += subregionLeft;
          featuresLast[totalFeatureNum + k].y += subregionTop;

          int xInd = (featuresLast[totalFeatureNum + k].x + 0.5) / showDSRate;
          int yInd = (featuresLast[totalFeatureNum + k].y + 0.5) / showDSRate;

          // 如果强角点也是harris角点？
          //cout << ((float*)(harrisLast->imageData + harrisLast->widthStep * yInd))[xInd] << endl;

          if (((float*)(harrisLast->imageData + harrisLast->widthStep * yInd))[xInd] > 1e-7)
          {
            featuresLast[totalFeatureNum + numFound].x = featuresLast[totalFeatureNum + k].x;
            featuresLast[totalFeatureNum + numFound].y = featuresLast[totalFeatureNum + k].y;
            featuresInd[totalFeatureNum + numFound] = featuresIndFromStart;// 储存了特征点的序列号

            numFound++;
            featuresIndFromStart++;
          }
        }
        totalFeatureNum += numFound;// 原来是光流法追踪到的上一帧特征点，加上harris检测的特征点
        subregionFeatureNum[ind] += numFound;// 当前子区域找到的强角点数

        // 恢复Image指示的图像
        cvResetImageROI(imageLast);
      }
    }
  }

  //torchm

//    for (int l = 0; l < totalFeatureNum; ++l) {
//        //cout << "x: " <<featuresLast[l].x << "y: " <<featuresLast[l].y <<  endl;
//        cvDrawCircle(imageLast,cvPoint(featuresLast[l].x ,featuresLast[l].y),3,(255,0,0),-1);
//    }
//    cvShowImage("ssd",imageLast);
//    cvWaitKey(0);
    //torchm

  /*
   * 计算一个稀疏特征集的光流，使用金字塔中的迭代Lucas-Kanade方法
   * imageLast 在时间t的第一帧
   * imageCur 在时间t+dt的第一帧
   * pyrLast 第一帧的金字塔缓存，如何指针非null,则缓存必须有足够的空间来储存金字塔从层1到层#level的内存。
   *         尺寸(image_width+8)*image_height/3足够。
   * pyrCur 与pyrLast类似，用于第二帧
   * featureLast 需要发现光流的点集
   * featureCur 包含新计算出来的位置的点集
   * totalFeatureNum 特征点的数目
   * cvSize(winSize,winSize) 每个金字塔的搜索窗口尺寸
   * 3 最大金字塔的层数，如果为0，不适用金字塔，即金字塔为单层，如果为1，使用两层，下面一次类推。
   * featureFound 数组，如果对应特征的光流被发现，数组中的每一个元素都被设置为1，否则设置为0
   * featureError 双精度数组，包含原始图像碎片与移动点之间的差，为可选参数，可以是null.
   * cvTermCriteria 终止准则，指定在每个金字塔层，为某点寻找光流的 迭代过程的终止条件
   * 0 根据我们的金字塔是否建立了，设置不同的值，一般在第一次使用0，在一帧处理完保留当前金字塔为前一帧金字塔，下次处理时可以直接使用
   */

  cvCalcOpticalFlowPyrLK(imageLast, imageCur, pyrLast, pyrCur,
                         featuresLast, featuresCur, totalFeatureNum, cvSize(winSize, winSize),
                         3, featuresFound, featuresError,
                         cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01), 0);


  // 上上帧的特征点数置0
  for (int i = 0; i < totalSubregionNum; i++) {
    subregionFeatureNum[i] = 0;
  }

  ImagePoint point;
  int featureCount = 0; // 当前帧图像光流检测到的特征点数
  double meanShiftX = 0, meanShiftY = 0;
  for (int i = 0; i < totalFeatureNum; i++)// totalFeatureNUm ,这里的totalFeatureNum 包括上一帧和上上帧的特征点数
  {
    double trackDis = sqrt((featuresLast[i].x - featuresCur[i].x) 
                    * (featuresLast[i].x - featuresCur[i].x)
                    + (featuresLast[i].y - featuresCur[i].y) 
                    * (featuresLast[i].y - featuresCur[i].y));

    // 如果提取追踪到的特征点距离没有太远，没有越界,
    if (!(trackDis > maxTrackDis || featuresCur[i].x < xBoundary || 
      featuresCur[i].x > imageWidth - xBoundary || featuresCur[i].y < yBoundary || 
      featuresCur[i].y > imageHeight - yBoundary))
    {
      int xInd = (int)((featuresLast[i].x - xBoundary) / subregionWidth);
      int yInd = (int)((featuresLast[i].y - yBoundary) / subregionHeight);
      int ind = xSubregionNum * yInd + xInd;

      if (subregionFeatureNum[ind] < maxFeatureNumPerSubregion) //
      {
        featuresCur[featureCount].x = featuresCur[i].x;
        featuresCur[featureCount].y = featuresCur[i].y;
        featuresLast[featureCount].x = featuresLast[i].x;
        featuresLast[featureCount].y = featuresLast[i].y;
        featuresInd[featureCount] = featuresInd[i];

        // 当前帧图像光流法追踪到的特征点转换至归一化平面，这里的归一化平面是负的
        point.u = -(featuresCur[featureCount].x - kImage[2]) / kImage[0];
        point.v = -(featuresCur[featureCount].y - kImage[5]) / kImage[4];
        point.ind = featuresInd[featureCount];
        imagePointsCur->push_back(point);// 储存的是光流法追踪的当前帧特征点

        // i< recordFeatureNum 为上上一帧harris特征点
        // i> recordFeatureNum 为上一帧harris特征点
        if (i >= recordFeatureNum)
        {
          point.u = -(featuresLast[featureCount].x - kImage[2]) / kImage[0];
          point.v = -(featuresLast[featureCount].y - kImage[5]) / kImage[4];
          imagePointsLast->push_back(point); // 储存的是上一帧的光流法特征点和harris特征点
        }

        meanShiftX += fabs((featuresCur[featureCount].x - featuresLast[featureCount].x) / kImage[0]);
        meanShiftY += fabs((featuresCur[featureCount].y - featuresLast[featureCount].y) / kImage[4]);

        featureCount++;
        subregionFeatureNum[ind]++;
      }
    }
  }

  totalFeatureNum = featureCount;
  meanShiftX /= totalFeatureNum;
  meanShiftY /= totalFeatureNum;

  // 发布转换到归一化平面后的特征点话题
  sensor_msgs::PointCloud2 imagePointsLast2;
  pcl::toROSMsg(*imagePointsLast, imagePointsLast2);
  imagePointsLast2.header.stamp = ros::Time().fromSec(timeLast);
  imagePointsLast2.header.frame_id = "image";
  imagePointsLastPubPointer->publish(imagePointsLast2);

  showCount = (showCount + 1) % (showSkipNum + 1);
  if (showCount == showSkipNum) {
    //Mat imageShowMat(imageShow);
    Mat imageShowMat(cv::cvarrToMat(imageShow));
    bridge.image = imageShowMat;
    bridge.encoding = "mono8";
    sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
    imageShowPubPointer->publish(imageShowPointer);
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "featureTracking");
  ros::NodeHandle nh;

  mapx = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);
  mapy = cvCreateImage(imgSize, IPL_DEPTH_32F, 1);

   /*
   * 计算形变和非形变图像的对应(map)
   * kMat 摄像机内参数矩阵
   * dMat 形变系数向量
   * mapx x坐标对应的矩阵
   * mapy y坐标对应的矩阵
   */
  
  cvInitUndistortMap(&kMat, &dMat, mapx, mapy);

  CvSize subregionSize = cvSize((int)subregionWidth, (int)subregionHeight);
  imageEig = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);
  imageTmp = cvCreateImage(subregionSize, IPL_DEPTH_32F, 1);

  CvSize pyrSize = cvSize(imageWidth + 8, imageHeight / 3);
  pyrCur = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);
  pyrLast = cvCreateImage(pyrSize, IPL_DEPTH_32F, 1);

  ros::Subscriber imageDataSub = nh.subscribe<sensor_msgs::Image>("/image/raw", 1, imageDataHandler);

  ros::Publisher imagePointsLastPub = nh.advertise<sensor_msgs::PointCloud2> ("/image_points_last", 5);
  imagePointsLastPubPointer = &imagePointsLastPub;

  ros::Publisher imageShowPub = nh.advertise<sensor_msgs::Image>("/image/show", 1);
  imageShowPubPointer = &imageShowPub;

  ros::spin();

  return 0;
}
