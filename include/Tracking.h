/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"

#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    //对输入的图片进行处理，提取特征和立体匹配
    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    //设置局部地图、设置局部闭环检测、设置视图
    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    // 把焦距考虑进去改变MapPoint的scale
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    //设置只对摄像头的位姿进行计算
    void InformOnlyTracking(const bool &flag);


public:

    // Tracking states
    //追踪的状态
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };
    //当前状态
    eTrackingState mState;
    //上一过程的状态
    eTrackingState mLastProcessedState;

    // Input sensor
    //输入的传感器
    int mSensor;

    // Current Frame
    //当前帧和图片
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    //单目相机的匹配数目、特征点和3D地标点的容器
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    //恢复全部相机轨迹的数列，保存每一帧相机的位姿和相应的参考帧
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    //局部建图过程关闭和我们只进行定位为真.
    bool mbOnlyTracking;

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    //与输入传感器无关的追踪函数
    void Track();

    // Map initialization for stereo and RGB-D
    //双目和深度的初始化
    void StereoInitialization();

    // Map initialization for monocular
    //单目相机的初始化
    void MonocularInitialization();
    //创建单目相机的初始化地图
    void CreateInitialMapMonocular();

    //检查被替换的上一帧
    void CheckReplacedInLastFrame();
    //跟踪参考关键帧
    bool TrackReferenceKeyFrame();
    //更新上一帧
    void UpdateLastFrame();
    //通过运动模型估计运动
    bool TrackWithMotionModel();
    //重定位函数
    bool Relocalization();
    //更新局部地图的函数
    void UpdateLocalMap();
    //更新局部地图中的地标点
    void UpdateLocalPoints();
    //更新局部地图中的关键帧
    void UpdateLocalKeyFrames();
    //跟踪局部地图
    bool TrackLocalMap();
    //搜索局部地标点
    void SearchLocalPoints();
    //需要新的关键帧
    bool NeedNewKeyFrame();
    //创建新的关键帧
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    //只进行定位，如果没有和地图中地标点匹配的信息，标记为真。如果这样，系统便尝试重新定位
    bool mbVO;

    //Other Thread Pointers
    //其他两个线程
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    //ORB的特征提取算子
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    //词袋库和关键帧数据库
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    //仅适合单目的初始化类变量
    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    //创建局部地图
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    //定义SLAM系统
    System* mpSystem;
    
    //Drawers
    //定义视图绘画器
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    //创建整个地图
    Map* mpMap;

    //Calibration matrix
    //定义内参矩阵
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    //根据fps，新关键帧的规则
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    //可以近距离被立体相机观测到的地标点通过一帧便可以插入地图，远的点需要两个关键帧插入
    float mThDepth;

    //如果只有RGB-D相机的输入，深度本来就有
    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    //帧中当前匹配的内点数量
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    //上一帧，关键帧和重新定位的信息与序号
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    //运动模型
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    //颜色类型，RGB为真，BGR为假
    bool mbRGB;

    //地标点的数列
    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
