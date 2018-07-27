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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{
//Tracking过程的构造函数
Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    //畸变系数
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

//在System.cc中初始化过程有调用
void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)//立体相机的跟踪函数
{
    mImGray = imRectLeft;//左图像
    cv::Mat imGrayRight = imRectRight;//右图像

    if(mImGray.channels()==3)//灰度转换
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)//灰度转换
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }
    //当前帧的初始化,主要包括特征提取
    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

//新的一帧图像来临时,直接进入该函数,不会再进行类的构造的过程, 所以类的状态可以不断传承下去
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    //把输入的图片设为当前帧
    mImGray = im;
    //若图片是三、四通道的，还需要转化为灰度图
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    //若状态是为初始化或者当前获取的图片是第一帧的时候，那么就需要进行初始化
    //传入的参数就是当前帧（灰度图），时间戳，ORB特征，DBOW的字典，标定矩阵，畸变参数，立体视觉的（baseline）x（fx），可靠点的距离的阈值
    //mCurrentFrame就是对当前帧进行处理，获取ORB特征，获取尺度信息
    //mpIniORBextractor比mpORBextractorLeft多提取了一倍的特征点
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    //追踪，初始化以及特征匹配
    //需要继续研究
    Track();

    //mTcw是位姿
    //clone是cv：Mat特有的方法
    //实现的就是完全拷贝，把数据完全拷贝而不共享数据
    return mCurrentFrame.mTcw.clone();
}

//这部分在提取出特征点和初始化参数之后，跟踪线程
void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    //上一个地图锁，表示地图不能进行更新和改变
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            //单目初始化过程，这部分需要大力研究
            //连续两帧初始化完成后,mstate=ok
            MonocularInitialization();
        //更新视图,且视图类中的mstate=mLastProcessedState,落后一个状态
        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    //初始化过程完成，则开始跟踪过程
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        //不是只进行跟踪(也有建图过程)
        //通过运动模型或者重定位来进行当前帧的位置初始估计
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            //建图过程也在活跃中,初始化过程成功
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                //局部地图线程也许会改变上一帧中的一些地标点，所以提前对上一帧的信息进行检测, 看其中的地标点是否存在替换的可能
                CheckReplacedInLastFrame();
                //如果没有运动模型或者当前帧与重定位的序号仅差1
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    //利用关键帧（PnP）进行跟踪
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    //利用运动模型进行更新，期间mbVO可能为TRUE
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        //如果更新错误还是选择参考帧更新的方法
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                //跟踪丢失，那么需要重新定位
                bOK = Relocalization();
            }
        }
        //只有跟踪线程，局部地图不开启，这时候需要检测当前帧与地图的关系，即判断变量mbVo的值
        else
        {
            // Localization Mode: Local Mapping is deactivated
            //建图的过程未被激活，仅有跟踪过程
            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                //mbvo为真表示地图中与当前帧的匹配信息丢失，只能依靠上一帧进行视觉里程计的步骤或尝试进行重定位
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        //期间mbVO可能为TRUE
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.
                    //在这个过程中，进行两个方向，一个通过运动模型进行位置估计，一个通过重定位的方法。
                    //重定位如果成功便直接使用结果
                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    //如果运动模型不为空
                    if(!mVelocity.empty())
                    {
                        //运动模型预测成功则返回真，此时其中的mCurrentFrame已被更新
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        //mvbOutlier在OptimalPose得到赋值， （边的误差在阈值内）地标点对应的mvbOutlier为否
                        //（边的误差不在阈值内）地标点对应的mvbOutlier为真
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    //重定位模型运行成功则返回真
                    bOKReloc = Relocalization();
                    //当运动模型为真而重定位失败时执行
                    if(bOKMM && !bOKReloc)
                    {
                        //初始化当前位姿
                        mCurrentFrame.SetPose(TcwMM);
                        //当前帧的地标点
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        //当前帧的异常点
                        mCurrentFrame.mvbOutlier = vbOutMM;
                        //mbVO为真表示地图中跟踪地标点丢失
                        if(mbVO)
                        {
                            //N表示特征点的个数
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                //特征点是地标点且不是异常点
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    //当前帧中地标点的被查找次数增加一
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    //如果重定位成功，那么将mbVO更新为假，表示地图中的地标点与当前帧存在匹配关系
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }
                    //运动模型与重定位有一个成功，则认为tracking过程中的当前帧初始估计完成
                    bOK = bOKReloc || bOKMM;
                }
            }
        }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //mpReferenceKF跟踪过程中最新的一个关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        //如果局部建图过程被激活
        if(!mbOnlyTracking)
        {
            //当前帧的初始位姿估计完成
            if(bOK)
                bOK = TrackLocalMap();
        }
        //只有跟踪线程在运行，只有跟踪线程的话需要进行判断当前帧是否与地图匹配成功，即判断变量mbVO
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            //初始估计完成且地图与当前帧存在匹配关系
            //mbVO为真假变量，mbVo为真表示当前帧中只有很少特征点与地图存在匹配，因此一般采用视觉里程计的估计方法或者重定位的方法
            //为假表示跟踪过程正常
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }
        //如果局部跟踪正常，那么当前状态变为OK
        if(bOK)
            mState = OK;
        else
            //局部地图跟踪不正常，此时就需要进行重新定位
            mState=LOST;

        // Update drawer
        //更新图像帧的绘图建模变量
        mpFrameDrawer->Update(this);
        /////////////////////////////////////////////////////////////////////////////////////////////////////////
        //局部跟踪完成
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // If tracking were good, check if we insert a keyframe
        //如果局部地图跟踪的结果良好，则需要考虑是否可以插入新的关键帧
        //如果局部地图跟踪正常
        if(bOK)
        {
            // Update motion model
            //上一帧的位姿估计成功
            if(!mLastFrame.mTcw.empty())
            {
                //设置一个4阶的单位阵
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                //将上一帧位姿矩阵中旋转部分的逆矩阵赋值给了mlastFrame矩阵的3*3子矩阵
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                //将上一帧位姿矩阵中的光心世界点赋值给MlastFrame矩阵的最后一列
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                //当前帧位姿与上一帧位姿逆相乘得到了速度的结果
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                //如果上一帧位姿估计失败，则认为位姿模型是空的
                mVelocity = cv::Mat();
            //更新地图建模器
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            //清除VO的匹配
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    //如果当前帧中地标点没有被关键帧观测到，则认为它只是视觉里程计（VO）生成的地标点，不可靠，删去
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            //Track类中的成员mlpTemporalPoints，不太懂，只有在单目或者不开启局部建图线程的条件下
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            //判断是否需要插入关键帧
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            //剔除当前帧中异常点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                //地标点且异常值检测为真
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }
        /////////////////////////////////////////////////////////////////////////////////////////////////////////
        //新关键帧的判断完成
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Reset if the camera get lost soon after initialization
        //局部跟踪失败，状态变为LOST
        if(mState==LOST)
        {
            //如果地图中的关键帧数量小于5,认为一初始化后就跟踪丢失，需要重新复位
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }
        //如果当前帧的参考帧为空的话，将系统的参考帧进行赋值
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        //当前帧作为系统跟踪的上一帧
        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    //保存位姿信息以便可以重新得到完整的相机运行轨迹
    if(!mCurrentFrame.mTcw.empty())
    {
        //计算当前帧到参考关键帧的位姿转换关系
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        //下面都是写保存当前帧与参考关键帧信息的容器
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        //这里面为什么添加LOST，但是不知道
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        //.back()表示返回尾元素的引用,即相当于将容器中的尾元素复制了一个添加到最后
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}

//立体视觉初始化
void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)//N表示左图像特征点的数量
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));//设置初始帧的坐标为世界坐标

        // Create KeyFrame
        // 当前帧、地图、关键帧数据库
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        //地图中加入初始关键帧
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];//提取每一个特征点的深度值
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);//将特征点投影到世界坐标系
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);//设置地标点
                pNewMP->AddObservation(pKFini,i);//往地标点中添加关键帧的观测
                pKFini->AddMapPoint(pNewMP,i);//初始关键帧中添加地标点信息
                pNewMP->ComputeDistinctiveDescriptors();//计算地标点的最佳描述子
                pNewMP->UpdateNormalAndDepth();//更新地标点的方向和深度
                mpMap->AddMapPoint(pNewMP);//地图中添加地标点

                mCurrentFrame.mvpMapPoints[i]=pNewMP;//当前帧中添加地标点信息
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);//将初始关键帧插入到局部建图线程中

        mLastFrame = Frame(mCurrentFrame);//当前帧变为上一帧
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;//上一个关键帧

        mvpLocalKeyFrames.push_back(pKFini);//局部关键帧集合中加入初始关键帧
        mvpLocalMapPoints=mpMap->GetAllMapPoints();//得到地图中的所有地标点
        mpReferenceKF = pKFini;//参考关键帧
        mCurrentFrame.mpReferenceKF = pKFini;//当前帧的参考关键帧

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);//地图的初始参考点

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);//地图的初始关键帧

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);//将当前帧的位姿传递给视图绘制类

        mState=OK;//表示初始化完成
    }
}

//单目初始化
//只有两个连续帧的特征点数量大于100,才可以进行初始化
void Tracking::MonocularInitialization()
{
    //mpInitializer为一个初始化类的变量，当mpInitializer为空则执行
    //连续帧的第一帧，之后mpInitializer就不会为空
    if(!mpInitializer)
    {
        // Set Reference Frame
        //如果当前帧mCurrentFrame的ORB特征点数量大于100
        if(mCurrentFrame.mvKeys.size()>100)
        {
            //将当前帧作为初始帧
            mInitialFrame = Frame(mCurrentFrame);
            //将当前帧作为上一帧
            mLastFrame = Frame(mCurrentFrame);
            //设置mvbPrevMatched的初始值，为一个二维点的容器
            //mvKeysUn为当前帧的经过矫正后的特征点
            //std::vector<cv::Point2f> mvbPrevMatched;
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            //将mvKeysUn中的值赋值给mvbPrevMatched
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;
            //如果存在mpInitializer，则将其置空
            if(mpInitializer)
                delete mpInitializer;
            //给mpInitializer赋初始值，利用mpInitializer的构造函数
            //将当前帧mCurrentFrame的内参和特征点赋值到mpInitializer中
            //1为sigma值，200为最大迭代次数
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);
            //将mvIniMatches的每一个值赋值为-1
            //std::vector<int> mvIniMatches;
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            //返回值为空
            return;
        }
    }
    //mpInitializer不为空
    //连续帧的第二帧
    else
    {
        // Try to initialize
        //尝试初始化，如果当前帧的特征点数量小于100
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            //删除mpInitializer
            delete mpInitializer;
            //mpInitializer赋初始值
            mpInitializer = static_cast<Initializer*>(NULL);
            //将mvIniMatches的每一个值赋值为-1
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        //ORBmatcher类的初值，mfNNratio的值为0.9,mbCheckOrientation为true
        ORBmatcher matcher(0.9,true);
        //寻找初始帧mInitialFrame、当前帧mCurrentFrame的匹配点，mvbPrevMatched是初始帧的特征点，mvIniMatches为结果
        //mvIniMatches的维度和mInitialFrame一样大，记录mvbPrevMatched中每一个点在mCurrentFrame中匹配信息的序号
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        //检测匹配到的点对小于100
        if(nmatches<100)
        {
            //删除初始指针
            delete mpInitializer;
            //并且赋值为空
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }
        //当前帧的旋转矩阵
        cv::Mat Rcw; // Current Camera Rotation
        //当前帧的平移向量
        cv::Mat tcw; // Current Camera Translation
        //三角化的容器
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        //初始化指针中初始化位姿和三角化容器
        //mvIniMatches为初始参考帧中每一个特征点在当前帧中匹配点的序号
        //std::vector<cv::Point3f> mvIniP3D;
        //Rcw表示从w到c的坐标转换，w表示初始参考帧，c表示当前帧
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            //每一个匹配点对
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                //如果匹配点成功但是三角化失败
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    //将该匹配点对作为-1,即认为这个匹配点对失败
                    mvIniMatches[i]=-1;
                    //减少匹配的数量
                    nmatches--;
                }
            }

            // Set Frame Poses
            //让初始帧的位姿为单位阵4*4
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            //利用Rcw和tcw确定当前帧的位姿
            //Tcw表示从w（参考帧）到（c）当前帧的位姿转换关系
            mCurrentFrame.SetPose(Tcw);
            //创建初始地图
            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    //创建关键帧
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    //计算两个关键帧的词袋模型向量
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    //将这两个关键帧插入地图中
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    //创建地标点
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        //mvIniMatches表示匹配失败
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        //std::vector<cv::Point3f> mvIniP3D;
        cv::Mat worldPos(mvIniP3D[i]);
        //创建地标点
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);
        //初始关键帧中加入第i个地标点
        pKFini->AddMapPoint(pMP,i);
        //当前关键帧中加入第mvIniMatches[i]个地标点
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);
        //pMP地标点在初始关键帧pKFini中可见，且投影到第i个特征点的位置
        pMP->AddObservation(pKFini,i);
        //pMP地标点在当前关键帧pKFcur中可见，且投影到第mvIniMatches[i]个特征点的位置
        pMP->AddObservation(pKFcur,mvIniMatches[i]);
        //计算最佳的描述子
        pMP->ComputeDistinctiveDescriptors();
        //更新深度
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        //当前帧中的第mvIniMatches[i]个地标点为pMP
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        //且该位置不是异常值点
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        //将地标点pMP加入到地图中去
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    //初始关键帧更新关联性
    pKFini->UpdateConnections();
    //当前关键帧更新关联性
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;
    //BA优化计算mpMap，20次迭代
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    //计算中值深度
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;
    //如果平均深度小于零或者当前关键帧跟踪的地标点数量小于100
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        //重置
        Reset();
        return;
    }

    // Scale initial baseline
    //缩放了当前帧的位姿
    //得到当前关键帧的位姿（相对初始关键帧）
    cv::Mat Tc2w = pKFcur->GetPose();
    //第四列中的前三个元素, 即平移向量，缩小尺度
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    //重新确定位姿信息
    pKFcur->SetPose(Tc2w);

    // Scale points
    //从初始关键帧pKFini中得到所有的地标点
    //缩放了初始关键帧中所有的地标点
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            //将地标点的位姿都缩放
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }


    //局部地图中插入初始参考帧pKFini与当前关键帧pKFcur
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    //将当前帧mCurrentFrame的位姿信息更新
    mCurrentFrame.SetPose(pKFcur->GetPose());
    //当前关键帧作为上一关键帧的序号
    mnLastKeyFrameId=mCurrentFrame.mnId;
    //当前关键帧为上一个关键帧
    mpLastKeyFrame = pKFcur;

    //mvpLocalKeyFrames是局部地图的关键帧集合
    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    //mvpLocalMapPoints是局部地图的地标点集合
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    //参考帧变为当前关键帧
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;
    //当前帧变为上一帧
    mLastFrame = Frame(mCurrentFrame);
    //地图中参考地标点的集合
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    //画图工具
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
    //地图起始关键帧
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
    //状态ok
    mState=OK;
}
//检测上一个关键帧
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        //上一个关键帧中的每一个地标点
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            //是否存在可代替的地标点
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

//根据关键帧来跟踪
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    //计算当前帧的词袋向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    //0.7为第一匹配结果和第二匹配结果比值的阈值,true为需要进行方向检测
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;
    //匹配参考帧和当前帧
    //vpMapPointMatches的维数和当前帧特征点个数相同, 储存的是---当前帧特征点序号对应的地标点
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    //匹配个数小于15
    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    //初始位姿为上一帧的位姿
    mCurrentFrame.SetPose(mLastFrame.mTcw);
    //优化当前帧的姿态
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                //是异常点
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                //异常点为空
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                //地标点pMp没有被跟踪到
                pMP->mbTrackInView = false;
                //最后一个观测到pMP地标点的帧序号
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            //该地标点被观测到的次数大于0, 该观察次数只有在关键帧插入的时候才计数
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    //当前帧中的10个以上地标点可以被观测到(跟踪成功)
    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    //上一帧的参考关键帧
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    //mlRelativeFramePoses为上一帧相对参考关键帧的位姿Trc容器
    cv::Mat Tlr = mlRelativeFramePoses.back();
    //得到上一帧的位姿Tlw
    mLastFrame.SetPose(Tlr*pRef->GetPose());
    //单目就不用检查了~，下面暂时不看了，开启局部建图线程也不用进行检查，下面步骤直接跳过
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}
//根据速度模型的更新，期间mbVO可能为TRUE
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();
    //初步预测，得到当前帧初步的位姿
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    //如果是单目的
    if(mSensor!=System::STEREO)
        th=15;
    else//双目搜索半径的阈值为7
        th=7;
    //匹配, 匹配到的地标点储存在CurrentFrame.mvpMapPoints
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    //放大窗口搜索
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }
    //还是小的话认为失败
    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                //地标点异常
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                //地标点pMp没有跟踪成功
                pMP->mbTrackInView = false;
                //地标点最后被观测到的一帧
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                //nmatchesMap为跟踪成功的点数目
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        //如果只能跟踪, 为真需要的条件更大
        //mbVO为真假变量，mbVo为真表示当前帧中只有很少特征点与地图存在匹配，因此一般采用视觉里程计的估计方法或者重定位的方法
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

//局部地图跟踪
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    //创建当前帧的局部地图
    UpdateLocalMap();
    //将初始位姿估计中没经过匹配的地标点加入mCurrentFrame.mvpMapPoints
    SearchLocalPoints();

    // Optimize Pose
    //优化当前位姿
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                //不是异常点，增加该地标点被找到的次数
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    //不只跟踪
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        //可以观测到该地标点的关键帧数量大于0, 则匹配内点数量加1
                        mnMatchesInliers++;
                }
                else
                    //仅跟踪
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)//如果该地标点异常且双目，则将该地标点赋值为空
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

//判断是否需要插入新的关键帧
bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        //如果只被允许Tracking过程， 则肯定不行
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    //如果局部地图线程被闭环线程冻结，则不需要插入新的关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;
    //提取地图中所有的关键帧的数量
    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    //如果刚经历了重定位
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    // 跟踪地标点最少观察次数
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    //找到当前参考关键帧跟踪的一些地标点，其被观测次数大于nMinObs
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    //当前是否接受关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    //立体视觉中
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)//特征点的深度满足要求
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;//深度信息正确的特征点数
                else
                    nNonTrackedClose++;
            }
        }
    }
    //单目为FALSE，双目中正确特征点的点数较少
    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)//指代初始的几帧
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    //当前帧的序号相比上一帧已经过了很久
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    //当前帧相比上一帧已经过了很久，且局部地图需要
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    //立体的情况，如果局部地图跟踪的内点过少或者双目需要添加
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    //mnMatchesInliers在TrackLocalMap（）中有定义，表示局部地图中跟踪到的内点。其数量过少。
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            //当前帧需要，则需要关键帧，为true
            return true;
        }
        else
        {
            //停止BA优化过程
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)//局部建图队列中的新插入关键帧数量较少
                    return true;
                else
                    return false;
            }
            else
                return false;//单目直接认为不能插入关键帧
        }
    }
    else
        return false;
}

//创建新的关键帧
void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))//如果局部建图线程不能被设置为（一直运行），即局部地图正在停止，则直接返回
        return;
    //初始化关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
    //当前关键帧复制给参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;
    //如果是立体视觉
    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();//计算当前帧的旋转和平移、光心坐标

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);//容器大小为当前帧特征点的数量
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));//储存深度—特征点序号
            }
        }

        if(!vDepthIdx.empty())//不为空
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());//将特征点按照深度排序

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;//序号

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];//提取当前帧该特征点处的地标点
                if(!pMP)//不存在地标点
                    bCreateNew = true;//需要创建为真
                else if(pMP->Observations()<1)//地标点没有被其它的关键帧观测
                {
                    bCreateNew = true;//需要创建为真
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);//将原来错误的删掉
                }

                if(bCreateNew)//如果需要创建该特征点对应的地标点
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);//将第i个特征点投影到世界坐标系中
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);//初始化地标点
                    pNewMP->AddObservation(pKF,i);//将关键帧的信息添加到地标点中
                    pKF->AddMapPoint(pNewMP,i);//将地标点的信息添加到关键帧中
                    pNewMP->ComputeDistinctiveDescriptors();//计算地标点的最佳描述子
                    pNewMP->UpdateNormalAndDepth();//更新深度和方向
                    mpMap->AddMapPoint(pNewMP);//往地图中添加地标点

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;//当前帧的第i个特征点处对应创建的地标点
                    nPoints++;//创建点的数量加一
                }
                else
                {
                    nPoints++;//地标点数量加一
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)//地标点深度不满足要求或者地标点数量已经超过100,则直接跳出循环
                    break;
            }
        }
    }
    //局部地图插入当前关键帧
    mpLocalMapper->InsertKeyFrame(pKF);
    //赋值状态为false，表示停止
    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

//该函数是用来搜索当前帧局部地图中未经过匹配且可以投影到当前帧（初始位姿跟踪过程）的地标点，将其加入局部地图中mCurrentFrame.mvpMapPoints
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        //遍历当前帧的所有地标点
        MapPoint* pMP = *vit;
        if(pMP)
        {
            //地标点出错
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                 //pMP地标点应该被观测到的次数
                pMP->IncreaseVisible();
                //之前通过TrackWithMotionModel或TrackReferenceKeyFrame或Relocalization已经得到的当前帧对应的地标点
                //局部地图将不再进行添加和搜索
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    //所有的局部地图中的点
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        //判断地图中的点是否为当前帧中的地标点，不包括已经存在在mCurrentFrame.mvpMapPoints中的地标点
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        //判断pMP的投影是否有可能在当前帧中
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            //且在函数（isInFrustum）中将pMP->mbTrackInView = true;
            //pMP可以被看到
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        //如果相机刚刚经过重定位
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        //在pMP->mbTrackInView = true的前提下才会运行该匹配函数，得到能投影到mCurrentFrame中所有的局部地标点
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

//局部地图更新
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    //mvpLocalMapPoints表示局部地图地标点
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

//更新局部地图点
void Tracking::UpdateLocalPoints()
{
    //清楚局部地图点
    mvpLocalMapPoints.clear();
    //遍历局部关键帧
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        //提取关键帧特征点对应的地标点
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
        //遍历每一个地标点
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            //mnTrackReferenceForFrame最开始为0，mnTrackReferenceForFrame用来标记地标点归属于哪个局部地图
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                //赋值地标点的跟踪参考帧
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        //遍历当前帧对应的所有地标点
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                //pMP->GetObservations()会返回所有可以观测pMP的关键帧，size-t为地标点在该关键帧中对应的序号
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    //keyframeCounter储存这些关键帧出现重复的次数，共享地标点的数量
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        //遍历当前帧的所有共享关键帧
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;
        //挑选局部地图中出现最多的关键帧，pKFmax有最多的共享关键帧
        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }
        //往局部地图中加入共享关键帧
        mvpLocalKeyFrames.push_back(it->first);
        //局部地图中共享关键帧的跟踪参考帧为当前帧，mnTrackReferenceForFrame为局部地图的编号
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // 局部地图中共享关键帧
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        //如果局部地图中的关键帧数量大于80
        if(mvpLocalKeyFrames.size()>80)
            break;
        //遍历每一个共享关键帧
        KeyFrame* pKF = *itKF;
        //提取关键帧pKF附近共享点最多的10个共享关键帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        //共享关键帧附近的临近关键帧
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                //该临近关键帧不是当前帧的共享关键帧
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    //将该临近关键帧加入局部地图中
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    //mnTrackReferenceForFrame为局部地图的编号，赋值为当前帧的ID
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
        //共享关键帧的孩子节点
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    //将不是属于局部地图的孩子节点添加到局部地图中
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
        //共享关键帧的父节点
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                //将不属于局部地图的父节点添加到局部地图中
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        //将mpReferenceKF赋值为最佳共享关键帧
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}
//重定位函数
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    //计算当前帧的Bow向量
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    //在关键帧数据库中寻找匹配到的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;
    //得到候选关键帧的数量
    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    //匹配类型
    ORBmatcher matcher(0.75,true);
    //PnP变量的维数与候选关键帧的数量
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);
    //每一个候选关键帧都对应一群匹配到的地标点
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        //提取每一个候选关键帧
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            //vvpMapPointMatches[i]的维数与mCurrentFrame特征点数量相同
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                //初始化PnP算法, 最小内点数目=10, 最大迭代次数300, 最少地标点数目4
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            //vbDiscarded为真,丢弃该候选关键帧
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            //储存地标点是否内点
            vector<bool> vbInliers;
            //PNP内点的数量
            int nInliers;
            //点不够
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            //PnP算法计算
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);
                //这个候选关键帧的地标点集合
                set<MapPoint*> sFound;
                //内点的大小
                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    //PNP中RANACS后检验为内点的地标点
                    if(vbInliers[j])
                    {
                        //是内点
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        //不重复的插入地标点
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }
                //优化当前位姿
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;
                //剔除异常的地标点
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    //按照投影重新匹配
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points

                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                //当前帧的地标点(投影匹配成功后的)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            //再次投影匹配
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                //最后一次优化
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
