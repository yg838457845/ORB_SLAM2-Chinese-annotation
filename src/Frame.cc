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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;
//const string filename1 = "xiewenzhang.txt";

Frame::Frame()
{}

//Copy Constructor
//Frame的拷贝函数
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

//双目相机的当前帧初始化
//从左到友依次是：左灰度图，右灰度图，时间戳，左图像特征提取算子，有图像特征提取算子，词袋模型，内参矩阵，畸变系数，基线距离，深度
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();//尺度层数
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();//尺度因子
    mfLogScaleFactor = log(mfScaleFactor);//尺度的log
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();//尺度的容器
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();//尺度逆的容器
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();//尺度方差的容器
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();//尺度方差逆的容器

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);//左图像的特征提取线程
    thread threadRight(&Frame::ExtractORB,this,1,imRight);//右图像的特征提取线程
    threadLeft.join();//必须等待threadLeft线程全部结束才可以运行主函数
    threadRight.join();//必须等待threadright线程全部结束才可以运行主函数

    N = mvKeys.size();//特征点的数量

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();//特征点的畸变矫正

    ComputeStereoMatches();//双目特征匹配

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    //仅在第一帧图像中进行调用
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);
        //
        //#define FRAME_GRID_ROWS 48，纵向有48个小格
        //#define FRAME_GRID_COLS 64，横向有64个小格

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);//计算每小格宽度的倒数
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);//计算每小格高度的倒数

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;//mb是以公制为单位的
    //把每一帧分割为48*64个网格
    //根据关键点的畸变矫正后的位置分为不同的网格里面
    AssignFeaturesToGrid();
}

//在tracking类中的GrabImageRGBD函数有此函数的使用，有深度信息的输入
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

//在tracking类中的GrabImagemoncular函数有此函数的使用，无深度信息的输入
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    //当前帧的ID
    //每一帧都有唯一的一个ID做标识
    //从0开始
    mnId=nNextId++;

    // Scale Level Info
    //获得尺度信息
    //这些都是Frame的变量
    //尺度第几层
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    //尺度因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //提取ORB特征
    //可以继续研究
    ExtractORB(0,imGray);
    //提取特征点信息输入到mvKeys中
    //keyPoint（float x,float y，float -size，float -angle=-1,float -response=0,int -octive=0,int -class-id=-1)
    //进行ORB特征提取以后可以得到关键点的数量，mvKey为提取的特征点集合
    N = mvKeys.size();

    //没有找到关键点，返回
    if(mvKeys.empty())
        return;

    //畸变矫正，找到关键点实际应该在普通摄像头中的位置
    UndistortKeyPoints();

    // Set no stereo information
    //把右图、立体信息部分设置为-1
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    //初始化地标点、初始化outliner的状态
    //所以说mvpMapPoints和mvKeys是相同维的容器，只是一个储存地标点（3维），另一个储存特征点（2维）
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    //这一步仅在第一帧或者标定矩阵发生变化时发生
    if(mbInitialComputations)
    {
        //计算畸变矫正之后的边界
        ComputeImageBounds(imGray);

        //mnMax（MIN）X（Y）表式畸变矫正以后的边界，得到矫正图像的宽度和高度
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);
        //从配置文件中读取数据赋给相应元素
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;
        //false表示不是第一帧图像
        mbInitialComputations=false;
    }
    //计算立体匹配的时候的baseline
    //mbf来自配置文件
    mb = mbf/fx;

    //把每一帧分割为48*64个网格
    //根据关键点的畸变矫正后的位置分为不同的网格里面
    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];
        //mGrid表示在其位置附近，都有那些特征点
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    //单目，只用左图像
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else//右图像
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

//计算旋转、平移和光心坐标
void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

//判断地标点的投影是否在帧中，viewingCosLimit观测角度限制
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    //得到地标点的三维信息
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
   //计算投影
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    //投影到像素平面
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;
    //检测投影点是否在当前帧平面内
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    //地标点距离的范围
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    //计算地标点到当前帧光心的距离
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    //得到该地标点与多个可见帧光心的平均连线（单位化后）
    cv::Mat Pn = pMP->GetNormal();
    //计算当前帧连线与平均连线的夹角cos值
    const float viewCos = PO.dot(Pn)/dist;
    //角度过大
    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    //地标点可以被看到
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}
//在特征点附近搜寻匹配
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            //vCell为在ix、iy附近特征点的序号
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                //kpUn为ix、iy附近的特征点
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }
                //计算kpUn与（x、y）的距离
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    //该点（x、y）附近特征点的序号
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}
//posX,posY为离kp最近点的坐标
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);//最近小格子的X起点
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);//最近小格子的Y起点

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        //每一行作为矩阵,分别储存在容器vCurrentDesc中
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        //BowVector表示该特征点描述子对应的向量, 叶节点序号----权重的加和
        //FeatureVector表示该特征点描述子对应的特征向量, 指定层相应节点的序号-----与该节点对应特征点在当前帧序号的集合
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

//特征点矫正
void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

//计算畸变矫正后图像的边界
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

//双目特征匹配
void Frame::ComputeStereoMatches()
{
    //N为左图像特征点的个数
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;//（100+50)/2

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;//左图像第一层的行数

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();//右图像特征点的个数

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];//提取右图像的特征点
        const float &kpY = kp.pt.y;//特征点的纵坐标
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];//尺度的2倍作为半径
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);//特征点上r行和下r行数组对应的都是第iR个特征点
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;//求焦距？

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    //string& filename1 = "xiewenzhang.txt";
    ofstream f("xiewenzhang.txt");
    //f.open(filename1.c_str());
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];//提取左图像的特征点
        const int &levelL = kpL.octave;//特征点对应的尺度层数
        const float &vL = kpL.pt.y;//特征点的纵坐标
        const float &uL = kpL.pt.x;//特征点的横坐标

        const vector<size_t> &vCandidates = vRowIndices[vL];//右图像中第vL行对应的特正点序号

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;//maxD=fx，fx横向焦距，1mm有多少像素点
        const float maxU = uL-minD;//minD=0

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);//左图像第iL个点对应的描述子

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];//右图像在对应行的特征点序号
            const cv::KeyPoint &kpR = mvKeysRight[iR];//特征点

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)//满足尺度要求
                continue;

            const float &uR = kpR.pt.x;//右特征点的横坐标

            if(uR>=minU && uR<=maxU)//横坐标满足偏移距离要求(略微偏左)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);//描述子
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);//计算距离

                if(dist<bestDist)//选取最小距离的右图特征点
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        //找到满足条件的点
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;//右图匹配点的横坐标
            const float scaleFactor = mvInvScaleFactors[kpL.octave];//特征点对应尺度的逆
            const float scaleduL = round(kpL.pt.x*scaleFactor);//用左图像的尺度去恢复左图像的特征点（缩放特征点的坐标）
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);//用左图像的尺度去恢复右图像的特征点横坐标

            // sliding window search
            const int w = 5;
            //当前尺度层特征点的范围内的像素矩阵块，左图像中
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);//得到了矩阵块每个元素与中心像素的偏差

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);//11个空间的大小

            const float iniu = scaleduR0+L-w;//右图像横坐标的下限
            const float endu = scaleduR0+L+w+1;//右图像横坐标的上限
            //用左图像的层数去提取右图像金字塔，mpORBextractorRight->mvImagePyramid[kpL.octave]
            //右图像的特征点横坐标不能太大也不能小于零
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)//从-5到+5
            {
                //滑动窗口，右图像中
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);//数据类型转换
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);//得到矩阵块每个元素与中心元素的偏差

                float dist = cv::norm(IL,IR,cv::NORM_L1);//计算两个像素块的差值
                if(dist<bestDist)//找到最小的匹配像素块
                {
                    bestDist =  dist;
                    bestincR = incR;//最佳像素块的平移量
                }

                vDists[L+incR] = dist;//储存每个像素块的相似度

                 f <<dist<<"  ";
                 if(incR==5)  f<<std::endl;

            }

            if(bestincR==-L || bestincR==L)//窗口滑动到最近头获得了最小距离值，直接跳过
                continue;

            // Sub-pixel match (Parabola fitting)
            //得到最佳匹配像素块相邻两边对应的像素块距离
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));//两边距离与每两个距离差的比值，最小值和两边相差比较大

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);//恢复特征点在右图中横坐标的原始尺度信息

            float disparity = (uL-bestuR);//uL为特征点的横坐标

            if(disparity>=minD && disparity<maxD)//匹配得到的和原始特征点横坐标相差不到1mm
            {
                if(disparity<=0)//只有等于零，计算的右特征点居然右移了
                {
                    disparity=0.01;//认为的减一点，左移一点点
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;//计算深度值
                mvuRight[iL] = bestuR;//储存在右图像中匹配点的横坐标
                vDistIdx.push_back(pair<int,int>(bestDist,iL));//注意：第一个元素是距离
            }
        }
    }
    f.close();
    sort(vDistIdx.begin(),vDistIdx.end());//排序，vDistIdx中表示左图像中每一个特征点，及其对应像素块与右图特征像素块的相似度
    const float median = vDistIdx[vDistIdx.size()/2].first;//中值
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else//排除像素块差异过大的特征点
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];//第i个特征点的深度值
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;//特征点的横坐标
        const float v = mvKeysUn[i].pt.y;//特征点的纵坐标
        const float x = (u-cx)*z*invfx;//计算投影到相机坐标系中的x坐标
        const float y = (v-cy)*z*invfy;//计算投影到相机坐标系中的y坐标
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);//得到特征点的世界坐标
        return mRwc*x3Dc+mOw;//投影到世界坐标点
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
