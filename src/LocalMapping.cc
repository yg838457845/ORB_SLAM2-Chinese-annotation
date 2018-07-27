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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        //设置是否接受关键帧
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        //检查队列中是否存在关键帧
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            //关键帧的BOW计算和处理
            ProcessNewKeyFrame();

            // Check recent MapPoints
            //地标点筛选
            MapPointCulling();

            // Triangulate new MapPoints
            //三角化新的地标点
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                //如果检查没有新的关键帧了，即已经提取的一帧是最后的一个
                //该函数用来利用共享关键帧来更新当前帧的地标点信息
                SearchInNeighbors();
            }
            //false表示允许进行进行BA
            mbAbortBA = false;
            //队列中没有新关键帧且不需要停止
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    //局部BA
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                //删除多余关键帧
                KeyFrameCulling();
            }
            //转入loop线程
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            //如果处于停止状态，且不结束
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }
        //如果需要则重置
        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        // 系统可以接受新的关键帧
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }
    //设置停止状态
    SetFinish();
}
//插入关键帧
void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


//检查队列中是否有关键帧
bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

//处理新的关键帧
void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        //取关键帧列中的第一个关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        //删去关键帧列中的第一个关键帧
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    //计算特征向量
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    //提取当前帧的地标点
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                //如果pMP没有被关键帧mpCurrentKeyFrame观测到
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    //添加当前关键帧的观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    //更新平均深度
                    pMP->UpdateNormalAndDepth();
                    //更新地标点的描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    //地标点已经被当前关键帧观测到，初始两帧也属于这种情况
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    //更新当前关键帧的加权位姿图
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    //往地图中加入当前关键帧
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

//地标点的筛选
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    //提取新建的地标点
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    //提取当前关键帧的id
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            //删除mlpRecentAddedMapPoints中将要处理的地标点
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            //该地标点能被跟踪到的次数/应该被观测到的次数小于25%
            ///（跟踪的次数体现在局部跟踪优化后地标点为内点的次数），（观测的次数体现在局部跟踪中地标点可能被当前帧看到的次数，未优化前）
            /// 当局部建图线程中进行局部地图融合时，地标点如果覆盖了其他错误的地标点，则正确的地标点两个次数同时加一
            //设置地标点为坏的
            pMP->SetBadFlag();
            //删除地标点
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            //地标点被观测到的次数小于cnThObs, 当前帧已经过了两个关键帧,( 感觉这个有问题)
            //也就是连续两帧都没有观测到这个地标点, 设置为坏
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            //该地标点已经被观测过了3个关键帧
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            //换下一个地标点
            lit++;
    }
}

//创建地标点
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        //单目
        nn=20;
    //得到图中与mpCurrentKeyFrame邻接的关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);
    //得到当前关键帧的位姿旋转矩阵
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    //转置
    cv::Mat Rwc1 = Rcw1.t();
    //当前帧位姿的平移
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    //得到光心在世界坐标系下的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;
    //尺度因子
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    //遍历所有的邻接关键帧
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            //如果队列中还有新插入的关键帧, 那么只考虑最佳的匹配关键帧(i=0)
            return;
        //每一个临近关键帧
        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        //得到临近关键帧的光心世界坐标
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        //相机移动的距离
        cv::Mat vBaseline = Ow2-Ow1;
        //模长
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)//移动距离过于小，小于基线长度，则不进行地标点的创建了
            continue;
        }
        else
        {
            //得到临近关键帧pKF2地标点深度的中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            //移动距离/深度中值
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            //移动距离过小
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        //计算基础矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        //匹配,得到匹配结果(序号,序号)
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);
        //pKF2的旋转矩阵
        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        //pKF2的平移向量
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        //Tcw为世界坐标系到相机坐标系
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));
        //临近关键帧的内参
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        //匹配的点对
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            //idx1表示匹配点对的第一个序号（当前帧中）
            const int &idx1 = vMatchedIndices[ikp].first;
            //idx2表示匹配点对的第二个序号（临近关键帧中）
            const int &idx2 = vMatchedIndices[ikp].second;
            //当前关键帧中的特征点
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];//右图像中对应特征点的横坐标
            bool bStereo1 = kp1_ur>=0;
            //临近关键帧中的特征点
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];//右图像中对应特征点的横坐标
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            //计算归一化相机坐标系
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);
            //Rwc1表示从相机坐标系到世界坐标系的转换矩阵
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            //计算该地标点到两帧光心连线的夹角（视差）cos值
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;
            //双目的情况下
            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));//第idx1个特征点对应地标点在当前帧下的视角
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));//第idx2个特帧点对应地标点在共享关键帧下的视角

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);//选取最小的视角

            cv::Mat x3D;
            //（地标点离双目相机过远，cosParallaxStereo过大）（cosParallaxRays过小，地标点的视差角过大，前后两帧的位移略大），视差角不能过小。
            //满足上面条件，则利用三角化进行地标点的确定
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                //对系数矩阵A进行奇异值分解
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
                //提取特征值最接近0的特征向量
                x3D = vt.row(3).t();
                //最后一行归一化
                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            //当前帧双目的右图像横坐标存在且地标点到当前帧相机的距离较近
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);//将第idx1特征点投影到世界坐标系
            }
            //共享帧双目的右图像横坐标存在且地标点到共享帧相机的距离较近
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);//将第idx2特征点投影到世界坐标系
            }
            else
                //单目且是低视差
                continue; //No stereo and very low parallax
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            //检测地标点在相机坐标系下的深度是否为正
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            //检查重投影误差
            //提取特征点在当前帧的层数
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            //计算世界点投影在当前帧相机坐标中的坐标
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            //深度的逆
            const float invz1 = 1.0/z1;
            //如果不是第一种立体视觉的情况
            if(!bStereo1)
            {
                //计算当前帧投影点信息
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                //计算二次投影误差
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                //如果误差过大则直接跳出
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else//当前帧双目的情况下
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;//计算右图的横坐标
                float v1 = fy1*y1*invz1+cy1;//纵坐标
                float errX1 = u1 - kp1.pt.x;//计算二次投影误差
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;//计算右图像横坐标和真实横坐标之间的误差
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            //检测第二帧中的二次投影误差
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            //检查尺度一致性
            //计算地标点到光心的距离
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;
            //////////////注意距离和尺度成反比
            //距离比值
            const float ratioDist = dist2/dist1;
            //尺度因子的比值
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            // 反正一个限制条件，也不知道怎么来的
            //const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
            //距离的比值与尺度的比值相差不大
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            //经过了上面的各种限制，三角化成功
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
            //添加当前关键帧为地标点的可视关键帧
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            //添加临近关键帧为地标点的可是关键帧
            pMP->AddObservation(pKF2,idx2);
            //当前关键帧地标点容器中加入地标点
            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            //临近关键帧pKF2地标点容器中加入地标点
            pKF2->AddMapPoint(pMP,idx2);
            //计算地标点的特征描述子
            pMP->ComputeDistinctiveDescriptors();
            //更新地标点的平均方向和深度信息
            pMP->UpdateNormalAndDepth();
            //地图中加入该地标点
            mpMap->AddMapPoint(pMP);
            //最近地标点容器中也加入该地标点
            mlpRecentAddedMapPoints.push_back(pMP);
            //新点个数加一
            nnew++;
        }
    }
}

//在mlNewKeyFrames容器中没有新关键帧的情况下
//该函数是用来利用共享关键帧的信息来更新当前关键帧的地标点信息
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    //单目中提取当前关键帧的共享关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        //共享关键帧的mnFuseTargetForKF初始为0，一个共享关键帧只能被保存一次
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        //共享关键帧的集合
        vpTargetKFs.push_back(pKFi);
        //标记共享关键帧
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        //寻找共享关键帧的共享关键帧
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            //该二次共享关键帧既不是一次共享关键帧，也不是当前关键帧
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    //当前关键帧的所有地标点
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        //遍历所有共享关键帧
        KeyFrame* pKFi = *vit;
        //当前帧的地标点投影到共享关键帧，并进行融合
        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    //vpFuseCandidates的维数为（地标点数量*共享关键帧的数量）
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        //遍历所有的共享关键帧
        KeyFrame* pKFi = *vitKF;
        //共享关键帧中所有的地标点
        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;//所有地标点
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            //表示该地标点是当前帧局部地图中的
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            //储存共享关键帧中所有的地标点
            vpFuseCandidates.push_back(pMP);
        }
    }
    //将vpFuseCandidates中的地标点投影到当前关键帧，进行融合
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    //提取局部地图投影处理后的地标点
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        //遍历处理后的地标点
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                //更新描述子
                pMP->ComputeDistinctiveDescriptors();
                //更新深度和方向
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    //更新图的连接关系
    mpCurrentKeyFrame->UpdateConnections();
}

//计算两个关键帧之前的基础矩阵
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    //计算第二个关键帧到第一个关键帧的旋转和平移
    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
    //求反对称矩阵
    cv::Mat t12x = SkewSymmetricMatrix(t12);
    //求内参矩阵
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    //得到基础矩阵
    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    //在插入关键帧的时候，mbNotStop为True，局部建图线程不能停止
    //插入关键帧后，mbNotStop为False
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

//接受关键帧
bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

//设置是否接受关键帧的状态
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

//插入关键帧前，要求mbNotStop为True
//插入关键帧后，mbNotStop为false
bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);
    //mbStopped的初始为false
    //在局部地图停止的情况下，mbStopped为true
    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

//剔除多余的关键帧
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    //提取当前关键帧的邻接关键帧
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        //遍历所有的邻接关键帧////////////////////////////////////
        KeyFrame* pKF = *vit;
        //如果是第一帧，则直接跳过（不删除初始帧）
        if(pKF->mnId==0)
            continue;
        //提取每一个邻接帧的地标点
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];//遍历每一个地标点/////////////////////////////////////
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)//非单目
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)//可以观测到该地标点的关键帧个数大于阈值
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;//提取该地标点在关键帧pKF上对应特征点的尺度层数
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();//提取观察到该地标点的所有关键帧
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)//遍历每一个可观测关键帧////////////////
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)//如果该帧就是当前这个邻接关键帧
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;//获取尺度层数

                            if(scaleLeveli<=scaleLevel+1)//满足尺度要求
                            {
                                nObs++;//可被观测的次数加一（除了邻接关键帧）
                                if(nObs>=thObs)//超过阈值，则直接跳出一层循环
                                    break;
                            }
                        }
                        if(nObs>=thObs)//如果该地标点被其他至少3个关键帧观测到
                        {
                            nRedundantObservations++;//冗余参数加一
                        }
                    }
                }
            }
        }  
        //如果冗余数量大于该关键帧地标点数量的90%
        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

//求解反对称矩阵
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
