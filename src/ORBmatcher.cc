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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

//投影匹配函数，用于局部地图跟踪过程中，vpMapPoints表示局部地图中的地标点
//利用那些不属于当前帧的地标点，但是投影在当前帧内的地标点进行投影特征匹配
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        //局部地图中的地标点且之前没有被当前帧观测到
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        //计算搜索半径
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if(bFactor)
            r*=th;

        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())
            continue;
        //该地标点的描述子
        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(F.mvpMapPoints[idx])
                 //当前帧对应的地标点经过了周围多个关键帧的观察，说明已经有值
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            if(F.mvuRight[idx]>0)//双目且存在右图横坐标
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);//投影点右图横坐标和特征点右图横坐标的差值
                if(er>r*F.mvScaleFactors[nPredictedLevel])//大于阈值
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(MPdescriptor,d);
            //提取第一和第二佳的匹配点
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;
            //更新当前帧特征点信息
            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;
    //计算点到直线的距离
    const float dsqr = num*num/den;
    //距离要满足合适的阈值
    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

//通过DOW库寻找关键帧与当前帧的匹配, 不需要提前知道位姿信息
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    //提取参考帧的参考地标点
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    //vpMapPointMatches为大小为N的容器
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));
    //关键帧的特征向量
    //FeatureVector表示该特征点描述子对应的特征向量, 指定层相应节点的序号-----与该节点对应特征点在当前帧序号的集合
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;
    //HISTO_LENGTH=30
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
        if(KFit->first == Fit->first)
        {
            //两帧在指定层匹配到的节点相同, 在相同的节点约束下
            //并提取该节点对应的特征点序号
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;
            //在该节点对应特征点的集合内(参考关键帧)
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                //特征点序号
                const unsigned int realIdxKF = vIndicesKF[iKF];
                //地标点
                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;                
                //该地标点的描述子
                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;
                int bestIdxF =-1 ;
                int bestDist2=256;
                //在该节点对应特征点的集合内(当前帧)
                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    //特征点序号
                    const unsigned int realIdxF = vIndicesF[iF];
                    //已有匹配结果,直接跳过
                    if(vpMapPointMatches[realIdxF])
                        continue;
                    //该特征点的描述子
                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);
                    //计算特征点见的距离
                    const int dist =  DescriptorDistance(dKF,dF);
                    //提取最小和第二小的距离
                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }
                //bestDist1小于阈值
                if(bestDist1<=TH_LOW)
                {
                    //最小与第二小差值足够
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        //当前帧第bestIdxF个特征点匹配到的地标点为pMP
                        vpMapPointMatches[bestIdxF]=pMP;
                        //参考帧特征点
                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];
                        //方向检查
                        if(mbCheckOrientation)
                        {
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            //bin表示一个角度的估值
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            //该角度对应容器中加入匹配序号
                            rotHist[bin].push_back(bestIdxF);
                        }
                        nmatches++;
                    }
                }

            }
            //特征向量加一,换下一个节点
            KFit++;
            Fit++;
        }
        //参考帧对应的节点序号较小
        else if(KFit->first < Fit->first)
        {
            //返回一个不小于Fit->first的值(迭代器)
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            //返回一个不小于KFit->first的值(迭代器)
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

//根据投影映射来寻找匹配点, th=10
//该函数用于loopmapping线程中, 用于完善相似变换矩阵
//3D-2D的投影
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    //一般旋转矩阵的每一行的模值都要为1
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));//得到尺度
    cv::Mat Rcw = sRcw/scw;//得到旋转矩阵
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;//得到光心的世界坐标

    // Set of MapPoints already found in the KeyFrame
    //vpMatched中储存了所有已经匹配好的内点,其维度与pKF的特征点数量相同
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));//删除那些没有进行匹配过的位置

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    //遍历所有的地标点(相似回环帧局部地图内的)
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];//提取每一个地图中的点

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))//已经被匹配过则不进行处理
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();//得到地标点的世界坐标

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;//投影到当前帧的相机中

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;
        //投影到像平面上
        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;//地标点离光心的距离
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)//满足距离的要求
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();//得到点的方向

        if(PO.dot(Pn)<0.5*dist)//cos(当前帧的观测方向,该地标点的平均观测方向)<0.5, 即角度过大
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);//预测该地标点所在的层数

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];//搜索半径

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);//寻找投影点最近的特征点

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();//地标点的描述子

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;//遍历每一个离投影点较近的特征点
            if(vpMatched[idx])//已经完成了匹配的特征点(地标点)则不需要
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;//计算尺度层数

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)//满足尺度要求
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);//当前特征点的描述子

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;//最佳的匹配特征序号
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;//设置该特征点处匹配成功,得到相应的地标点
            nmatches++;//额外匹配到的个数加一
        }

    }

    return nmatches;
}
//初始化时用的特征匹配函数，windowSize为匹配的窗口大小
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{

    int nmatches=0;
    //给vnMatch12赋值为-1的容器，大小和F1的特征点数量一样
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);
    //HISTO_LENGTH是类中的一个成员值，为30
    //这种定义表示定义一个数组，数组中每一个元素都是vector<int>类型的
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        //给rotHist预留500的空间
        rotHist[i].reserve(500);
    //定义因子
    const float factor = 1.0f/HISTO_LENGTH;
    //定义匹配距离的容器，大小和F2的特征点数目一样
    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    //给vnMatch21赋值为-1的容器，大小和F2的特征点数量一样
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        //kp1为F1特征点
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        //octave代表是从金字塔哪一层提取的得到的数据
        int level1 = kp1.octave;
        //level1大于0,不是底层图像,跳出循环继续
        if(level1>0)
            continue;
        //在vbPrevMatched为初始参考帧的特征点、即F1, 搜索窗口的大小为windowsSIze，第level1层
        //知道了参考帧F1中第i1个特征点位置，在F2中一样的位置寻找临近点，序号储存在vIndices2
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);
        //如果没有搜寻到，则跳出循环继续
        if(vIndices2.empty())
            continue;
        //d1为F1中第一个第i1个特征的描述子
        cv::Mat d1 = F1.mDescriptors.row(i1);
        //最佳匹配距离
        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;
        //遍历vIndices2,vIndices2为kp1位置临近的特征点（临近点位于K2)
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            //i2为临近点的序号（临近点位于K2)
            size_t i2 = *vit;
            //提取该临近点的描述子（临近点位于K2)
            cv::Mat d2 = F2.mDescriptors.row(i2);
            //计算描述子距离
            int dist = DescriptorDistance(d1,d2);
            //vMatchedDistance维数与k2中特征点数量相同，vMatchedDistance中的元素为最大距离阈值
            if(vMatchedDistance[i2]<=dist)
                continue;
            //距离小于最佳距离
            //dist小于bestDist或者是在bestDist与bestDist2中间，然后重新赋值
            //bestDist最小距离，bestDist2为倒数第二距离
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                //将bestDist重新赋值
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                //将bestDist2重新赋值
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                //如果图2中序号bestIdx2的点已经存在匹配，则将其替换
                if(vnMatches21[bestIdx2]>=0)
                {
                    //我觉得这里有问题，因为bestIdx2如果已经有匹配，那么应该将vMatchedDistance[bestIdx2]中的值与bestDist的值进行比较，哪个小就认为哪个是bestIdx2真正的匹配点；
                    //如果vMatchedDistance[bestIdx2]小就应该直接跳出，如果bestdist小，就执行下面的内容
                    //将匹配值复原到-1
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                //vnMatches12表示从1到2, 2中离kp1最近点的序号
                vnMatches12[i1]=bestIdx2;
               // vnMatches21表示从2到1, 1中离序号bestIdx2最近的点
                vnMatches21[bestIdx2]=i1;
                //距离的容器
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;
                //mbCheckOrientation为判断是否需要进行角度检测的变量
                if(mbCheckOrientation)
                {
                    //匹配点对的角度差
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    //如果rot小于0
                    if(rot<0.0)
                        rot+=360.0f;
                    //factor = 1.0f/HISTO_LENGTH;
                    //HISTO_LENGTH为30,factor为1/30
                    int bin = round(rot*factor);
                    //rot为900时，bin才会为30,与HISTO_LENGTH相等
                    //这里不太理解
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    //assert的作用是，如果它的条件返回错误，则终止程序
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    //bin只能在0-30间
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        //提取匹配点对中相同角度出现最多的组
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
        //rotHist数组中序号表示匹配点对的角度为0-30,每一维表示该角度上匹配点对的序号
        //ind1为rotHist数组中元素最多的那一维(与最多匹配点对对应的角度)
        //ind2为rotHist数组中元素第二多的那一维
        //HISTO_LENGTH为30
        //只取那些角度差出现最多的三组点对
        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

//在闭环检测中用到的匹配
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;//提取特征点
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;//提取特征向量（节点——特征点序号）
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();//得到地标点
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;//计算描述子

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);
    //HISTO_LENGTH=30
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)//相同的节点
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)//节点的每一个
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];//遍历同一个节点中pKF1每一个特征点
                //错误的直接跳过
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;
                //该特征点的描述子
                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;
                //遍历同一个节点中pKF2每一个特征点
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];//遍历同一个节点中pKF2每一个特征点

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);//计算特征点的距离

                    if(dist<bestDist1)//找最小的
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)//找第二小的
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)//小于阈值
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))//第一小比第二小的小很多
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];//1--2的匹配关系，
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)//检测方向性
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;//计算角度差
                            if(rot<0.0)
                                rot+=360.0f;
                            //factor=1/30
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);//属于角度bin的特征点序号
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);//提取元素最多的三个数组序号，储存在ind1,ind2,ind3

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}
//特征匹配，在localmapping线程中会使用在地标点的创建过程中，其中bOnlyStereo=false
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    //提取两个关键帧的特征向量
    ////FeatureVector表示该特征点描述子对应的特征向量, 指定层相应树节点的序号-----与该节点对应特征点在当前帧序号的集合
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    //计算极线
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    //计算第一帧的光心在第二帧上相机坐标系的坐标
    cv::Mat C2 = R2w*Cw+t2w;
    //深度（第一帧关心到第二帧光心的距离）的逆
    const float invz = 1.0f/C2.at<float>(2);
    //计算第一帧的光心在第二帧像素平面上的坐标
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    //匹配点对容器的大小和第一帧特征点数量相同
    vector<int> vMatches12(pKF1->N,-1);
    //HISTO_LENGTH=30
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)//判断树节点序号是否相同
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                ////提取第一帧中相应节点下的特征点序号
                const size_t idx1 = f1it->second[i1];
                //看idx1对应处是否已經存在地标点
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                //idx1处已经存在地标点，则直接跳过
                // If there is already a MapPoint skip
                if(pMP1)
                    continue;
                //判断是不是双目
                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;
                //bOnlyStereo是false
                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                //提取第一帧的特征点
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                //其对应的描述子
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                //最小距离的阈值
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    //提取第二帧中相同节点下的特征点
                    size_t idx2 = f2it->second[i2];
                    //看第二帧中idx2位置处是否已经有地标点
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    //存在了，就直接跳过
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    //计算描述子间的距离
                    const int dist = DescriptorDistance(d1,d2);
                    //距离过大则直接跳出
                    if(dist>TH_LOW || dist>bestDist)
                        continue;
                    //提取第二帧的特征点
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
                    //单目
                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        //距离极点过近
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }
                    //检测是否满足极限条件
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    //存储匹配结果
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        //factor为1/30
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        //必须同时满足这两个条件
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            //f1it返回到一个比f2it->first略大或相等的位置
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        //计算rotHist中储存元素最多的三组，分别为ind1,ind2,ind3
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }
    //给容器vMatchedPairs赋值
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        //给pair型容器vMatchedPairs添加值
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

//地标点的融合，在localmapping中有所应用，pKF指的是共享关键帧，vpMapPoints对当前帧的地标点容器
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    //获取旋转和平移
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();
    //获取内参
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;
    //光心的世界点坐标
    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;
        //地标点是坏的或者地标点已经存在与共享关键帧中
        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;
        //获取地标点的信息
        cv::Mat p3Dw = pMP->GetWorldPos();
        //投影到相机坐标系中
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;
        //再投影到像素平面
        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //判断投影点是否超出了图像的边界
        if(!pKF->IsInImage(u,v))
            continue;
        //计算投影点在右图中的横坐标
        const float ur = u-bf*invz;
        //地标点的距离范围
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        //计算该地标点到共享关键帧光心的距离
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        //距离（地标点距离共享关键帧）必须在尺度范围以内（该尺度范围以当前关键帧为参照）
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        //地标点与光心（所有能观测到该地标点的关键帧）连线的单位向量的平均值（平均观测方向上的单位向量）
        cv::Mat Pn = pMP->GetNormal();
        //计算（地标点与共享关键帧连线）在（平均观测方向）上的投影，小于，（地标点与共享关键帧连线）距离的一半，即角度过大
        if(PO.dot(Pn)<0.5*dist3D)
            continue;
        //预测pMP在共享关键帧pKF中的层数
        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        //计算搜索半径
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        //在投影点（u,v）附近搜索特征点，其序号储存在vIndices
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //匹配最接近的特征点
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            //遍历每一个可能相似的特征点
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];
            //提取特征点所在的层数
            const int &kpLevel= kp.octave;
            //特征点的层数与之前预测的层数相差过大
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;
            //多目
            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];//右图中实际的横坐标
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;//计算投影点和特征点在右图中横坐标的差值
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            //单目
            else
            {
                //计算投影点与特征点的距离
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;
                //距离过大
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }
            //共享关键帧中该特征点的描述子
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);
            //计算距离
            const int dist = DescriptorDistance(dMP,dKF);
            //取最小的距离
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        //
        if(bestDist<=TH_LOW)
        {

            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            //如果共享关键帧bestIdx位置已经存在地标点
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    //共享关键帧中已存在的地标点被观测的次数更多
                    if(pMPinKF->Observations()>pMP->Observations())
                        //更新当前帧中的地标点
                        pMP->Replace(pMPinKF);
                    //已存在的地标点被观测的次数少
                    else
                        //更新共享关键帧中的地标点
                        pMPinKF->Replace(pMP);
                }
            }
            //如果共享关键帧bestIdx位置不存在地标点
            else
            {
                //地标点中添加被pKF关键帧的观测
                pMP->AddObservation(pKF,bestIdx);
                //共享关键帧pKF中添加地标点
                pKF->AddMapPoint(pMP,bestIdx);
            }
            //融合点数量加一
            nFused++;
        }
    }

    return nFused;
}

//地标点的融合，用于回环检测线程中
//pKF表示当前帧的共享关键帧，Scw表示该帧的矫正后的位姿矩阵，vpPoints表示回环相似帧局部地图中的地标点，th为4,vpReplacePoint
//vpReplacePoints中储存了回环帧局部地图点对应的当前共享帧地标点的信息，其维度和vpPoints相同
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    //该共享帧的内参
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);//尺度*旋转部分
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));//计算尺度因此，因为该数值在不考虑尺度的前提下应该等于一
    cv::Mat Rcw = sRcw/scw;//旋转矩阵
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;//平移向量
    cv::Mat Ow = -Rcw.t()*tcw;//光心的世界坐标

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();//该共享帧已经跟踪到的地标点

    int nFused=0;//初始化融合次数

    const int nPoints = vpPoints.size();//回环相似帧局部地图中的地标点个数

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];//遍历回环相似帧局部地图中的每一个地标点

        // Discard Bad MapPoints and already found
        //坏的或者已经被跟踪到的直接跳过
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.3维世界信息
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;//转换到相机坐标系下

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;
        //投影到图像像素平面
        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;//计算地标点到该帧的距离
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)//距离过大则跳过
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        //地标点到该帧连线 与 平均观测连线的角度cos值小于0.5（角度过大）,直接跳过
        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        //预测层数
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        //寻找该层的尺度因子并计算搜索半径
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        //在pKF中搜索距离（u，v）最近的一些特征点（序号）
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();//地标点的描述子

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;//特征点的层数

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)//满足预测要求
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);//特征点的描述子

            int dist = DescriptorDistance(dMP,dKF);//计算地标点描述子与特征点描述子间的距离

            if(dist<bestDist)//找到最小的一个特征点
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        //距离小于阈值
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)//该匹配得到的特征点已经存在值
            {
                if(!pMPinKF->isBad())//如果特征点正常
                    vpReplacePoint[iMP] = pMPinKF;//执行替换，回环帧下第iMP个地标点与地标点pMPinKF相对应
            }
            else//如果该特征点没有对应地标点
            {
                pMP->AddObservation(pKF,bestIdx);//直接添加该共享帧到地标点的观测中
                pKF->AddMapPoint(pMP,bestIdx);//添加地标点到该共享帧中第bestIdx个特征点对应的位置
            }
            nFused++;//融合次数加1
        }
    }

    return nFused;
}

//利用相似变换矩阵SIM求取更多的匹配点，th=7.5
//vpMatches12的维度和pKF1的特征点数量相同，里面储存在对应到pKF2中的地标点
//这是一个3D到2D的匹配过程
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    //提取第一个关键帧pKF1的内参矩阵
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    //得到第一帧pKF1的旋转和平移信息
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    //得到第二帧pKF2的旋转和平移信息
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    //2到1的旋转矩阵
    cv::Mat sR12 = s12*R12;
    //1到2的旋转矩阵
    cv::Mat sR21 = (1.0/s12)*R12.t();
    //1到2的平移向量
    cv::Mat t21 = -sR21*t12;
    //关键帧pKF1对应的地标点
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();
    //关键帧pKF2对应的地标点
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];//提取第i个特征点对应的匹配点
        if(pMP)//存在匹配好的特征点对
        {
            vbAlreadyMatched1[i]=true;//标记第i个地标点已经匹配完成
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);//得到该匹配点对在关键帧pKF2中的序号
            if(idx2>=0 && idx2<N2)//序号在范围之内
                vbAlreadyMatched2[idx2]=true;//标记第idx个地标点已经完成匹配
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];
        //不考虑已经匹配完成的地标点
        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;
        //pKF1中未经匹配的地标点位姿
        cv::Mat p3Dw = pMP->GetWorldPos();
        //投影到pKF1的相机坐标系中
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        //通过相似变换矩阵投影到pKF2的相机坐标系中
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        //必须保证深度值为正的
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;
        //计算第一帧的地标点在第二帧头像上的投影
        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //满足图像的区域要求
        if(!pKF2->IsInImage(u,v))
            continue;
        //地标点的
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);//地标点距离pKF2光心的距离

        // Depth must be inside the scale invariance region
        //深度值不在阈值中
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        //计算地标点在pKF2中的层数
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        //计算搜索半径
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];
        //得到投影点附近pKF2帧特征点的序号
        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];//提取投影点附近的特征点

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)//只取在预测层数上的特征点
                continue;
            //得到附近特征点在第二帧的描述子
            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);
            //计算特征点相似度距离
            const int dist = DescriptorDistance(dMP,dKF);
            //提取最小值
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }
        //最小值要小于阈值
        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;//pKF1中第i1个地标点对应pKF2中第bestIdx个特征点
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];//遍历所有的在pKF2中的地标点

        if(!pMP || vbAlreadyMatched2[i2])//没有值或已经存在匹配点对
            continue;

        if(pMP->isBad())
            continue;
        //提取pKF2中的地标点
        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        //投影到pKF1相机坐标系中
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;
        //投影到pKF1的相平面中
        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        //计算地标点到pKF1光心的距离
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        //满足距离要求
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        //预测尺度信息
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        //计算搜索半径
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];
        //在pKF1中寻找投影点附近的特征点，提取序号
        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //计算描述子
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;//遍历每一个待匹配的pKF1中的特征点

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)//判断是否满足层数约束
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);//计算描述子间距离

            if(dist<bestDist)//提取最相似的特征点
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;//储存序号
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];//pKF1中第i1个地标点对应pKF2中第idx2个地标点

        if(idx2>=0)//通过SIM匹配到了，且之前没有匹配到
        {
            int idx1 = vnMatch2[idx2];//反取，pKF2中第idx2个地标点对应pKF1中第idx1个地标点
            if(idx1==i1)//来反验证
            {
                vpMatches12[i1] = vpMapPoints2[idx2];//pKF1中第i1个地标点位置对应了pKF2中第idx2个地标点
                nFound++;
            }
        }
    }

    return nFound;//额外找到的匹配点对数量
}

//上一帧和当前帧的投影匹配关系（注意此处有函数的重载）
//提取上一帧的地标点,然后利用当前帧初始的位姿将地标点投影到当前帧上,寻找每一个地标点的最佳匹配特征点.....
//然后将匹配成功的地标点放入特征点序号对应的mvpMapPoints中
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    //rotHist一共有HISTO_LENGTH维，每一维都为一个vector<int>型的容器
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        //给每一维的容器都预留500的空间
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    //当前帧的旋转和平移
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    //旋转的转置乘平移向量(旋转矩阵是正交矩阵)
    const cv::Mat twc = -Rcw.t()*tcw;

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);
    //tlc表示从当前帧到前一帧的平移
    const cv::Mat tlc = Rlw*twc+tlw;
    //不是单目, 且平移tlc中的第三个元素大于当前帧的基线(第三个元素为正)
    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    //不是单目, 且平移tlc中第三个元素的相反数大于当前帧的基线(第三个元素为负)
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

    for(int i=0; i<LastFrame.N; i++)
    {
        //提取前一帧的地标点
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                //不是异常点
                // Project
                //提取地标点的世界坐标
                cv::Mat x3Dw = pMP->GetWorldPos();
                //投影到当前帧中
                cv::Mat x3Dc = Rcw*x3Dw+tcw;
                //x3Dc的x, y坐标
                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                //深度的倒数
                const float invzc = 1.0/x3Dc.at<float>(2);
                //为负, 跳出循环
                if(invzc<0)
                    continue;
                //将x3Dc投影到像素平面中
                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
                //水平像素超过边界
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                //垂直像素超过边界
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;
                //提取该地标点对应特征点所在的金字塔层
                int nLastOctave = LastFrame.mvKeys[i].octave;

                // Search in a window. Size depends on scale
                //th为阈值, 当前帧在此层的尺度因子
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;
                //bForward为真,前移, (立体)，往金字塔顶端搜索
                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                //bBackward为真,后移, (立体)，往金字塔底端搜索
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                //单目
                else
                    //在u,v附近,在nlastOctave-1层到nLastOctive+1层, 搜寻临近的特征点, 其序号储存在vIndices2
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;
                //遍历(u,v)附近的特征点
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;
                    //正常情况下地标点一般为空
                    //附近特征点的地标点
                    if(CurrentFrame.mvpMapPoints[i2])
                        //这些地标点的观测次数大于0，说明已经存在的地标点
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;
                    //如果是立体视觉
                    if(CurrentFrame.mvuRight[i2]>0)
                    {
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)//投影点右图横坐标和特征点右图横坐标的差
                            continue;
                    }
                    //该特征点的描述子
                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);
                    //计算描述子间的距离
                    const int dist = DescriptorDistance(dMP,d);
                    //选取最小的距离
                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH)
                {
                    //将地标点赋值给当前帧的地标点容器中
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;
                    //匹配特征点的角度检测
                    if(mbCheckOrientation)
                    {
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        //找到rotHist中数量最多的三个容器序号
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

//另一种投影搜索匹配的方法（存在相关的地标点信息）
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;
    //提取当前帧的旋转矩阵和平移向量
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    //当前帧光心在世界坐标系下的坐标
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    //得到关键帧的地标点
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        //提取每一个地标点
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //地标点准确且在地图中未被找到
                //Project
                //得到地标点的位姿并进行投影
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);
                //投影到当前帧像素平面中
                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                //计算当前帧光心与三维世界点的距离
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);
                //得到地标点的距离
                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;
                //预测该地标点在当前帧的第几层
                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                //得到搜索窗口的半径
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];
                //在投影点附近搜索符合距离阈值的匹配点
                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    //如果已经有匹配结果，则跳出
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        //找到最佳的匹配点
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    //小于阈值则认为匹配成功， 找到匹配当前帧中第bestIdx2个特征点对应的地标点
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        //方向检查
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}
//ComputeThreeMaxima，L为histo的维数
//histo中哪一维元素的数量最多, 最大值储存在max1中,对应序号储存在ind1
//第二大储存在max2中,对应序号储存在ind2
//第三大储存在max3中,对应序号储存在ind3
//vector<int>* 表示定义了一个vector<int>型的数组,数组的每一个元素都是vector<int>
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            //从1向后推移一个值
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            //从2向后推移一个值
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            //直接对3赋值
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
