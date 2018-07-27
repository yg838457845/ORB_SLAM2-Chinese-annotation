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


#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2
{


Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale):
    mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    vector<MapPoint*> vpKeyFrameMP1 = pKF1->GetMapPointMatches();//pKF1的地标点

    mN1 = vpMatched12.size();//匹配点对的数量

    mvpMapPoints1.reserve(mN1);
    mvpMapPoints2.reserve(mN1);
    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    cv::Mat Rcw1 = pKF1->GetRotation();
    cv::Mat tcw1 = pKF1->GetTranslation();
    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat tcw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(mN1);

    size_t idx=0;
    for(int i1=0; i1<mN1; i1++)
    {
        if(vpMatched12[i1])
        {
            MapPoint* pMP1 = vpKeyFrameMP1[i1];
            MapPoint* pMP2 = vpMatched12[i1];

            if(!pMP1)
                continue;

            if(pMP1->isBad() || pMP2->isBad())
                continue;

            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);//在关键帧pKF1中地标点pMP1的序号
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);//在关键帧pKF2中地标点pMP2的序号

            if(indexKF1<0 || indexKF2<0)
                continue;
            //特征点
            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];
            //特征点在第几层
            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            //对应层最大的误差
            mvnMaxError1.push_back(9.210*sigmaSquare1);
            mvnMaxError2.push_back(9.210*sigmaSquare2);
            //储存地标点对
            mvpMapPoints1.push_back(pMP1);
            mvpMapPoints2.push_back(pMP2);
            mvnIndices1.push_back(i1);//储存匹配好的特征点对的序号
            //pKF1中地标点的三维坐标
            cv::Mat X3D1w = pMP1->GetWorldPos();
            //转换到相机坐标系
            mvX3Dc1.push_back(Rcw1*X3D1w+tcw1);
            //pKF2中地标点的三维点
            cv::Mat X3D2w = pMP2->GetWorldPos();
            //转换到相机坐标系
            mvX3Dc2.push_back(Rcw2*X3D2w+tcw2);
            //mvAllIndices储存了特征点对的序号（直接从1排到最后一个，相当于计数）
            mvAllIndices.push_back(idx);
            idx++;
        }
    }
    //内参矩阵
    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    FromCameraToImage(mvX3Dc1,mvP1im1,mK1);//计算投影点
    FromCameraToImage(mvX3Dc2,mvP2im2,mK2);

    SetRansacParameters();
}

//设置参数
void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;//0.99
    mRansacMinInliers = minInliers;//20
    mRansacMaxIts = maxIterations;//300

    N = mvpMapPoints1.size(); // number of correspondences， 匹配点对的数量

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));//ceil返回大于或者等于指定表达式的最小整数, mRansacProb=0.99, pow计算epsilon的3次幂

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    mnIterations = 0;
}

//迭代, nIterations=5
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    //mN1为当前帧特征点的数量
    vbInliers = vector<bool>(mN1,false);
    nInliers=0;//内点数量

    if(N<mRansacMinInliers)//mRansacMinInliers=20, N为匹配点对的数量
    {
        //特征点对的数量过少
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    cv::Mat P3Dc1i(3,3,CV_32F);
    cv::Mat P3Dc2i(3,3,CV_32F);

    int nCurrentIterations = 0;
    //mnIterations为0, nIterations=6
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;

        vAvailableIndices = mvAllIndices;//匹配点对的序号

        // Get min set of points
        for(short i = 0; i < 3; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);//在范围[0----vAvailableIndices.size()-1]随机数

            int idx = vAvailableIndices[randi];//提取一个匹配点对的序号

            mvX3Dc1[idx].copyTo(P3Dc1i.col(i));//将pKF1地标点在相机坐标系下位置储存到P3Dc1i
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));//将pKF2地标点在相机坐标系下位置储存到P3Dc2i

            vAvailableIndices[randi] = vAvailableIndices.back();//删去提取的随机数,即不重复随即采样
            vAvailableIndices.pop_back();
        }

        ComputeSim3(P3Dc1i,P3Dc2i);//计算相似变换矩阵

        CheckInliers();//检查内点
        //提取有最多内点的迭代过程
        if(mnInliersi>=mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            if(mnInliersi>mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for(int i=0; i<N; i++)//N为特征点对的数量
                    if(mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;//mvnIndices1[i]储存的是:第i个特征点对在当前帧中对应的特征点序号
                return mBestT12;//返回pKF1和pKF2之间的相似变换矩阵
            }
        }
    }

    if(mnIterations>=mRansacMaxIts)//跌代已经超了最大的迭代次数
        bNoMore=true;

    return cv::Mat();
}

cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers12,nInliers);
}

void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    cv::reduce(P,C,1,CV_REDUCE_SUM);//将矩阵P按行累加——>C
    //求平均
    C = C/P.cols;

    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i)=P.col(i)-C;//减去均值, P中储存偏差
    }
}

//计算相似变换矩阵Sim3
//P1中储存了pKF1地标点在相机坐标系下位置
//P2中储存了pKF2地标点在相机坐标系下位置
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates

    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

    ComputeCentroid(P1,Pr1,O1);//O1为P1按行累加的平均, Pr1为减去均值的偏差
    ComputeCentroid(P2,Pr2,O2);//O2为P2按行累加的平均, Pr2为减去均值的偏差

    // Step 2: Compute M matrix
    //偏差矩阵的相乘
    cv::Mat M = Pr2*Pr1.t();

    // Step 3: Compute N matrix

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
    N12 = M.at<float>(1,2)-M.at<float>(2,1);
    N13 = M.at<float>(2,0)-M.at<float>(0,2);
    N14 = M.at<float>(0,1)-M.at<float>(1,0);
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    // Step 4: Eigenvector of the highest eigenvalue

    cv::Mat eval, evec;
    //计算特征值和特征向量，只有第一个特征值是正的（也是最大的），所以只管第一行，它就是最佳的旋转向量（四元数）
    cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation

    cv::Mat vec(1,3,evec.type());
    //提取第一行的特征向量后三位（范围的左边界，而不取右边界）
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    //第一行特征向量的第一个元素与后面三个元素向量模的夹角，得到四元数角的一半
    double ang=atan2(norm(vec),evec.at<float>(0,0));
    //2*ang为旋转角度，vec/norm(vec)为旋转方向
    vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half

    mR12i.create(3,3,P1.type());
    //根据洛德李德公式计算旋转矩阵
    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis

    // Step 5: Rotate set 2
    //旋转矩阵乘质心化后的第二帧地标点坐标
    cv::Mat P3 = mR12i*Pr2;

    // Step 6: Scale

    if(!mbFixScale)//单目
    {
        //即在这里计算了（求和符号）Pr1*R12*Pr2
        double nom = Pr1.dot(P3);//整个Mat矩阵扩展成一个行向量，之后执行向量的点乘运算
        cv::Mat aux_P3(P3.size(),P3.type());//也是3*3维的矩阵
        aux_P3=P3;
        cv::pow(P3,2,aux_P3);//求平方
        double den = 0;

        for(int i=0; i<aux_P3.rows; i++)
        {
            for(int j=0; j<aux_P3.cols; j++)
            {
                den+=aux_P3.at<float>(i,j);//求矩阵的和，其实这里对应文章中：（求和符号）质心化后的第二帧地标点坐标的模值平方和
            }
        }

        ms12i = nom/den;//得到尺度信息
    }
    else//双目的尺度系数直接为1
        ms12i = 1.0f;

    // Step 7: Translation

    mt12i.create(1,3,P1.type());
    mt12i = O1 - ms12i*mR12i*O2;//计算平移，利用坐标的均值

    // Step 8: Transformation

    // Step 8.1 T12
    mT12i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sR = ms12i*mR12i;

    sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));//赋值旋转矩阵
    mt12i.copyTo(mT12i.rowRange(0,3).col(3));//赋值平移矩阵

    // Step 8.2 T21

    mT21i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sRinv = (1.0/ms12i)*mR12i.t();

    sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv*mt12i;
    tinv.copyTo(mT21i.rowRange(0,3).col(3));
}


void Sim3Solver::CheckInliers()
{
    vector<cv::Mat> vP1im2, vP2im1;
    Project(mvX3Dc2,vP2im1,mT12i,mK1);//将第二帧的地标点利用相似变换矩阵投影到第一帧图像平面山
    Project(mvX3Dc1,vP1im2,mT21i,mK2);//将第一帧的地标点利用相似变换矩阵投影到第二帧图像平面山

    mnInliersi=0;
    //mvP1im1是将第一帧的地标点利用相似变换矩阵投影到第一帧图像平面山
    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i]-vP2im1[i];//2—1投影点和1本身的投影点之间的距离差
        cv::Mat dist2 = vP1im2[i]-mvP2im2[i];//1—2投影点和2本身的投影点之间的距离差

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])//距离差均满足阈值
        {
            mvbInliersi[i]=true;//该匹配点对对应的位置为内点，判定为真
            mnInliersi++;//内点数量加1
        }
        else
            mvbInliersi[i]=false;
    }
}


cv::Mat Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation.clone();
}

float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}

void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;
        const float invz = 1/(P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0)*invz;
        const float y = P3Dc.at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
{
    const float &fx = K.at<float>(0,0);
    const float &fy = K.at<float>(1,1);
    const float &cx = K.at<float>(0,2);
    const float &cy = K.at<float>(1,2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
    {
        const float invz = 1/(vP3Dc[i].at<float>(2));
        const float x = vP3Dc[i].at<float>(0)*invz;
        const float y = vP3Dc[i].at<float>(1)*invz;

        vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
    }
}

} //namespace ORB_SLAM
