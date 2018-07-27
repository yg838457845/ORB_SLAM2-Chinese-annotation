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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{

//mpmap的值随着tracking线程的不断进行, 便同时便在不断更新
MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];//2
    mCameraSize = fSettings["Viewer.CameraSize"];//0.08
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

}

//画地标点
void MapDrawer::DrawMapPoints()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();//得到地图中所有的地标点
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();//得到当前局部地图中的地标点

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);//地标点的绘制大小
    //开始绘制点
    glBegin(GL_POINTS);
    //设置颜色,黑色
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))//地标点在局部地图中出现了,直接跳过继续(即提取非局部地图中的地标点)
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();//得到地标点的世界坐标
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));//设置画图的顶点
    }
    //结束绘制点
    glEnd();

    glPointSize(mPointSize);//地标点的绘制大小
    glBegin(GL_POINTS);//开始绘制点
    glColor3f(1.0,0.0,0.0);//红色

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)//遍历局部地图中的地标点
    {
        if((*sit)->isBad())
            continue;
        cv::Mat pos = (*sit)->GetWorldPos();//得到地标点的世界坐标
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));//设置画图的顶点

    }
    //结束绘制
    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;//0.05
    const float h = w*0.75;
    const float z = w*0.6;

    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();//得到地图中的所有关键帧

    if(bDrawKF)//为真
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twc = pKF->GetPoseInverse().t();//关键帧位姿的逆

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));//与栈顶的位姿信息相乘(右乘), 栈顶就是下面绘制的矩形信息

            glLineWidth(mKeyFrameLineWidth);//设置线宽
            glColor3f(0.0f,0.0f,1.0f);//设置颜色,蓝色
            glBegin(GL_LINES);//设置起点
            //以下是画矩形, (w,h,z), 相机坐标系下的
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            //绘制结束
            glEnd();
            //将栈顶矩阵弹出
            glPopMatrix();
        }
    }

    if(bDrawGraph)//画图为真
    {
        glLineWidth(mGraphLineWidth);//线宽
        glColor4f(0.0f,1.0f,0.0f,0.6f);//绿色
        glBegin(GL_LINES);//开始绘制

        for(size_t i=0; i<vpKFs.size(); i++)//遍历地图中的每一个关键帧
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);//得到关键帧权值大于100的邻接关键帧
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();//得到关键帧的光心坐标
            if(!vCovKFs.empty())//如果没有邻接关键帧
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)//遍历每一个邻接关键帧
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)//邻接帧的ID小于关键帧的ID,直接跳过(避免重复)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();//邻接帧的光心
                    //画出邻接关键帧与其邻接帧的光心连线
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();//关键帧的父节点
            if(pParent)//存在父节点
            {
                cv::Mat Owp = pParent->GetCameraCenter();//得到父节点的光心
                //画关键帧到到其父节点的连线
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();//关键帧的父节点
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)//回环帧的ID小于关键帧的ID,直接跳过(避免重复)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();//回环帧的光心
                //绘制关键帧到回环帧的连线
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }
        //绘制结束
        glEnd();
    }
}

//绘制相机位姿
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;//0.08
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();//矩阵堆栈的压入操作

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);//用当前矩阵与任意矩阵相乘
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);//设定光栅线段的宽
    glColor3f(0.0f,1.0f,0.0f);//设置颜色,绿色
    glBegin(GL_LINES);//定义一个或一组原始的顶点
    //下面应该是画相机
    glVertex3f(0,0,0);//定义一个顶点
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();//矩阵堆栈的弹出操作
}

//往地图绘制指针类中加入帧位姿变量（一般局部建图线程结束后加入位姿信息）
void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

//把位姿信息储存在M中
void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())//类中存在位姿信息
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();//提取位姿的旋转（逆矩阵，从相机到世界坐标系）
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);//提取位姿的平移（从相机到世界坐标系）
        }
        //M为从相机到世界坐标系的位姿变换矩阵
        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

} //namespace ORB_SLAM
