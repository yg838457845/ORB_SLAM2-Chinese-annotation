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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

namespace ORB_SLAM2
{


//BA优化函数
//pMap为地图，nIterations迭代次数，pbStopFlag为是否停止BA，nLoopKF帧的序号，bRobust为false
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();//所有的关键帧
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();//所有的地标点
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);//BA优化
}

//BA的优化函数
void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());//开辟一个地标点数量大小的容器

    g2o::SparseOptimizer optimizer;//离散优化器
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;//6为优化变量维度，3为误差项的维度

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();//初始化线性优化器

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);//块优化器

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);//LM非线性优化法
    optimizer.setAlgorithm(solver);//将线性求解器添加到优化器中

    if(pbStopFlag)//如果需要停止
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    ///////////////////////////////////////////////////////////////////////// Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];//遍历地图中每一个关键帧
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();//定义顶点
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));//将关键帧的位姿添加到顶点中
        vSE3->setId(pKF->mnId);//顶点的序号
        vSE3->setFixed(pKF->mnId==0);//设置初始关键帧为固定的
        optimizer.addVertex(vSE3);//添加顶点
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;//最大的关键帧序号maxKFid
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    ///////////////////////////////////////////////////////////////////////////// Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];//地标点
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();//定义顶点
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));//将地标点的坐标添加到顶点中
        const int id = pMP->mnId+maxKFid+1;//地标点的序号
        vPoint->setId(id);//设置顶点序号
        vPoint->setMarginalized(true);//边缘化过程为真
        optimizer.addVertex(vPoint);//添加顶点

       const map<KeyFrame*,size_t> observations = pMP->GetObservations();//能观测到地标点pMP的关键帧

        int nEdges = 0;//边的数量
        //////////////////////////////////////////////////////////////////////SET EDGES
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            KeyFrame* pKF = mit->first;//遍历能观测到该地标点的关键帧
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;//边的数量+1

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];//地标点在该关键帧pKF中对应的特征点

            if(pKF->mvuRight[mit->second]<0)//单目
            {
                Eigen::Matrix<double,2,1> obs;//观测
                obs << kpUn.pt.x, kpUn.pt.y;//输入观测

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();//初始边

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));//边中添加地标点的顶点
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));//边中添加关键帧的顶点
                e->setMeasurement(obs);//边的观测
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];//地标点在帧pKF中的尺度（逆）
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);//设置信息矩阵
                //bRobust为FALSE
                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;//边的内参
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);//添加边
            }
            else//双目相机
            {
                Eigen::Matrix<double,3,1> obs;//三维观测
                const float kp_ur = pKF->mvuRight[mit->second];//特征点（序号是mit->second）在关键帧pKF中右图的横坐标
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;//特征点坐标与右图中横坐标

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));//边中地标点的顶点
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));//边中关键帧的顶点
                e->setMeasurement(obs);//设置观测
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];//地标点在帧pKF中的尺度（逆）
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;//信息矩阵
                e->setInformation(Info);

                if(bRobust)//bRobust为false
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }
        //如果这个边不存在
        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);//删除了顶点（地标点）
            vbNotIncludedMP[i]=true;//第i个地标点不被包括
        }
        else
        {
            vbNotIncludedMP[i]=false;//第i个地标点被包括
        }
    }

    //////////////////////////////////////////////////////////////////////// Optimize!
    /// //开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    /////////////////////////////////////////////////////////////////////////Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];//遍历所有的关键帧
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));//提取该关键帧的顶点
        g2o::SE3Quat SE3quat = vSE3->estimate();//得到关键帧的位姿
        if(nLoopKF==0)//如果是初始关键帧
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));//更新初始关键帧的位姿
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);//更新pKF->mTcwGBA
            pKF->mnBAGlobalForKF = nLoopKF;//表示帧pKF依靠第nLoopKF帧进行BA优化更新
        }
    }

    /////////////////////////////////////////////////////////////////////Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];//遍历每一个地标点

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));//提取该地标点的顶点

        if(nLoopKF==0)//如果是初始关键帧
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));//更新地标点的坐标
            pMP->UpdateNormalAndDepth();//更新地标点的深度与方向
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);//更新pMP->mPosGBA
            pMP->mnBAGlobalForKF = nLoopKF;//表示地标点pMP依靠第nLoopKF帧进行BA优化更新
        }
    }

}

//位姿优化函数
int Optimizer::PoseOptimization(Frame *pFrame)
{
    //离散优化器
    g2o::SparseOptimizer optimizer;
    //线性求解器
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    //初始化线性求解器
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    //求解器指针
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    //定义梯度下降方法，LM
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //设置求解器
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    //位姿顶点
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    //图中添加顶点
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;
    //边的容器
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    //size_t类型的容器
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);
    //立体视觉的边容器
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        //地标点
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            //单目的情况下
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;
                //2维的观测点
                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                //将特征点坐标赋值给obs
                obs << kpUn.pt.x, kpUn.pt.y;
                //定义一个边e
                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                //向边e中添加了(序号为0)的位姿顶点
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                //边中的观测
                e->setMeasurement(obs);
                //提取该特征点所在金字塔层的方差信息
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                //边中添加对角信息矩阵
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                //定义核函数
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                //定义核函数定义域中的界
                rk->setDelta(deltaMono);
                //将内参赋值给边
                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                //地标点的世界坐标
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);
                //图中加入边
                optimizer.addEdge(e);
                //向边容器中加入该边
                vpEdgesMono.push_back(e);
                //对应边的序号加入容器中
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation，双目
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];//特征点右图的横坐标
                //三维观测点
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        //将位姿顶点的位姿转换为李代数
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        //10次迭代
        optimizer.optimize(its[it]);

        nBad=0;
        //vpEdgesMono为边的容器
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            //取一条边
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];
            //提取边对应的序号
            const size_t idx = vnIndexEdgeMono[i];
            //如果是异常点
            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }
            //_error.dot(info*_error), 误差*信息*误差, 误差在单目中应该是2维的
            const float chi2 = e->chi2();
            //大于核函数的阈值
            if(chi2>chi2Mono[it])
            {
                //该地标点为异常点
                pFrame->mvbOutlier[idx]=true;
                //e的level设置为1
                e->setLevel(1);
                nBad++;
            }
            else
            {
                //小于阈值则为非异常值点
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                //第三次迭代, 取消核函数
                e->setRobustKernel(0);
        }
        //有深度信息的输入
        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }
        //如果图中的边数量小于10
        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}

//局部BA优化
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;

    lLocalKeyFrames.push_back(pKF);
    //mnBALocalForKF表示是这个局部BA地图的序号
    pKF->mnBALocalForKF = pKF->mnId;
    //提取帧pKF的共享关键帧
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        //遍历所有的共享关键帧
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            //往局部关键帧中添加帧
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        //遍历局部地图中的所有帧，并提取每一帧对应的地标点
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        //将该地标点的局部BA地图的序号赋值, 即一个地标点只能储存一次
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    //遍历所有的局部BA地图地标点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();//能观测到该地标点的关键帧
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;//遍历每一个可观察到局部地标点的关键帧
            //关键帧不在局部关键帧的队列中，且未经过处理
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;//标记为非直接相连的局部关键帧
                if(!pKFi->isBad())
                    //提取那些可以观测到局部地图地标点但又不属于局部关键帧
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    //初始化pbStopFlag为false
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    //遍历局部关键帧
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        //固定当前关键帧的信息
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            //提取最大的ID
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    //遍历那些非直接的局部共享关键帧
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        //貌似设置这些定位为固定的
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    //设置尺寸大小
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();
    //边
    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    //关键帧
    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    //地标点
    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);
    //立体的
    ///////////////////////////////////////////////////////////////////////////////
    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);
    //////////////////////////////////////////////////////////////////////////////
    //单目的阈值
    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        //遍历局部地标点
        MapPoint* pMP = *lit;
        //创建顶点
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        //三维点坐标的类型转换
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        //地标点序号要跳出关键帧的序号范围
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        //边缘化为true
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        //可以观察到该地标点的关键帧
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;//可以观察到该地标点的关键帧（注意在优化过程中，那些非直接的临近局部关键帧是保持固定的），即只优化了与当前帧直接相连的局部关键帧

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];//该地标点对应的特征点

                // Monocular observation单目
                if(pKFi->mvuRight[mit->second]<0)
                {
                    //定义观测
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));// 边中的地标点
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));//边中的关键帧
                    e->setMeasurement(obs);//边的观测
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];//该地标点在帧pKFi中的尺度方差逆
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;//内核函数
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    //优化器添加边
                    optimizer.addEdge(e);
                    //边的集合
                    vpEdgesMono.push_back(e);
                    //关键帧的集合，会有重复
                    vpEdgeKFMono.push_back(pKFi);
                    //地标点的集合
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];//提取地标点在关键帧pKFi中特征点的右图横坐标
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;//三维观测量

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);//设置边的观测
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];//尺度的逆
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }
    //pbStopFlag为真
    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);
    //一般都为真
    bool bDoMore= true;
    //需要停止
    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;
    //进行更多的迭代
    if(bDoMore)
    {

    // Check inlier observations
    //遍历所有的边
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;
        //排除误差过大或者不是正的边
        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }
        //重新定义核函数
        e->setRobustKernel(0);
    }
    //立体的不管
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    // Optimize again without the outliers
    //再次进行优化，但是不用level1的边
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    //遍历每一个边
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];//错误的边
            vToErase.push_back(make_pair(pKFi,pMP));//将错误边对应的地标点和关键帧储存在vToErase
        }
    }
    //立体的
    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);
    //删除错误的关键帧和地标点
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            //除去关键帧pKFi中不应该存在的地标点pMPi
            pKFi->EraseMapPointMatch(pMPi);
            //除去地标点pMPi中不应该观测到它的关键帧pKFi
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes
    //遍历所有的局部关键帧
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        //得到优化后的关键帧信息
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    //遍历所有的局部地标点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        //获取地标点信息
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}



//闭环检测中的位姿图优化函数
//pLoopKF表示回环帧，pCurKF表示当前关键帧，NonCorrectedSim3是未经修正的当前帧局部地图的位姿信息，
//CorrectedSim3是修正后的当前帧局部地图的位姿信息，LoopConnections中储存了当前时刻共享关键帧的邻接关键帧(权值小于15) 和回环帧局部地图中的关键帧，单目bFixScale为false
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);//// 不要输出调试信息
    //线性方程求解器，每个误差项优化变量维度为7,误差值维度为3
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();//typedef BlockSolver< BlockSolverTraits<7, 3> > BlockSolver_7_3;
    //矩阵块求解器
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    //梯度下降方法
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);//设置一个合适的lamda
    optimizer.setAlgorithm(solver);//设置求解器

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();//得到地图中的所有关键帧
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();//得到地图中的所有地标点

    const unsigned int nMaxKFid = pMap->GetMaxKFid();//地图中最大关键帧的序号

    //aligned_allocator管理C++中的各种数据类型的内存
    //但是在Eigen管理内存和C++11中的方法是不一样的，所以需要单独强调元素的内存分配和管理。
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);//原始位姿
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);//矫正后的位姿
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);//顶点容器

    const int minFeat = 100;

    // Set KeyFrame vertices
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////关键帧位姿顶点的设置
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];//遍历地图中所有的关键帧
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;//关键帧的ID

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);//容器的迭代器，找到pKF对应的矫正后的位姿，

        if(it!=CorrectedSim3.end())//该关键帧的位姿被矫正过，其实就是看该帧是否在当前帧的局部地图中（回环融合之前）
        {
            vScw[nIDi] = it->second;//容器中添加位姿信息（对应第nIDi个关键帧）
            VSim3->setEstimate(it->second);//加入顶点中
        }
        else//该关键帧的位姿信息没有被矫正过
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());//关键帧的旋转
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());//平移
            g2o::Sim3 Siw(Rcw,tcw,1.0);//得到相似变换矩阵
            vScw[nIDi] = Siw;//容器中添加位姿信息（对应第nIDi个关键帧）
            VSim3->setEstimate(Siw);//加入顶点
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);//回环帧的位姿固定

        VSim3->setId(nIDi);//给顶点复制id，即第nIDi个关键帧对应的顶点
        VSim3->setMarginalized(false);//不进行边缘化
        VSim3->_fix_scale = bFixScale;//尺度不固定

        optimizer.addVertex(VSim3);//添加顶点到求解器

        vpVertices[nIDi]=VSim3;//位姿顶点的容器
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();//7*7的矩阵初始化为单位阵
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////设置边
    /// 设置回环边（当前关键帧地图与回环帧地图中相似的帧《阈值大于100》），使用的是矫正后的位姿
    // Set Loop edges
    //设置回环边
    //LoopConnections为共享关键帧对应它的邻接关键帧(权值小于15)，以及回环帧局部地图中的关键帧
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;//当前关键帧以及共享关键帧中的一个
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;//权值小于15的一个（pKF的）邻接关键帧或者是在回环帧地图中的相似关键帧
        const g2o::Sim3 Siw = vScw[nIDi];//当前帧或共享关键帧pKF的位姿
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            //当前帧和回环帧同时出现，或者有权重大于100的两帧，程序才能执行下去
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];//回环帧的位姿
            const g2o::Sim3 Sji = Sjw * Swi;//当前帧到回环帧的位姿转换矩阵

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));//以回环帧为边的一个顶点
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));//以当前帧为边的一个顶点
            e->setMeasurement(Sji);//位姿转换矩阵

            e->information() = matLambda;//7*7阶单位阵为信息矩阵

            optimizer.addEdge(e);//优化求解器中添加边

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));//边的顶点序号对
        }
    }

    // Set normal edges
    //////////////////////////////////////////////////////////////////////////////////设置普通边，不考虑位姿矫正后的信息（CorrectedSim3），考虑初始求解得到的位姿信息（NonCorrectedSim3）
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];//遍历地图中所有的关键帧

        const int nIDi = pKF->mnId;//帧的ID

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);//未经过矫正的位姿容器中寻找pKF，其实就是看该帧是否在当前帧的局部地图中（回环融合之前）

        if(iti!=NonCorrectedSim3.end())//该帧是在当前帧的局部地图中
            Swi = (iti->second).inverse();//未矫正的位姿的逆
        else
            Swi = vScw[nIDi].inverse();//正常求解得到的位姿

        KeyFrame* pParentKF = pKF->GetParent();//pKF的父节点

        // Spanning tree edge  搜索树
        /////////////////////////////////
        ///第一种普通边，即考虑两个关键帧，其有最大权值（即最佳共享关键帧之间的边）
        if(pParentKF)//存在父节点，即共享最多地标点的关键帧
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())//父节点是在当前帧的局部地图中
                Sjw = itj->second;//父节点未经矫正的位姿
            else
                Sjw = vScw[nIDj];//父节点正常求解得到的位姿

            g2o::Sim3 Sji = Sjw * Swi;//当前帧到父节点的相对位姿

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));//以父节点为边的一个顶点
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));//以地图中的一帧为边的一个顶点
            e->setMeasurement(Sji);//观测

            e->information() = matLambda;//单位阵为信息矩阵
            optimizer.addEdge(e);//添加边
        }

       //////////////////////////////////////////////////////////////////////////////// // Loop edges，第二种边，以前出现过的回环边
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();//得到回环边，此时当前帧的回环边还不能检测到，因为位姿图优化完了后才将其加入到回环帧容器中
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;//遍历pKF的回环帧
            if(pLKF->mnId<pKF->mnId)//回环帧发生在pKF之前
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);//未经过矫正的位姿容器中寻找pKF，其实就是看该帧是否在当前帧的局部地图中（回环融合之前）

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;//得到回环帧的位姿（未矫正）
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;//pKF帧到其回环帧的位姿转换矩阵
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));//以回环帧为边的一个顶点
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));//以pFK为边的另一个顶点
                el->setMeasurement(Sli);//设置观测量
                el->information() = matLambda;//边的信息矩阵，单位阵
                optimizer.addEdge(el);//添加边
            }
        }

        //// Covisibility graph edges
        /// ///////////////////////////第三种边，考虑权值大于100的两个关键帧，且不是之前的两种情况
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);//提取pKF帧的邻接帧（权值大于100)
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;//pKF帧的权值大于100的邻接帧
            //存在权值大于100的邻接帧pKFn，且其不是帧pKF的父节点（即最多共享关键帧），且pKFn不是pkF的子节点，且pKFn不是pKF的回环边
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)//如果pKFn比pKF出现的晚
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))//这两个满足要求的帧之前出现过（因为当前帧和回环帧就满足上面两个条件，需排除）
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);//未经过矫正的位姿容器中寻找pKFn，其实就是看该帧是否在当前帧的局部地图中（回环融合之前）

                    if(itn!=NonCorrectedSim3.end())//在局部地图中，赋值未矫正过的位姿
                        Snw = itn->second;
                    else//不在局部地图中
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;//得到pKF到其权值大于100的邻接帧pKFn的位姿转换关系

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();//初始化边
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));//添加权值大于100的邻接帧pKFn顶点
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));//添加pKF为另一个顶点
                    en->setMeasurement(Sni);//设置观测
                    en->information() = matLambda;//信息矩阵初始化单位阵
                    optimizer.addEdge(en);//添加边
                }
            }
        }
    }

    // Optimize!
    /////////////////////////////////////////////////进入优化过程
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    ///////////////////////////////////////////////////////得到所有优化后的位姿信息
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];//遍历地图中所有的关键帧

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));//得到优化后的顶点
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();//优化后的位姿
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();//储存位姿的逆到容器中
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();//旋转
        Eigen::Vector3d eigt = CorrectedSiw.translation();//平移
        double s = CorrectedSiw.scale();//尺度

        eigt *=(1./s); //[R t/s;0 1]，保证旋转，尺度归一

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);//得到旋转和平移

        pKFi->SetPose(Tiw);//设置pKF帧的位姿
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    ////////////////////利用矫正后的关键帧位姿信息与更新所有地标点的信息
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];//遍历所有的地标点

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)//该地标点已经被矫正过
        {
            nIDr = pMP->mnCorrectedReference;//表示地标点被哪一帧矫正过
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();//获得地标点的参考关键帧
            nIDr = pRefKF->mnId;//参考关键帧的ID
        }


        g2o::Sim3 Srw = vScw[nIDr];//获得位姿
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];//优化后位姿的逆

        cv::Mat P3Dw = pMP->GetWorldPos();//地标点的世界坐标
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);//格式转换
        //利用初始（仅利用SIM）矫正的位姿将地标点投影到相机坐标系下，然后又利用优化后的位姿信息将其重投影到世界坐标系中
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);//格式转换
        pMP->SetWorldPos(cvCorrectedP3Dw);//更新地标点位姿信息

        pMP->UpdateNormalAndDepth();//更新地标点信息
    }
}

//优化相似变换矩阵，th2为10, bFixScale为false，pKF1为当前帧，pKF2为回环帧
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();
//////////////////////////////////////////////////////////////////////////////////////////////添加SIM相似变换矩阵对应的顶点
    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();//pKF1中特征点的数量
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;
/////////////////////////////////////////////////////////////////////////////添加两帧地标点对应的顶点
    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);
    //th2=10
    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];//匹配点对，中pKF1中的地标点
        MapPoint* pMP2 = vpMatches1[i];//匹配点对，中pKF2中的地标点

        const int id1 = 2*i+1;//1,3,5,7
        const int id2 = 2*(i+1);//2,4,6,8,10

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);//匹配点在pKF2中的序号

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w;//匹配点在pKF1的相机坐标系中
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);//赋值id，奇数
                vPoint1->setFixed(true);//设置为固定
                optimizer.addVertex(vPoint1);//添加顶点

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;//匹配点在pKF2的相机坐标系中
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);//赋值id，大于0的偶数
                vPoint2->setFixed(true);//设置为固定
                optimizer.addVertex(vPoint2);//添加顶点
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////边
        // Set edge x1 = S12*X2，设置边
        Eigen::Matrix<double,2,1> obs1;//观测
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];//提取pKF1中第i个特征点
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));//添加pKF2中地标点到边中
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));//添加SIM矩阵到边中
        e12->setMeasurement(obs1);//添加观测
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];//观测方差的逆
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);//设置信息矩阵

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;//核函数
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);//sqrt（10)
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];//提取pKF2中第i2个特征点
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));//添加pKF1中地标点到边中
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));//添加SIM矩阵到边中
        e21->setMeasurement(obs2);//添加观测
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);//信息矩阵

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;//设置核函数
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);//保存2投影到1上的边
        vpEdges21.push_back(e21);//保存1投影到2上的边
        vnIndexEdge.push_back(i);//保存特征点对在pKF1中的序号
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);//优化
/////////////////////////////////////////////////////////////////////////////////////////////////////内点检查和误差边剔除
    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];//遍历2--->1的每一个边
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];//遍历1--->2的每一个边
        if(!e12 || !e21)//有一个边不存在便停止执行
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)//有一个边的误差阈值过大,则
        {
            size_t idx = vnIndexEdge[i];//得到错误点对在第一帧中的序号
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);//将错误点从匹配点对中删除
            optimizer.removeEdge(e12);//移除这两个边
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;//坏边的数量
        }
    }

    int nMoreIterations;
    if(nBad>0)//如果有坏的边
        nMoreIterations=10;
    else
        nMoreIterations=5;
    //nCorrespondences为边的总数
    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers
//////////////////////////////////////////////////////////////////////////////二次优化
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);//再次优化

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        //提取每一边
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)//边的误差过大
        {
            size_t idx = vnIndexEdge[i];//提取边的序号
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);//从容器中删除边
        }
        else
            nIn++;//内点数量加1
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));//获得优化后的SIM相机变换矩阵
    g2oS12= vSim3_recov->estimate();//得到求解参数

    return nIn;
}


} //namespace ORB_SLAM
