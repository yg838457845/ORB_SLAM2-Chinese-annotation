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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())//检查队列中是否有关键帧
        {
            // Detect loop candidates and check covisibility consistency
            //检查是否出现回环
            if(DetectLoop())
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
                //计算相似变换矩阵
               if(ComputeSim3())
               {
                   // Perform loop fusion and pose graph optimization
                   //回环融合和位姿图优化
                   CorrectLoop();
               }
            }
        }       
        //如果需要则重置系统
        ResetIfRequested();

        if(CheckFinish())//检查是否完成整个过程
            break;
         //休眠
        usleep(5000);
    }
    //设置完结状态
    SetFinish();
}

//向回环检测线程中插入关键帧
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);//上锁
    if(pKF->mnId!=0)//如果不是第一帧
        mlpLoopKeyFrameQueue.push_back(pKF);//加入到回环检测序列中
}

//检查回环帧检测队列中是否有数据
bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

//回环检测函数
bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();//从队列开头提取需要被检测的帧
        mlpLoopKeyFrameQueue.pop_front();//从队列中删除需要检测的该帧
        // Avoid that a keyframe can be erased while it is being process by this thread
        //在检测该帧是否满足回环条件的时候，设置该帧为不可被删除的
        mpCurrentKF->SetNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    //如果该帧距离上次回环检测仅相差不到10个关键帧，mLastLoopKFid初值为0
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        //关键帧数据库中加入了该关键帧
        mpKeyFrameDB->add(mpCurrentKF);
        //设置其为可删除的
        mpCurrentKF->SetErase();
        //返回false，即没有检测到回环
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();//提取权重大于15的邻接关键帧
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;//计算待检测关键帧的词袋向量，BowVector表示（叶节点序号----权重的加和）
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];//遍历所有的邻接关键帧
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;//得到邻接关键帧的词袋向量

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);//计算待检测关键帧与邻接关键帧的相似度

        if(score<minScore)
            minScore = score;//求取最小的相似度
    }

    // Query the database imposing the minimum score
    //检测关键帧数据库中的回环关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())//如果没有回环帧
    {
        mpKeyFrameDB->add(mpCurrentKF);//往数据库中添加该关键帧
        mvConsistentGroups.clear();//清除
        mpCurrentKF->SetErase();//设置可以清除
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it//连续的回环帧
    //下面代码的大概意思是：当一个回环成立的时候，需要连续几帧都被检测出来有回环且它们的回环帧之间有交叉，重叠达到一定的阈值
    mvpEnoughConsistentCandidates.clear();

    //typedef pair<set<KeyFrame*>,int> ConsistentGroup;
    //vCurrentConsistentGroups中有关键帧和连续的次数
    vector<ConsistentGroup> vCurrentConsistentGroups;

    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];//遍历当前帧的所有回环候选帧

        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();//回环帧的所有邻接帧
        spCandidateGroup.insert(pCandidateKF);//往邻接帧中加入回环帧

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        //mvConsistentGroups的每一个元素都是类型pair(set<KeyFrame*>,int)
        //mvConsistentGroups与vCurrentConsistentGroups相同
        //mvConsistentGroups中储存的是（之前待检测帧的回环帧及其邻接帧——连续次数），且如果当前帧没有被检测出回环效应，则被置空
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;//关键帧序列(以前出现过的回环帧及其邻接关键帧)

            bool bConsistent = false;
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)//遍历当前帧的一个回环帧局部地图
            {
                if(sPreviousGroup.count(*sit))//如果以前待检测帧的回环帧局部图中出现了当前待检测帧的回环帧局部地图
                {
                    bConsistent=true;//连续为真
                    bConsistentForSomeGroup=true;//连续为真
                    break;
                }
            }

            if(bConsistent)//当前回环帧信息在之前出现过
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;//iG为之前出现过的回环帧的序号
                int nCurrentConsistency = nPreviousConsistency + 1;//计算这一回环帧的连续次数加一
                if(!vbConsistentGroup[iG])//每一个以前的回环帧都对应一个逻辑量,初始值为false(目的是对应的一个iG只能被执行一次，即一个以前的回环帧只能和当前的回环帧匹配一次)
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);//spCandidateGroup为(当前回环帧及其邻接帧, 连续的次数)
                    vCurrentConsistentGroups.push_back(cg);//添加进vCurrentConsistentGroups
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once, 回环帧对应的值为真
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)//连续的次数大于阈值
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);//添加可用回环帧
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once,一个候选帧只能插入一次
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)//连续的次数不够, 一般第一个待检测帧会有这种情况,同时初始化vCurrentConsistentGroups
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups 更新(可视一致)组
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);//当前帧添加到数据库
    //mvpEnoughConsistentCandidates为可用回环帧
    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

//计算相似矩阵
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
    //计算相似变换矩阵
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();//提炼的候选关键帧

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    //容器的大小和候选关键帧数量相同
    vpSim3Solvers.resize(nInitialCandidates);
    //双层容器
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];//提取每一个候选关键帧

        // avoid that local mapping erase it while it is being processed in this thread
        //设置该关键帧不能被删除
        pKF->SetNotErase();
        //如果该关键帧是坏的
        if(pKF->isBad())
        {
            vbDiscarded[i] = true;//认为是坏的
            continue;
        }
        //vvpMapPointMatches[i]为当前关键帧mpCurrentKF与第i个候选关键帧匹配到的地标点，其维度与mpCurrentKF的特征点数量相同
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);
        //匹配点对数量过少
        if(nmatches<20)
        {
            vbDiscarded[i] = true;//认为第i个候选关键帧是坏的
            continue;
        }
        else
        {
            //mSensor!=MONOCULAR, 不是单目为真, 单目为假
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);//设置ransac参数
            vpSim3Solvers[i] = pSolver;//vpSim3Solvers,每一个候选回环关键帧都对应一个sim求解器
        }

        nCandidates++;//符合要求的回环关键帧
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    //RANSAC迭代法
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])//候选回环关键帧是错的
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];//提取每一个回环关键帧

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;//内点准则
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];//每一个回环关键帧的SIM求解器
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);//返回从回环关键帧到当前关键帧的相似转换矩阵

            // If Ransac reachs max. iterations discard keyframe
            //如果迭代的次数已经超出了限制，bNoMore=true，表明该回环帧为错误的，需要删除，不进行相似度计算
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;//回环帧数量减一
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())//得到相似变换矩阵SCM后
            {
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));//第i个回环帧匹配的点对
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)//内点容器vbInliers的维数等于当前帧特征点的数量,与vvpMapPointMatches[i]的维数相同
                {
                    if(vbInliers[j])//第j个匹配点对满足SCM矩阵的内点要求
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];//赋值是内点的匹配点对,即当前帧第j个特征点对应在回环帧i上的地标点
                }

                cv::Mat R = pSolver->GetEstimatedRotation();//得到旋转
                cv::Mat t = pSolver->GetEstimatedTranslation();//得到平移
                const float s = pSolver->GetEstimatedScale();//得到估计尺度
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);//利用相似变换匹配之前没有匹配到的特征点对，且将两次匹配得到的点对都储存在容器vpMapPointMatches中

                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);//赋值原始的相似变换矩阵
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);//优化相似变换矩阵, 优化结果也在gScm

                // If optimization is succesful stop ransacs and continue
                //优化成功，内点数量大于20, 停止ransac
                if(nInliers>=20)
                {
                    bMatch = true;//匹配成功
                    mpMatchedKF = pKF;//该回环关键帧设置为相似回环参考帧
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);//得到相似回环帧的位姿
                    mg2oScw = gScm*gSmw;//得到当前帧的位姿（相似回环帧的位姿*相似变换矩阵）
                    mScw = Converter::toCvMat(mg2oScw);//格式转换

                    mvpCurrentMatchedPoints = vpMapPointMatches;//内点的匹配点对
                    break;
                }
            }
        }
    }

    if(!bMatch)//如果RANSAC匹配成功
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();//回环帧可以被删除
        mpCurrentKF->SetErase();
        return false;//返回错的
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    //利用局部地图信息再进行一次更新
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();//提取相似回环帧的邻接关键帧(大于15)
    vpLoopConnectedKFs.push_back(mpMatchedKF);//相似回环帧及其邻接帧的集合
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;//遍历相似回环帧及其邻接关键帧
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();//提取其地标点
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];//遍历局部回环帧地图的每一个地标点
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)//判断该地标点是否在回环的局部地图中（只能执行一次）
                {
                    mvpLoopMapPoints.push_back(pMP);//回环局部地图的地标点（不会重复）
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    //利用更新后的mScw和回环帧局部地图中的地标点(更多)  找到更多的匹配点, 将其和之前已经得到的内点 一起储存在mvpCurrentMatchedPoints中
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;//当前帧和相似回环帧局部地图间所有的匹配点对
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)//相似回环帧及其邻接帧与当前帧的匹配点对大于40
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();//将不是相似回环帧的关键帧设置为可删除
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();//全部回环候选帧都设置为可删除
        mpCurrentKF->SetErase();
        return false;
    }

}

//回环校正
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    //停止局部建图（休眠），不会进行局部优化，也不会删除地图中关键帧
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    //mbRunningGBA初始化为false
    if(isRunningGBA())//如果在运行全局优化
    {
        unique_lock<mutex> lock(mMutexGBA);//上锁
        mbStopGBA = true;//停止全局优化的参数

        mnFullBAIdx++;//全局优化的次数加一

        if(mpThreadGBA)//如果存在全局优化线程
        {
            mpThreadGBA->detach();//分散线程
            delete mpThreadGBA;//删除线程
        }
    }

    // Wait until Local Mapping has effectively stopped
    //如果局部建图线程没有完全停止（例如tracking创建关键帧的过程中，局部地图线程不能停止）
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    //更新当前帧的位姿图（因为之前求出了回环帧和相似变换矩阵）
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    //得到当前关键帧的邻接共享关键帧(大于15)
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    //当前帧以其共享关键帧组成的局部地图
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);
    //说白了KeyFrameAndPose为类型map<KeyFrame*,g2o::Sim3>,
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    //当前关键帧对应 当前位姿mg2oScw(由回环帧位姿与相似变换矩阵计算得到)
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    //从当前帧到世界坐标系的位姿转换关系
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);//上锁

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;//遍历当前帧以其共享关键帧组成的局部地图

            cv::Mat Tiw = pKFi->GetPose();//得到位姿

            if(pKFi!=mpCurrentKF)//是共享关键帧而不是当前关键帧
            {
                cv::Mat Tic = Tiw*Twc;//得到当前帧到共享关键帧的位姿转换关系
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);//提取旋转矩阵
                cv::Mat tic = Tic.rowRange(0,3).col(3);//提取平移向量
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);//得到当前帧到共享帧的相似变换矩阵(尺度因子为1)
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;//世界坐标系到共享关键帧
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;//构建map型:map<pKFi, g2oCorrectedSiw>
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);//提取世界坐标系到共享关键帧的旋转矩阵
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);//提取世界坐标系到共享关键帧的平移向量
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);//原始的世界坐标系到共享关键帧
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        //////////////////更新当前帧和其共享关键帧的位姿,地标点信息
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;//提取共享关键帧
            g2o::Sim3 g2oCorrectedSiw = mit->second;//对应的相似变换矩阵
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();//矩阵的逆

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];//提取共享关键帧原始的相似变换矩阵

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();//得到共享关键帧的地标点
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];//遍历共享关键帧的每一个地标点
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)//一个地标点只能执行下面过程一次
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();//地标点三维世界坐标
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                //利用原始相似变换g2oSiw将地标点投影到共享关键帧相机坐标系中,再次利用回环矫正后相似变换矩阵g2oCorrectedSwi反向投影到世界坐标系中
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);//重置每一个地标点的位姿信息
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;//每个地标点只能执行一次
                pMPi->mnCorrectedReference = pKFi->mnId;//标记地标点被哪一个共享关键帧利用相似变换矩阵进行了优化
                pMPi->UpdateNormalAndDepth();//更新该地标点的深度和方向
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            //更新后的共享关键帧的位姿信息
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();
            //利用尺度信息对平移进行修正
            eigt *=(1./s); //[R t/s;0 1]
            //得到修正后共享关键帧的位姿矩阵
            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);
            //重新设置共享关键帧的位姿
            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            //更新连接关系
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        //开始回环融合
        //mvpCurrentMatchedPoints为当前帧所有的匹配点,其维度与当前帧特征点的个数相同
        ///////////融合当前帧地标点与相应的回环帧局部地图地标点
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];//提取每一个匹配到的地标点(第i个特征点在回环帧的局部地图中对应的地标点)
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);//再得到当前帧第i个特征点对应的地标点
                if(pCurMP)//如果当前帧该位置已经储存了地标点,则直接替换
                    pCurMP->Replace(pLoopMP);
                else//如果还没有地标点
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);//则直接将该地标点加入该关键帧
                    pLoopMP->AddObservation(mpCurrentKF,i);//地标点中添加当前帧的观测
                    pLoopMP->ComputeDistinctiveDescriptors();//更新该地标点的描述子
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //////////////////融合当前帧局部地图地标点与回环帧局部地图地标点的信息
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;//遍历当前帧以及共享关键帧
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();//得到每一个共享关键帧的邻接帧(大于15)（只是当前帧附近的局部地图）

        // Update connections. Detect new links.
        pKFi->UpdateConnections();//更新连接关系，注意就在此时添加了回环边
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();//得到共享关键帧的全部邻接帧（包括了回环帧，以及当前帧附近的局部地图）
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);//删去当前帧局部地图中权重大于15的邻接关键帧
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);//删去当前帧及其邻接关键帧
        }
    }

    // Optimize graph
    //利用共享关键帧的邻接关键帧(权值小于15)进行优化, mpMatchedKF相似回环帧, mpCurrentKF为当前帧, NonCorrectedSim3储存当前帧局部地图以及它们的原始位姿信息
    //CorrectedSim3储存当前帧局部地图以及它们矫正后的位姿信息, LoopConnections中储存了共享关键帧的邻接关键帧(权值小于15)和回环帧局部地图中的关键帧, mbFixScale单目为false
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();//地图大型修正次数加一

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);//添加回环边
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    //执行全局优化
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    //全局优化线程
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    //释放局部建图线程
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId;  //最近一次回环的id
}

//将当前帧及其共享帧对应的地标点和回环帧以及共享帧对应的地标点进行融合
//CorrectedPosesMap中储存的是(当前帧的共享关键帧-------矫正后的位姿)
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;//提取当前帧的共享关键帧

        g2o::Sim3 g2oScw = mit->second;//矫正后的位姿信息
        cv::Mat cvScw = Converter::toCvMat(g2oScw);
        //mvpLoopMapPoints表示相似回环帧的局部地图中的地标点
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        //将其与当前帧以及共享关键帧中的地标点进行融合,结果储存在vpReplacePoints中,vpReplacePoints中储存了回环帧局部地图点对应的当前共享帧地标点的信息
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];//提取第i个回环帧局部地图点对应的当前共享帧地标点信息
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);//将当前共享帧地标点更换变为回环帧局部地图点
            }
        }
    }
}


void LoopClosing::RequestReset()
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
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

//全局优化
void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    //mbStopGBA = false; nLoopKF = mpCurrentKF->mnId（也是回环的序号）；
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)//如果不停止BA
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();//要求局部地图停止
            // Wait until Local Mapping has effectively stopped
            //如果局部地图没有停止且没有完成
            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex，上锁
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());//最开始只有第一个初始关键帧

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();//初始关键帧
                const set<KeyFrame*> sChilds = pKF->GetChilds();//初始关键帧的孩子节点
                cv::Mat Twc = pKF->GetPoseInverse();//初始帧的位姿
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;//遍历孩子节点
                    if(pChild->mnBAGlobalForKF!=nLoopKF)//该孩子未经矫正过
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;//计算初始关键帧到孩子节点的位姿转换
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;，计算世界坐标到孩子节点的位姿转换，pKF->mTcwGBA为世界坐标系到初始关键帧的位姿转换（BA矫正后）
                        pChild->mnBAGlobalForKF=nLoopKF;//孩子节点被标记（已经矫正过）

                    }
                    lpKFtoCheck.push_back(pChild);//往初始关键帧的容器中加入了新的孩子节点
                }

                pKF->mTcwBefGBA = pKF->GetPose();//储存未经矫正过的位姿
                pKF->SetPose(pKF->mTcwGBA);//赋值新的位姿
                lpKFtoCheck.pop_front();//删去已经处理过的关键帧节点
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();//得到地图中所有的地标点

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];//遍历每一个地标点

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)//已经BA优化后
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);//替换世界位姿
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();//得到地标点的参考关键帧

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)//该参考关键帧没有被矫正过
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);//未经矫正的参考关键帧的旋转
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);//未经矫正的参考关键帧的平移
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;//将地标点投影到参考关键帧上

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();//参考关键帧的逆（矫正后）
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);//矫正后的旋转矩阵
                    cv::Mat twc = Twc.rowRange(0,3).col(3);//矫正后的平移向量

                    pMP->SetWorldPos(Rwc*Xc+twc);//重投影（矫正后）后的地标点位姿
                }
            }            
            //通知地图一次大更新
            mpMap->InformNewBigChange();
            //释放局部建图线程
            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;//完成BA优化为真
        mbRunningGBA = false;//正在跑BA为假
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
