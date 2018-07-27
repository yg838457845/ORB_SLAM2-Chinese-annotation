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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

//关键帧数据库的初始化
KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    //mvInvertedFile的定义请看该类头文件的定义，其大小与词袋大小一样
    mvInvertedFile.resize(voc.size());
}


void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        //相关叶节点处添加帧pKF
        mvInvertedFile[vit->first].push_back(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

//探测回环关键帧
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();//提取权重大于15的邻接关键帧
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);
        //遍历词袋向量，mBowVec向量中储存了map类型数据（叶节点序号，权重和）
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];//提取vit->first叶节点对应的所有关键帧序列

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;//遍历这些同一个叶节点下的关键帧
                if(pKFi->mnLoopQuery!=pKF->mnId)//如果没有参与过该关键帧的回环检测
                {
                    pKFi->mnLoopWords=0;//初始化帧pKFi与pKF共享叶节点的数量
                    if(!spConnectedKeyFrames.count(pKFi))//不在待检测pKF帧的临近关键帧中
                    {
                        pKFi->mnLoopQuery=pKF->mnId;//标记pKFi为待检测帧pKF的回环检测帧(存在共享叶节点)
                        lKFsSharingWords.push_back(pKFi);//叶节点共享帧序列中添加该帧pKFi
                    }
                }
                //pKFi与pKF共享叶节点的数量
                pKFi->mnLoopWords++;
            }
        }
    }
    //与当前帧共享叶节点的帧列为空
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)//遍历与当前帧共享叶节点的帧列
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;//maxCommonWords中储存共享叶节点最多的数量
    }

    int minCommonWords = maxCommonWords*0.8f;//阈值

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;//遍历与当前帧共享叶节点的帧列

        if(pKFi->mnLoopWords>minCommonWords)//共享叶节点的数量大于一个阈值
        {
            nscores++;//计数一共有多少满足条件的帧

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);//计算相似度

            pKFi->mLoopScore = si;//该帧的回环得分
            if(si>=minScore)//minScore为通过当前帧与局部地图计算得到的相似度阈值
                lScoreAndMatch.push_back(make_pair(si,pKFi));//储存回环得分和共享叶节点的关键帧
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;//提取满足条件的相似关键帧
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);//提取该相似关键帧的邻接关键帧

        float bestScore = it->first;//该相似关键帧对应的回环得分(相似度)
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;//遍历相似关键帧的邻接关键帧
            //该邻接关键帧既与当前帧共享叶节点,且共享叶节点的数量大于一定阈值
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;//回环得分进行累加
                if(pKF2->mLoopScore>bestScore)
                {//提取最佳的相似关键帧
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }
        //相似关键帧和其邻接帧回环得分的累加--------最佳的相似关键帧
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)//累加得分大于阈值
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());//储存回环关键帧的容器

    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)//选出在25%之前的相似关键帧
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);//将这些相似帧添加到vpLoopCandidates,且不会重复添加
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

//重定位时寻找候选关键帧
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    //共享关键帧集合
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);
        //BowVector表示该帧中特征点描述子对应的向量, 叶节点序号----权重的加和
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            //储存关键帧序列，容器mvInvertedFile中，每一个list表示可以观测到一个叶节点的所有关键帧序列，所以mvInvertedFile的容量大小为词袋库单词的数量大小
            //std::vector<list<KeyFrame*> > mvInvertedFile;
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                 //该叶节点对应的所有关键帧
                KeyFrame* pKFi=*lit;
                //如果该关键帧没有被用于当前帧的重新定位
                if(pKFi->mnRelocQuery!=F->mnId)
                {
                    pKFi->mnRelocWords=0;
                    //关键帧的重定位序号为当前帧序号
                    pKFi->mnRelocQuery=F->mnId;
                    //添加共享叶节点关键帧
                    lKFsSharingWords.push_back(pKFi);
                }
                //关键帧重定位时共享的单词(叶节点)数目
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    //仅考虑与当前帧共享单词达到足够数目的关键帧
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        //maxCommonWords中储存了最大共享叶节点的数量
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;//遍历每一个共享叶节点的地标点

        if(pKFi->mnRelocWords>minCommonWords)
        {
            //提取共享叶节点数量大于阈值的关键帧
            nscores++;
            //计算满足阈值条件下的, 当前帧与共享关键帧间的得分
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            //得分与关键帧组成pair
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        //满足阈值的关键帧(叶节点数量大于阈值)
        KeyFrame* pKFi = it->second;
        //取该关键帧的相邻关键帧
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
        //该关键帧的得分
        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        //遍历共享叶节点关键帧的临近关键帧
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            //如果临近关键帧不是当前帧的共享叶节点关键帧,则直接跳过
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;
            //该共享关键帧得分和临近关键帧得分的累加
            accScore+=pKF2->mRelocScore;
            //提取临近关键帧中最佳得分的关键帧
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        //该共享关键帧的相邻关键帧的得分和--------该临近图中得分最佳的关键帧
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            //最佳得分和
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());//储存重定位候选帧的容器
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        //每一个临近图的得分和
        const float &si = it->first;
        if(si>minScoreToRetain)//位于前25%的重定位关键帧
        {
            //该临近图中最佳的关键帧
            KeyFrame* pKFi = it->second;
            //跳过重复的元素, 如果pKFi之前出现过,则函数返回真,则直接跳过
            if(!spAlreadyAddedKF.count(pKFi))
            {
                //添加共享关键帧附近图中最佳的关键帧,作为候选帧
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
