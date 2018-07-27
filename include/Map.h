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

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;

class Map
{
public:
    //默认构造函数
    Map();

    //添加关键帧
    void AddKeyFrame(KeyFrame* pKF);
    //添加点云
    void AddMapPoint(MapPoint* pMP);
    //删去点云
    void EraseMapPoint(MapPoint* pMP);
    //删去关键帧
    void EraseKeyFrame(KeyFrame* pKF);

    //设置参考点云集合
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

    //通知地图有大的变化（如重定位或回环融合造成地图大的变化）——博客上未出现的函数
    void InformNewBigChange();

    //得到地图大变化的指标——博客上未出现的函数
    int GetLastBigChangeIdx();

    //获得地图中所有关键帧的容器
    std::vector<KeyFrame*> GetAllKeyFrames();
    //获得地图中所有地标点的容器
    std::vector<MapPoint*> GetAllMapPoints();
    //获得地图中参考地标点的容器
    std::vector<MapPoint*> GetReferenceMapPoints();

    //地图中地标点的数量
    long unsigned int MapPointsInMap();
    //地图中关键帧的数量
    long unsigned  KeyFramesInMap();

    //不知道是什么，应该是得到最大keyFrame的序号
    long unsigned int GetMaxKFid();

    void clear();

    //不知道，字面意思是保存mvp起源关键帧的容器
    vector<KeyFrame*> mvpKeyFrameOrigins;

    //地图更新的线程互斥锁
    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    //点云创建线程的互斥锁
    std::mutex mMutexPointCreation;

protected:
    //地标点的集合
    std::set<MapPoint*> mspMapPoints;
    //关键帧的集合
    std::set<KeyFrame*> mspKeyFrames;
    //参考地标点的集合容器
    std::vector<MapPoint*> mvpReferenceMapPoints;
    //最大的关键帧id，其实不知道是什么
    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    //每次地图重大变化的序号，博客上面没有
    int mnBigChangeIdx;

    //地图的互斥锁
    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
