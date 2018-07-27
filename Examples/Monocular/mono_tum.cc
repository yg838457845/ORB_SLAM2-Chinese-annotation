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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images,定义图像的路径，集合进容器中
    vector<string> vstrImageFilenames;
    //定义时间戳容器
    vector<double> vTimestamps;
    //包含时间戳的图像文件名txt文件
    string strFile = string(argv[3])+"/rgb.txt";
    //将txt中的时间戳和图像文件名分别储存到容器vTimestamps，vstrImageFilenames中
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    //定义容器的大小
    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    //进入SLAM系统的接口中
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics，定义跟踪的容器
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file依次读取每一帧图像
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        //提取这一帧图像对应的时间戳
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        //如果编译器可以编译c+11,那就获取当前时间
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system，将图片以及对应的时间戳放入跟踪过程中
        SLAM.TrackMonocular(im,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
        //计算跟踪过程的耗时
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        //该帧跟踪花费的时间保存进容器中
        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        //计算下一帧图像时间戳与当前时间戳的差值T，如果是最后一帧的话计算其与之前一帧的时间戳差值。
        //如果追踪耗时小于时间戳的差值（与相机的FPS有关），则休眠一会，等待下一帧图像的到来
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    //追踪完所有的图片序列后，关掉当前线程
    SLAM.Shutdown();

    // Tracking time statistics
    //计算跟踪过程中数据的统计特性，对跟踪耗时的容器进行排序
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    //将容器中的时间加和
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;

    //计算中位数
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    //计算平均值
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

//将文件中的图像名和时间戳读取出来
void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    //定义输入文件流
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines，跳过前三行
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        //读取正式数据的第一行
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            //从s到ss的string流
            ss << s;
            double t;
            string sRGB;
            //从ss分别到doube（t）和sRGB的输入流，t为时间戳，sRGB为文件名。
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
