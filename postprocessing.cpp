#include<opencv2/opencv.hpp>
#include<iostream>
#include<time.h>
using namespace cv;
using namespace std;

void func(Mat src,Mat osrc,int croplinemid , int croplineup)
{       
    cvtColor(src, src, COLOR_RGB2GRAY);
    cvtColor(osrc, osrc, COLOR_RGB2GRAY);
    cvtColor(osrc, osrc,COLOR_GRAY2BGR);
    cv::Mat thresh;
    int img_width = src.cols;
    int img_height = src.rows;
    int oimg_width = osrc.cols;
    int oimg_height = osrc.rows;
    int flex = 10;  
    int th = 5;
    float average_imt = 0;
    int count = 0;
    threshold(src, thresh, th, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresh, contours,hierarchy,RETR_TREE, CHAIN_APPROX_SIMPLE); // Find the contours in the image
    Mat Contours = Mat::zeros(src.size(),CV_8UC3);
    int roi[50]={},rect2[50][4]={},imt[50]={},upper_index[50]={},bottom_index[50]={};
    int bound[4]={0,img_height,0,0};
    vector<Rect> boundRect(1);
    for(int i = 0; i < contours.size();++i){
        if(contourArea(contours[i]) > 150){
            boundRect[0] = boundingRect(contours[i]);
            if (boundRect[0].y < croplinemid &&  boundRect[1].y > bound[0]){
                bound[0] = boundRect[0].y;
                bound[2] = boundRect[0].y+boundRect[0].height;
            }else if (boundRect[0].y + boundRect[0].height > croplinemid && boundRect[0].y+boundRect[0].height < bound[1]){
                bound[1] = boundRect[0].y;
                bound[3] = boundRect[0].y+boundRect[0].height;
            }
        }
    }
    cout << contours[0][0].x << endl;
    cout << contours[0].size() << endl;
    for(int i = 0; i < contours.size();++i){
        if(contourArea(contours[i]) > 180){
            boundRect[0] = boundingRect(contours[i]);
            if (boundRect[0].y < croplinemid && (boundRect[0].y > bound[0] - flex)||(boundRect[0].y+boundRect[0].height > bound[0]-5) && (boundRect[0].y < bound[2])){
                drawContours(Contours, contours, i, Scalar(0,0,255), 2);
                roi[i] = 1;
                upper_index[i]=1;
            }else if(boundRect[0].y + boundRect[0].height > croplinemid && (boundRect[0].y+boundRect[0].height < bound[3] + flex)or(boundRect[0].y<bound[3]+5) && (boundRect[0].y+boundRect[0].height > bound[1])){
                drawContours(Contours, contours, i, Scalar(0,0,255), 2);
                roi[i] = 1;
                bottom_index[i]=1;
            }
        }
    }
    for(int i = 0;i < contours.size();++i){
        if(roi[i] == 1){
            boundRect[0] = boundingRect(contours[i]);
            rect2[i][0] = boundRect[0].x; 
            rect2[i][1] = boundRect[0].y; 
            rect2[i][2] = boundRect[0].width; 
            rect2[i][3] = boundRect[0].height; 
            imt[i] = boundRect[0].y + boundRect[0].height; 
        }
    }
    for(int i = 0;i < contours.size();++i){
        if(roi[i] == 1){
            if(imt[i] > croplinemid){
                average_imt += contourArea(contours[i]);
                count += rect2[i][2];
            }
        }
    }
    average_imt /= count;
    float average_up_imt = 0;
    int count2 = 0;
    for(int i = 0;i < contours.size();++i){
        if(roi[i] == 1){
            if(imt[i] < croplinemid){
                average_up_imt += contourArea(contours[i]);
                count2 += rect2[i][2];
            }
        }
    }
    average_up_imt /= count2;

    float c[img_width][4] = {};
    float averageRatio = 0,medianRatio = 0,minRatio = 1;
    float Ratio[img_width]={};
    float median_Ratio[img_width]={};
    int len_upper = 0,len_bottom = 0;
    for(int i = 0; i < 10; ++i){
        if(upper_index[i] == 1){
            len_upper += 1;
        }
        if(bottom_index[i] == 1){
            len_bottom += 1;
        }
    }
    if(len_upper < 1){
        cout << "No upper boundary was labeled!\n";    //上部沒有標記到，則不算Ratio，只算IMT，Ratio在後面也不會print出來
    }
    else{
        for(int u = 0; u < contours.size();++u){
            if(upper_index[u] == 1){
                boundRect[0] = boundingRect(contours[u]);
                int temp = boundRect[0].y+boundRect[0].height;
                for(int j = boundRect[0].x; j < boundRect[0].x +boundRect[0].width; ++j){
                    for(int i = boundRect[0].y; i < boundRect[0].y +boundRect[0].height; ++i){
                        if(thresh.at<uchar>(i,j)<thresh.at<uchar>(i+1,j)){
                            c[j][0] = i;
                            break;
                        }
                    }
                    for (int i = 0; i < boundRect[0].height; ++i){
                        if(thresh.at<uchar>(temp - i,j)<thresh.at<uchar>(temp-1-i,j)){
                            c[j][1] = temp - i;
                            break;
                        }
                    }
                }
            }
        }
        for(int u = 0; u < contours.size();++u){
            if(bottom_index[u] == 1){
                boundRect[0] = boundingRect(contours[u]);
                int temp = boundRect[0].y+boundRect[0].height;
                for(int j = boundRect[0].x; j < boundRect[0].x +boundRect[0].width; ++j){
                    for(int i = boundRect[0].y; i < boundRect[0].y +boundRect[0].height; ++i){
                        if(thresh.at<uchar>(i,j)<thresh.at<uchar>(i+1,j)){
                            c[j][2] = i;
                            break;
                        }
                    }
                    for (int i = 0; i < boundRect[0].height; ++i){
                        if(thresh.at<uchar>(temp - i,j)<thresh.at<uchar>(temp-1-i,j)){
                            c[j][3] = temp - i;
                            break;
                        }
                    }
                }
            }
        }
        count = 0;
        for(int i = 0; i < img_width;++i){
            if(c[i][0]>0 && c[i][1]>0 && c[i][2]>0 && c[i][3]>0){
                Ratio[i] = (c[i][2] - c[i][1])/(c[i][3] - c[i][0]);
            }
        }
        //sum ratio
        float sumRatio = 0;
        int count_median = 0;
        for(int i = 0; i < img_width;++i){
            if(Ratio[i] > 0){
                median_Ratio[count_median++] = Ratio[i];
                sumRatio += Ratio[i];
            }
        }
        //len ratio
        float lenRatio = 0;
        int count = 0;
        for(int i = 0; i < img_width;++i){
            if(Ratio[i] > 0){
                lenRatio += 1;
            }
        }
        averageRatio = sumRatio/lenRatio;
        //median Ratio
        float temp;
        for(int i=0;i<count_median-1;++i) {
            for(int j=0;j<count_median-i-1;++j) {
            if(median_Ratio[j]>median_Ratio[j+1]){
                temp = median_Ratio[j];
                median_Ratio[j] = median_Ratio[j+1];
                median_Ratio[j+1] = temp;
            }
        }
        medianRatio = median_Ratio[count_median/2];
        }

        //minRatio
        for(int i = 0; i < img_width;++i){
            if(Ratio[i] > 0){
                if(Ratio[i] < minRatio){
                    minRatio = Ratio[i];
                }
            }
        }
    }

    Rect rect(0,croplineup,img_width, img_height);
    // Mat image_cropped = Mat::zeros(src.size(),CV_8UC3);
    Mat image_cropped(osrc, rect);
    addWeighted(image_cropped,0.8,Contours,1.0,0,image_cropped);  //add weught of oringinal image and contours


    /*----------------------------------
                Print part
    Show (括號後面為變數名稱)
    1. IMT (float averageIMT，若只標記到上半部則是用float average_up_imt)
    2. average of LD/IAD (float averageRatio)
    3. min of LD/IAD (float minRatio)
    4. median of LD/IAD (float medianRatio)
    這4個變數是我們希望最後可以秀出來的數字
    ------------------------------------*/
    
    cout << "average imt = " << average_imt << " (pixels)\n";
    cout << "average_up_imt = " << average_up_imt << " (pixels)\n";
    if(averageRatio != 0){      //上部沒標記到則averageRato=0，所以不印
        cout << "averageRatio = " << averageRatio << "\n";
        cout << "minRatio = " << minRatio << "\n";
        cout << "medianRatio = " << medianRatio << "\n";
    }
    /**********疊回原圖*****************/



    /*----------------------------------
            OpenCV Show image part
    output image type of image_cropped => Mat
    label標記出來的結果畫在原圖上後的圖 (Mat image_cropped)
    若沒有要印出來則下兩行可以刪掉
    ------------------------------------*/
    // imshow("addweight",image_cropped); 
    // waitKey(0);
  


    /*********************************
            將文字秀在圖片上
        output image(Mat text_image)
    *********************************/
    Mat text_image = image_cropped.clone();
    //需要廠商的影像比例資料才可確認1個pixel比例是幾mm
    if(len_upper >0 && len_bottom >0){
        cout << len_bottom << endl;
        cout << len_upper << endl;
        string text("Average Ratio = "+std::to_string(averageRatio));
        string text1("Min Ratio     = "+std::to_string(minRatio));
        string text2("Median Ratio     = "+std::to_string(medianRatio));
        string text3("Average IMT   = "+std::to_string(average_imt)+"(pixels)");
        putText(text_image,text,Point(0,img_height-55),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        putText(text_image,text1,Point(0,img_height-40),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        putText(text_image,text2,Point(0,img_height-25),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        putText(text_image,text3,Point(0,img_height-10),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
    }else{   //上部or下部沒標記到IMT正常印其它NaN
        cout << len_upper << endl;
        cout << len_bottom << endl;
        string text("Average Ratio = NaN");
        string text1("Min Ratio     = NaN");
        string text2("Median Ratio     = NaN");
        string text3("Average IMT   = "+std::to_string(average_imt)+"(pixels)");
        string text4("Average IMT   = "+std::to_string(average_up_imt)+"(pixels)");
        string text5("Average IMT   = NaN");
        putText(text_image,text,Point(0,img_height-55),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        putText(text_image,text1,Point(0,img_height-40),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        putText(text_image,text2,Point(0,img_height-25),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        if(len_upper < 1 && len_bottom > 0){
            putText(text_image,text3,Point(0,img_height-10),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        }else if(len_bottom < 1 && len_upper > 0){
            putText(text_image,text4,Point(0,img_height-10),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        }else{
            putText(text_image,text5,Point(0,img_height-10),FONT_HERSHEY_SIMPLEX,0.4,Scalar(0,255,255),1);
        }
    }

    /*----------------------------------
            OpenCV Show image part
    output image type of image_cropped => Mat
    將數據加在image_cropped上的圖 (Mat image_cropped)
    若沒有要印出來則下兩行可以刪掉
    ------------------------------------*/
    imshow("123",text_image); 
    waitKey(0);

    /*
        要附有文字的圖片的話用 text_image這張圖
        反之則是使用 image_cropped
    */
    return;
}

int main(){
    int croplinemid = 125,croplineup = 120; 
    cv::Mat src = imread("capture3.png"); // predict image
    resize(src, src, Size(610, 246), INTER_LINEAR);

    // cv::Mat src = imread("co6.png"); // predict image
    cv::Mat osrc = imread("ca6.png"); // cropped image
    func(src,osrc,croplinemid,croplineup);
    /*
            Function 的四個參數
    1.Mat src: model predict出來接好的圖片
    2.Mat osrc:原始圖片
    3.int croplinemid:前處理裁切中線的相對座標(或者是說前處理產生的topimg的高) 
    4.int croplinemup:前處理裁切上端線的座標 
                                        */
    return 0;
}