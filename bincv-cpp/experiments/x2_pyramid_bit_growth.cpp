#include <opencv2/opencv.hpp>
#include <set>
#include <iostream>
int main(){
    // binary edge image, values {0,255} as SEAL's edge filter produces
    cv::Mat lvl = cv::Mat::zeros(256,256,CV_8UC1);
    cv::randu(lvl,0,2); lvl *= 255;
    for (int L=0; L<4; ++L){
        std::set<int> vals;
        for(int y=0;y<lvl.rows;++y) for(int x=0;x<lvl.cols;++x) vals.insert(lvl.at<uint8_t>(y,x));
        std::cout << "level " << L << ": " << lvl.cols << "x" << lvl.rows
                  << "  distinct values = " << vals.size();
        if (vals.size()<=9){ std::cout << "  {"; for(int v:vals) std::cout<<v<<" "; std::cout<<"}"; }
        std::cout << "  -> " << (int)std::ceil(std::log2(vals.size())) << " bits\n";
        // SEAL BOX_2x2: cv::blur 2x2 then subsample (gaussian disabled)
        cv::Mat blurred; cv::blur(lvl, blurred, cv::Size(2,2));
        cv::Mat next; cv::resize(blurred, next, cv::Size(lvl.cols/2, lvl.rows/2), 0,0, cv::INTER_NEAREST);
        lvl = next;
    }
}
