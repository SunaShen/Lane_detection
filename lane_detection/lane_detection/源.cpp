#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


Mat kern_L = (Mat_<char>(3, 3) << 2, 3, 0,
								  3, 0,-3,
								  0,-3,-2);

Mat kern_R = (Mat_<char>(3, 3) << 0,-3,-2,
								  3, 0,-3,
								  2, 3, 0);

int main()
{
	Mat src_img, tem_img_1, tem_img_2, dst_img_1, dst_img_2;

	VideoCapture capture("../../data_set/IMG_7563.mp4");

	if (!capture.isOpened())
	{
		cout << "Movie open Error" << endl;
		return -1;
	}


	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "帧率为:" << " " << rate << endl;
	cout << "总帧数为:" << " " << capture.get(CV_CAP_PROP_FRAME_COUNT) << endl;//输出帧总数

	//ofstream outfile("data.txt");

	//获取视频帧频

	cout << "按键q结束程序~" << endl;

	while (waitKey(33) != 'q') //按键q退出
	{

		if (!capture.read(src_img))
				break;

		cvtColor(src_img, src_img, CV_BGR2GRAY);

		imshow("src_img_gray", src_img);

		Mat element = getStructuringElement(MORPH_CROSS, Size(5, 5), Point(-1, -1));

		erode(src_img, src_img, element);

		imshow("src_img_erode", src_img);


		filter2D(src_img, tem_img_1, -1, kern_L);
		filter2D(src_img, tem_img_2, -1, kern_R);

		imshow("tem_img_1", tem_img_1);
		imshow("tem_img_2", tem_img_2);

		threshold(tem_img_1, dst_img_1, 230, 255, THRESH_TOZERO);
		threshold(tem_img_2, dst_img_2, 230, 255, THRESH_TOZERO);

		imshow("dst_img_1", dst_img_1);
		imshow("dst_img_2", dst_img_2);

		Mat res_img = dst_img_1 + dst_img_2;

		line(res_img, Point(0, res_img.rows / 2), Point(res_img.cols, res_img.rows / 2), Scalar(255, 255, 255));

		line(res_img, Point(0, res_img.rows - 10), Point(res_img.cols, res_img.rows - 10), Scalar(255, 255, 255));

		imshow("res_img", res_img);


		Mat det_img = res_img(Range(res_img.rows / 2, res_img.rows - 10), Range(0, res_img.cols));

		//cout << det_img.size() << endl;
		//cout << det_img.channels() << endl;
		imshow("det_img", det_img);

		//outfile << det_img << endl << endl;
	}
}
