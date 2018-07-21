#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstdio>
#include <math.h>
#define PI 3.1415926

 using namespace cv; 
 void watershed_myself (Mat source,int number);

 void contours_method(Mat label_record ,Mat aftergradient,int x,int y, int label);

 void flooding(Mat label_record ,Mat source,Mat aftergradient , int x ,int y,int waterlevel,int label,Mat edge);

 void sobel_myself(Mat source);

 void cellsize(Mat label_record, Mat source,int label,struct CELL cell[] );

 void onMouse(int Event,int x,int y,int flags,void* param);

 void onMouse_detail(int Event , int y , int x , int flags , void* userdata);

 void find_compactness(Mat label_record, Mat source,int label,struct CELL cell[] );

 void color_mark(Mat label_record, Mat source,int label,struct CELL cell[] );

 void find_square(Mat label_record, Mat source,int label,struct CELL cell[] );

 void LBP(Mat source);

 int threshold1 = 60;
 int xx=1;

 vector <int> x_collect;
 vector <int> y_collect;

 Mat marker ;
 Mat sample;
 Mat ultimate;
 Mat part_marker1 ;
 Mat edge_sample;

 Point lefttop;
 Point rightdown;
 Point newlefttop;
 Point newrightdown ;
 struct CELL{
	int cell_size;
	int center_x;
	int center_y;
	double compactness;
	double square;
	int mark;
	double radius;
	double square_edge;
 } cell [103];
 int main()
 {
	Mat stainedcell_small = imread("stainedcell_small.JPG", CV_LOAD_IMAGE_GRAYSCALE);
	Mat stainedcell_large = imread("stainedcell_large.JPG", CV_LOAD_IMAGE_GRAYSCALE);

	watershed_myself (stainedcell_small,1);
	setMouseCallback("stainedcell_small",onMouse, &stainedcell_small);
	//watershed_myself (stainedcell_large,2);
	waitKey(0);
	return 0 ;

 }
 void watershed_myself (Mat source,int number )
  {
	Mat temp(source.rows, source.cols, CV_16S);//G = Gx+Gy
	Mat temp_x(source.rows, source.cols, CV_16S);//Gx
	Mat temp_y(source.rows, source.cols, CV_16S);//Gy
	Mat label_record (source.rows,source.cols,CV_16U,Scalar(0));//����marker�ȡA��l���O0
	Mat edge (source.rows,source.cols,CV_16U,Scalar(0));
	float size=0;
	int size_marker =0;
	int label = 0;//�@�}�lmarker�O0
	int temp1 =0;//�p��̤j�Ȯɪ��Ʀr
	Mat aftergradient = source.clone();//�ƻs�榡
	
	//����smooth
	// 1 2 1
	// 2 4 2
	// 1 2 1
	for(int i=1 ; i < source.rows -1 ; i++)
		for(int j=1 ; j < source.cols -1 ; j++)
			aftergradient.at<uchar>(i,j) = (4*aftergradient.at<uchar>(i,j)+aftergradient.at<uchar>(i-1,j-1)+2*aftergradient.at<uchar>(i,j-1)+
											2*aftergradient.at<uchar>(i-1,j)+aftergradient.at<uchar>(i+1,j+1)+2*aftergradient.at<uchar>(i+1,j)+
											2*aftergradient.at<uchar>(i,j+1)+aftergradient.at<uchar>(i-1,j+1)+aftergradient.at<uchar>(i+1,j-1))/16;
	//��Sobel��Gx
	for(int i=1 ; i < source.rows -1 ; i++)
		for(int j=1 ; j < source.cols -1 ; j++)
			temp_x.at<short>(i,j) = abs(source.at<uchar>(i-1,j-1)*(-1) + source.at<uchar>(i-1,j+1)*(-1) + source.at<uchar>(i+1,j-1) + source.at<uchar>(i+1,j+1) + source.at<uchar>(i+1,j)*2 + source.at<uchar>(i-1,j)*(-2));
	//��Sobel��Gy
	for(int i=1 ; i < source.rows -1 ; i++)
		for(int j=1 ; j < source.cols -1 ; j++)	
			temp_y.at<short>(i,j) = abs(source.at<uchar>(i-1,j-1)*(-1) + source.at<uchar>(i+1,j-1)*(-1) + source.at<uchar>(i-1,j+1) + source.at<uchar>(i+1,j+1) + source.at<uchar>(i,j+1)*2 + source.at<uchar>(i,j-1)*(-2));
	//��Gx�̪��̤j��
	for(int i=1 ; i < source.rows -1 ; i++)
		for(int j=1 ; j < source.cols -1 ; j++)
			if(temp_x.at<short>(i,j) > temp1 ) temp1 = temp_x.at<short>(i,j);
	//�NGx��normalize
	for(int i=1 ; i < source.rows -1 ; i++)
		for(int j=1 ; j < source.cols -1 ; j++)
			temp_x.at<short>(i,j) = temp_x.at<short>(i,j) *255.0 / temp1 ;

	temp1 =0;
	//��Gy�̪��̤j��
	for(int i=1 ; i < source.rows -1 ; i++)
		for(int j=1 ; j < source.cols -1 ; j++)
			if(temp_y.at<short>(i,j) > temp1 ) temp1 = temp_y.at<short>(i,j);
	//�NGy��normalize
	for(int i=1 ; i < source.rows -1 ; i++)
		for(int j=1 ; j < source.cols -1 ; j++)
			temp_y.at<short>(i,j) = temp_y.at<short>(i,j) *255.0 / temp1 ;
	//Sobel ���� G=Gx+Gy
	for(int i=1 ; i < source.rows -1 ; i++)
		for(int j=1 ; j < source.cols -1 ; j++)
			aftergradient.at<uchar>(i,j) =sqrt(pow(temp_x.at<short>(i,j),2)+pow(temp_y.at<short>(i,j),2));
	

	
	//�}�l�аOmarker����m�A�qpixel(1,1)�}�l��marker�A���e�Ȭ�30
	for(int i=1 ; i < source.rows-1  ; i++){
		for(int j=1 ; j < source.cols-1  ; j++){
			//����e�C��30��
			if(aftergradient.at<uchar>(i,j) <= threshold1)
			{	//��label�٨S�Q�O��marker�Ȯ�
				if(label_record.at<short>(i,j)==0)
				{
					label++;//marker�аO��
					//printf("label =%d\n",label);
					contours_method(label_record ,aftergradient,i,j,label);//�}�l���j�аOmarker
				}
			}
		}
	}

	marker = label_record.clone();
	
	//�}�l����A�q��誺���e�ȶ}�l
	for(int water = threshold1 +1 ; water< 256 ;water++)
	{
		xx=1;//�o�O��dialtion�ݭn���ܼơA���C��marker�@�I�@�I�C�C�T
		while(xx!=0)
		{
			xx=0;
			for(int i=0 ; i < source.rows  ; i++){
				//�Q��pointer�g�����
				uchar *aftergradient1 = aftergradient.ptr<uchar>(i);
				short *label_record1 = label_record.ptr<short>(i);
				for(int j=0 ; j < source.cols ; j++)
					//index�Omarker �qmarker�}�l�C�C�� �]��marker1�N�O�I�����@����
					for(int index =2 ; index <label+1 ; index++)
						if( label_record1[j] == index && aftergradient1[j] < water )
							flooding(label_record ,source , aftergradient , i ,j,water,index,edge);
			}
		}
	}
	edge_sample = edge.clone();
	cellsize(label_record,source,label,cell);
	find_compactness(label_record,source,label,cell);
	find_square(label_record,source,label,cell);
	sample = source.clone();
	ultimate = source.clone();
	LBP(source);
	//�Ĥ@�ismall��
	if(number == 1){
  	namedWindow("stainedcell_small", WINDOW_NORMAL);
	//setMouseCallback("stainedcell_small",onMouse, &source);
	//setMouseCallback("stainedcell_small",onMouse, &source);
	imshow("stainedcell_small", source);
	}

	//�ĤG�i�j��
	else if(number == 2){
  	namedWindow("stainedcell_large", WINDOW_NORMAL);
	//setMouseCallback("stainedcell_small",onMouse,NULL);
	imshow("stainedcell_large", source);
	}
  }
 void contours_method(Mat label_record ,Mat aftergradient ,int x,int y ,int label)
  {
	 label_record.at<short>(x,y) = label;//�аOmarker��
	 //��ȧC����e=30���ɭ� && x,y�b�ɽu���� &&label�٬O�s���S���Q�аO�L
	 if(aftergradient.at<uchar>(x+1,y) <= threshold1 && x>0 && x+1 < aftergradient.rows  && y > 0 && y<aftergradient.cols && label_record.at<short>(x+1,y)==0) 
		 contours_method(label_record ,aftergradient,x+1,y ,label);
	 
	 if(aftergradient.at<uchar>(x-1,y) <= threshold1 && x-1>0 && x < aftergradient.rows  && y > 0 && y<aftergradient.cols && label_record.at<short>(x-1,y)==0 ) 
		 contours_method(label_record ,aftergradient,x-1,y ,label);
	 
	 if(aftergradient.at<uchar>(x,y+1) <= threshold1 && x>0 && x < aftergradient.rows  && y > 0 && y<aftergradient.cols+1 && label_record.at<short>(x,y+1)==0 ) 
		 contours_method(label_record ,aftergradient,x,y+1 ,label);
	 
	 if(aftergradient.at<uchar>(x,y-1) <= threshold1 && x>0 && x < aftergradient.rows  && y-1 > 0 && y<aftergradient.cols && label_record.at<short>(x,y-1)==0  ) 
		 contours_method(label_record ,aftergradient,x,y-1,label);
	 
	return;
  }
 void flooding(Mat label_record ,Mat source,Mat aftergradient , int x ,int y,int waterlevel,int label,Mat edge)
  {
	  //��x,y�W�X�ɽu�N����
		if( x < 0 || x > aftergradient.rows || y < 0 || y > aftergradient.cols) return;
	  //�ŧipointer
		uchar *aftergradient1 = aftergradient.ptr<uchar>(x);
		short *label_record1 = label_record.ptr<short>(x);
		//��p�����ɡAmarker�]�O�ۦP��
		if(aftergradient1[y] < waterlevel && label_record1[y] == label )
		{
			aftergradient1[y] = waterlevel;//����bgradient�ϤW
			flooding(label_record ,source,aftergradient,x+1,y,waterlevel,label,edge);
			flooding(label_record ,source,aftergradient,x,y+1,waterlevel,label,edge);
			flooding(label_record ,source,aftergradient,x-1,y,waterlevel,label,edge);
			flooding(label_record ,source,aftergradient,x,y-1,waterlevel,label,edge);
			label_record1[y] = label;
			return;
		}
		//��p������ && label�S���O�L�A�]�N�O0  �b�o�ӱ���U�]�O�i�H�T���� �ҥH�n�N�Lmarker�_��
		else if(aftergradient1[y] <= waterlevel && label_record1[y] == 0)
		{
			aftergradient1[y] = waterlevel;//�T��
			label_record1[y] = label;//�аOmarker
			xx+=1;//����xx�ܼơA�@�U1�@�U0���ت��O�Ʊ�o�i�H�@�I�@�I�C�C�T�A���n���j�U�h��ӲT�_��
		}
		//��b����H�U�A�ӥBmarker�����Lmarker��
		else if(aftergradient1[y] <= waterlevel && label_record1[y] != label)
		{
			aftergradient1[y] = 255;//�v��A��=255
			source.at<uchar>(x,y) = 255;//�e�b��ϤW
			edge.at<short>(x,y) = label;
			label_record1[y] = 0 ;//�Nmarker�k�s
		}
		else
		return;
  }
 void cellsize(Mat label_record, Mat source,int label,struct CELL cell[] )
{
	int size =0;
	float heart_x = 0;
	float heart_y = 0;
	for(int x = 2; x < label+1 ; x++){
		for(int i=1 ; i < source.rows -1 ; i++){
			for(int j=1 ; j < source.cols -1 ; j++){
				if(label_record.at<short>(i,j) == x){
					size++;
					x_collect.push_back(i);
					y_collect.push_back(j);
				}
			}
		}
		for(int order=0 ; order < x_collect.size()-1; order++){
				heart_x = heart_x + x_collect[order]; 
				heart_y = heart_y + y_collect[order];
			}
		cell [x-1].center_x = (int)heart_x/x_collect.size();
		cell [x-1].center_y = (int)heart_y/x_collect.size();

		for(int i=1 ; i < source.rows -1 ; i++){
			for(int j=1 ; j < source.cols -1 ; j++){
				if(label_record.at<short>(i,j) == x){
					cell [x-1] .cell_size = size;
					cell [x-1] .mark = x;
					cell [x-1] .radius = (double)sqrt(cell[x-1].cell_size/PI);
					cell [x-1] .square_edge = (double)sqrt(cell[x-1].cell_size);
				}
			}
		}
		x_collect.clear();
		y_collect.clear();
		size = 0;
		heart_x = 0; 
		heart_y = 0;
	}
}
 void find_compactness(Mat label_record, Mat source,int label,struct CELL cell[] )
{
	x_collect.clear();
	y_collect.clear();

	float compactness = 0;
	double distance = 0;
	int temp =0;
	
	for(int x = 2; x < label+1 ; x++){
		for(int i=1 ; i < source.rows -1 ; i++){
			for(int j=1 ; j < source.cols -1 ; j++){
				if(label_record.at<short>(i,j) == x){
				distance = sqrt(pow((cell[x-1].center_x-i),2)+pow(cell[x-1].center_y-j,2));
					if(distance <= cell[x-1].radius){
						temp++;
						distance = 0;
					}
				}
			}
		}

		cell[x-1].compactness = (double)temp/cell[x-1].cell_size;
		//printf("compactness is %f",cell[x-1].compactness);
		distance = 0;
		temp = 0;
	}
	
}
 void sobel_myself(Mat source) {
	 // -1 0 1     -1 -2 -1
	 // -2 0 2      0  0  0
	 // -1 0 1		1  2  1
	Mat temp(source.rows, source.cols, CV_16S);//G = Gx+Gy
	Mat temp_x(source.rows, source.cols, CV_16S);//Gx
	Mat temp_y(source.rows, source.cols, CV_16S);//Gy
	Mat afterSobel,afterSobel_x,afterSobel_y;//��X��G, Gx,Gy��

	//�i��sobel�B�n�B��A�]���B�n�����Y�ҥH���䳣�S���B��
	for(int i=1 ; i < source.rows -1 ; i++){
		for(int j=1 ; j < source.cols -1 ; j++){
			temp_x.at<short>(i,j) = abs(source.at<uchar>(i-1,j-1)*(-1) + source.at<uchar>(i-1,j+1)*(-1) + source.at<uchar>(i+1,j-1) + source.at<uchar>(i+1,j+1) + source.at<uchar>(i+1,j)*2 + source.at<uchar>(i-1,j)*(-2));
			temp_y.at<short>(i,j) = abs(source.at<uchar>(i-1,j-1)*(-1) + source.at<uchar>(i+1,j-1)*(-1) + source.at<uchar>(i-1,j+1) + source.at<uchar>(i+1,j+1) + source.at<uchar>(i,j+1)*2 + source.at<uchar>(i,j-1)*(-2));
			temp.at<short>(i,j) =sqrt(pow(temp_x.at<short>(i,j),2)+pow(temp_y.at<short>(i,j),2));
		}
	}
	
	//�N16S�ର8U
	convertScaleAbs(temp , afterSobel);
	convertScaleAbs(temp_x , afterSobel_x);
	convertScaleAbs(temp_y , afterSobel_y);
	

 }
 void onMouse(int Event , int y , int x , int flags , void* userdata)
{
	Mat * source = static_cast<Mat*>(userdata);
	//(*source).at<uchar>();
	if(Event==CV_EVENT_MOUSEMOVE)
	{
		*source = sample.clone();
		imshow("stainedcell_small", *source);
		system("CLS");
		printf("cursor is at position �G(%d,%d) now\n",x,y);
		if(marker.at<short>(x,y) -1 > 0)
		{	
			printf("----------------------------------------------------------\n");
			printf("cell number # %d\n",cell[marker.at<short>(x,y)-1].mark);
			printf("centroid is (%d,%d)\n",cell[marker.at<short>(x,y)].center_x,cell[marker.at<short>(x,y)-1].center_y);
			printf("size is %d\n",cell[marker.at<short>(x,y)-1].cell_size);
			printf("compactness is %f\n",cell[marker.at<short>(x,y)-1].compactness);
			printf("square-like degree is %f\n",cell[marker.at<short>(x,y)-1].square);

			for(int i=1 ; i < edge_sample.rows-1  ; i++)
				for(int j=1 ; j < edge_sample.cols-1  ; j++)
					if(marker.at<short>(x,y) == edge_sample.at<short>(i,j)) (*source).at<uchar>(i,j) = 0;			
				
			imshow("stainedcell_small", *source);

		}
	}

	FILE *pFile;
	pFile = fopen( "cell_record","a" );
	if(Event == CV_EVENT_RBUTTONDOWN)
	{
		if(marker.at<short>(x,y) -1 > 0)
		{	
			fprintf(pFile,"----------------------------------------------------------\n");
			fprintf(pFile,"cell number # %d\n",cell[marker.at<short>(x,y)-1].mark);
			fprintf(pFile,"centroid is (%d,%d)\n",cell[marker.at<short>(x,y)-1].center_x,cell[marker.at<short>(x,y)].center_y);
			fprintf(pFile,"size is %d\n",cell[marker.at<short>(x,y)-1].cell_size);
			fprintf(pFile,"compactness is %f\n",cell[marker.at<short>(x,y)-1].compactness);
			fprintf(pFile,"square-like degree is %f\n",cell[marker.at<short>(x,y)-1].square);

			fclose(pFile);

			for(int i=1 ; i < edge_sample.rows-1  ; i++)
				for(int j=1 ; j < edge_sample.cols-1  ; j++)
					if(marker.at<short>(x,y) == edge_sample.at<short>(i,j)) sample.at<uchar>(i,j) = 0;			
			
		}
	}

	if(Event == CV_EVENT_LBUTTONDOWN)
	{
		lefttop.x = x;
		lefttop.y = y;

		sample = ultimate.clone();
	}
	if(Event == CV_EVENT_LBUTTONUP)
	{
		int temp;
		rightdown.x = x;
		rightdown.y = y;
		
		if(lefttop.x >rightdown.x)
		{
			temp = rightdown.x;
			rightdown.x = lefttop.x;
			lefttop.x = temp;
		}
		if(lefttop.y >rightdown.y)
		{
			temp = rightdown.y;
			rightdown.y = lefttop.y;
			lefttop.y = temp;
		}
		
		Mat detail(rightdown.x-lefttop.x , rightdown.y-lefttop.y , CV_8U);
		
		newlefttop.x = lefttop.y;
		newlefttop.y = lefttop.x;
		newrightdown.x = rightdown.y;
		newrightdown.y = rightdown.x;

		*source = sample.clone();
		rectangle(sample, newrightdown, newlefttop, Scalar(0,0,0), 1);

		for(int i = 0;i < detail.rows;i++)
			for(int j =0 ;j < detail.cols ; j++)
				detail.at<uchar>(i,j) = sample.at<uchar>(lefttop.x+i , lefttop.y+j);

		imshow("stainedcell_small", *source);

		pyrUp(detail,detail,Size(detail.cols*2, detail.rows*2));
		pyrUp(detail,detail,Size(detail.cols*2, detail.rows*2));

		Mat part_marker  (detail.rows , detail.cols , CV_16U, Scalar(0));

		for(int i=0 ;i < detail.rows ; i++){
			for(int j = 0 ; j < detail.cols ; j++){
				part_marker.at<short>(i,j) = marker.at<short>(lefttop.x+i/4 , lefttop.y+j/4);
			}
		}
			
		part_marker1  = part_marker.clone();

		namedWindow("detail", WINDOW_NORMAL);
		setMouseCallback("detail",onMouse_detail,source);
		imshow("detail", detail);
	}
}
 void onMouse_detail(int Event , int y , int x , int flags , void* userdata)
{
	Mat * source = static_cast<Mat*>(userdata);
	//Mat * part_marker1 = static_cast<Mat*>(userdata);
	if(Event==CV_EVENT_MOUSEMOVE)
	{
		system("CLS");
		printf("cursor is at position �G(%d,%d) now\n",x,y);
		if(part_marker1.at<short>(x,y) -1 > 0)
		{	
			printf("----------------------------------------------------------\n");
			printf("cell number # %d\n",cell[part_marker1.at<short>(x,y)-1].mark);
			printf("centroid is (%d,%d)\n",cell[part_marker1.at<short>(x,y)].center_x,cell[part_marker1.at<short>(x,y)-1].center_y);
			printf("size is %d\n",cell[part_marker1.at<short>(x,y)-1].cell_size);
			printf("compactness is %f\n",cell[part_marker1.at<short>(x,y)-1].compactness);
			printf("square-like degree is %f\n",cell[part_marker1.at<short>(x,y)-1].square);

			for(int i=1 ; i < edge_sample.rows-1  ; i++)
				for(int j=1 ; j < edge_sample.cols-1  ; j++)
					if(part_marker1.at<short>(x,y) == edge_sample.at<short>(i,j)) (*source).at<uchar>(i,j) = 0;	
			imshow("stainedcell_small", *source);
		}
	}
	FILE *pFile;
	pFile = fopen( "cell_record","a" );
	if(Event == CV_EVENT_RBUTTONDOWN)
	{
		if(part_marker1.at<short>(x,y) -1 > 0)
		{	
			fprintf(pFile,"----------------------------------------------------------\n");
			fprintf(pFile,"cell number # %d\n",cell[part_marker1.at<short>(x,y)-1].mark);
			fprintf(pFile,"centroid is (%d,%d)\n",cell[part_marker1.at<short>(x,y)-1].center_x,cell[part_marker1.at<short>(x,y)].center_y);
			fprintf(pFile,"size is %d\n",cell[part_marker1.at<short>(x,y)-1].cell_size);
			fprintf(pFile,"compactness is %f\n",cell[part_marker1.at<short>(x,y)-1].compactness);
			fprintf(pFile,"square-like degree is %f\n",cell[part_marker1.at<short>(x,y)-1].square);

			fclose(pFile);

			for(int i=1 ; i < edge_sample.rows-1  ; i++)
				for(int j=1 ; j < edge_sample.cols-1  ; j++)
					if(part_marker1.at<short>(x,y) == edge_sample.at<short>(i,j)) sample.at<uchar>(i,j) = 0;			
		}
	}

}
 void find_square(Mat label_record, Mat source,int label,struct CELL cell[] )
 {
 
	int temp = 0;

	for(int x = 2; x < label+1 ; x++){
		for(int i=1 ; i < source.rows -1 ; i++){
			for(int j=1 ; j < source.cols -1 ; j++){
				 if(label_record.at<short>(i,j) == x){
					 if(abs(i-cell[x-1].center_x) <= cell[x-1].square_edge/2 && abs(j-cell[x-1].center_y) <= cell[x-1].square_edge/2)temp++;
				 }
			}
		}
		cell[x-1].square = (double)temp/cell[x-1].cell_size;
		temp = 0;
	}
 }
 void LBP(Mat source){
	
	Mat LBP(source.rows, source.cols, CV_8U);
		int pow0 ,pow1,pow2,pow3,pow4,pow5,pow6,pow7=0;	
		for(int i = 1; i < LBP.rows-1 ; i++){
			for(int j = 1 ; j < LBP.cols-1 ; j++){
				if(source.at<uchar>(i-1,j-1) > source.at<uchar>(i,j))pow0 =1;
				else pow0=0;
				
				if(source.at<uchar>(i-1,j) > source.at<uchar>(i,j))pow1 =1;
				else pow1=0;
				
				if(source.at<uchar>(i-1,j+1) > source.at<uchar>(i,j))pow2 =1;
				else pow2=0;
				
				if(source.at<uchar>(i,j+1) > source.at<uchar>(i,j))pow3 =1;
				else pow3=0;
				
				if(source.at<uchar>(i+1,j+1) > source.at<uchar>(i,j))pow4 =1;
				else pow4=0;
				
				if(source.at<uchar>(i+1,j) > source.at<uchar>(i,j))pow5 =1;
				else pow5=0;
				
				if(source.at<uchar>(i+1,j-1) > source.at<uchar>(i,j))pow6 =1;
				else pow6=0;
				
				if(source.at<uchar>(i,j-1) > source.at<uchar>(i,j))pow7 = 1;
				else pow7=0;

				LBP.at<uchar>(i,j) = pow0*pow(2,0)+pow1*pow(2,1)+pow2*pow(2,2)+pow3*pow(2,3)+pow4*pow(2,4)+pow5*pow(2,5)+pow6*pow(2,6)+pow7*pow(2,7);
			
			}
		}
	
		namedWindow("LBP", WINDOW_NORMAL);
		imshow("LBP", LBP);
 }