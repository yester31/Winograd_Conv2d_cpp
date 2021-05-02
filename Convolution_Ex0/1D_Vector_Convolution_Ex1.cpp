#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//이미지 데이터 및 이미지 이름 가져오기
vector<pair<Mat, string>> TraverseFilesUsingDFS(const string& folder_path)
{
	_finddata_t file_info;
	string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);
	vector<pair<Mat, string>> imgBox;

	//If folder_path exsist, using any_file_pattern will find at least two files "." and "..",
	//of which "." means current dir and ".." means parent dir
	if (handle == -1)
	{
		cerr << "folder path not exist: " << folder_path << endl;
		exit(-1);
	}

	//iteratively check each file or sub_directory in current folder
	do
	{
		string file_name = file_info.name; //from char array to string

		//check whtether it is a sub direcotry or a file
		if (file_info.attrib & _A_SUBDIR)
		{
			if (file_name != "." && file_name != "..")
			{
				string sub_folder_path = folder_path + "\\" + file_name;
				TraverseFilesUsingDFS(sub_folder_path);
				cout << "a sub_folder path: " << sub_folder_path << endl;
			}
		}
		else  //cout << "file name: " << file_name << endl;
		{
			int npo1 = file_name.find('_') + 1;
			int npo2 = file_name.find('.');
			int npo3 = npo2 - npo1;
			string newname = file_name.substr(npo1, npo3);
			string sub_folder_path2 = folder_path + "\\" + file_name;
			Mat img = imread(sub_folder_path2);
			imgBox.push_back({ { img }, { newname } });
		}
	} while (_findnext(handle, &file_info) == 0);

	//
	_findclose(handle);
	return imgBox;
}

float activationTanh(float x) {
	return tanh(x);
}

float activationSigmoid(float x) {
	return (1.f / (exp(-x) + 1.f));
}

float activationReLU(float x) {
	return (x > 0.f ? x : 0.f);
}

void valueCheck(vector<float> &valueCheckInput, int input_n, int input_c, int input_h, int input_w, int offset = 0) {
	if (offset == 1) { input_n = 1; }

	int temp1 = input_w * input_h * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
				   int g_idx = w_idx + temp4;
					cout << setw(5) << valueCheckInput[g_idx] << " ";
				}cout << endl;
			}cout << endl; cout << endl;
		}
	}
}

void valueCheck(vector<float> &valueCheckInput, int input_n, int input_c, int offset = 0) {
	if (offset == 1) {input_n = 1;}

	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int	temp2 = ⁠n_idx * input_c;

		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int ⁠g_idx = ⁠c_idx  + temp2;
			cout << setw(5) << valueCheckInput[⁠g_idx] << " ";
		}cout << endl;
	}cout << endl; cout << endl;
}

void zeroPadding(vector<float> &zeroPaddingOutput, vector<float> &zeroPaddingInput, int input_n, int input_c, int input_h, int input_w, int topPadingSize, int bottomPadingSize, int leftPadingSize, int rightPadingSize) {
	//cout << "===== Zero Padding ===== \n";
	//zeroPaddingOutput.resize(input_n * input_c*(input_h + topPadingSize + bottomPadingSize)*(input_w + leftPadingSize + rightPadingSize));
	int temp1 = input_w * input_h * input_c;
	int temp1o = (input_h + topPadingSize + bottomPadingSize)*(input_w + leftPadingSize + rightPadingSize)* input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		int temp2o = ⁠n_idx * temp1o;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			int temp3o = ⁠c_idx * (input_w + leftPadingSize + rightPadingSize) * (input_h + topPadingSize + bottomPadingSize) + temp2o;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				int temp4o = (⁠h_idx + topPadingSize)*(input_w + leftPadingSize + rightPadingSize) + leftPadingSize + temp3o;

				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int ⁠g_idx = w_idx + temp4;
					int g_idx_Output = w_idx + temp4o;
					zeroPaddingOutput[g_idx_Output] = zeroPaddingInput[⁠g_idx];
				}
			}
		}
	}
}

void activation(vector<float> &activationOutput, vector<float> &activationInput) {
	cout << "===== activation ===== \n";

	activationOutput.resize(activationInput.size());
	for (int i = 0; i < activationInput.size(); i++)
	{
		//activationInput[i] = activationSigmoid(activationInput[i]);
		activationOutput[i] = activationTanh(activationInput[i]);
		//activationInput[i] = activationReLU(activationInput[i]);
	}
}

void convolution(vector<float> &convOutput, vector<float> &convInput, vector<float> &kernel, int kernelSize, int stride, int input_n, int input_c, int input_h, int input_w, int ouput_c) {
	int outputHeightSize = ((input_h - kernelSize) / stride) + 1;
	int outputWidthSize = ((input_w - kernelSize) / stride) + 1;
	//Conv_output.resize(input_n * Ouput_C * outputHeightSize * outputHeightSize);
	//cout << "===== Convolution ===== \n";

	int temp1i = input_h * input_w *input_c;
	int temp1o = outputHeightSize * outputWidthSize * ouput_c;
	int temp1k = kernelSize * kernelSize * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2i = ⁠n_idx * temp1i;
		int temp2o = ⁠n_idx * temp1o;
		for (int k_idx = 0; k_idx < ouput_c; k_idx++)
		{
			int temp2k = k_idx * temp1k;
			int temp3o = k_idx * outputHeightSize * outputWidthSize + temp2o;
			for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
			{
				int temp3i = ⁠c_idx * input_w * input_h + temp2i;
				int temp3k = ⁠c_idx * kernelSize * kernelSize + temp2k;
				for (int rowStride = 0; rowStride < outputHeightSize; rowStride++) {
					int temp4o = rowStride * outputWidthSize + temp3o;
					for (int colStride = 0; colStride < outputWidthSize; colStride++) {
						float sum = 0;
						int g_idx_o = colStride + temp4o;
						for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
							int temp4i = x * input_w + temp3i;
							int temp4k = (x - rowStride * stride) * kernelSize + temp3k;
							for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
								int ⁠g_idx_i = y + temp4i;
								int g_idx_k = (y - colStride * stride) + temp4k;
								sum += convInput[⁠g_idx_i] * kernel[g_idx_k];
							}
						}
						convOutput[g_idx_o] += sum;
					}
				}
			}
		}
	}
}

void avgPooling(vector<float> &avgPoolingOutput, vector<float> &avgPoolingInput, int input_n, int input_c, int input_h, int input_w, int poolingWindow, int poolingStride, int poolingOutputHeight, int poolingOutputWidth) {
	//cout << "===== AvgPooling ===== \n";	
	//avgPoolingOutput.resize(input_n*input_c*poolingOutputHeight*poolingOutputWidth);
	float poolingWindowAreaInverse = 1.f / (poolingWindow * poolingWindow);
	int temp1i = input_h * input_w *input_c;
	int temp1o = poolingOutputHeight * poolingOutputWidth * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2i = ⁠n_idx * temp1i;
		int temp2o = ⁠n_idx * temp1o;
			for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
			{
				int temp3i = ⁠c_idx * input_w * input_h + temp2i;
				int temp3o = ⁠c_idx * poolingOutputHeight * poolingOutputWidth + temp2o;
			for (int rowStride = 0; rowStride < poolingOutputHeight; rowStride++) {
				int temp4o = rowStride * poolingOutputWidth + temp3o;
				for (int colStride = 0; colStride < poolingOutputWidth; colStride++) {
					float sum = 0;
					int g_idx_o = colStride + temp4o;
					for (int x = rowStride * poolingStride; x < rowStride * poolingStride + poolingWindow; x++) {
						int temp4i = x * input_w + temp3i;
						for (int y = colStride * poolingStride; y < colStride * poolingStride + poolingWindow; y++) {
							int ⁠g_idx_i = y + temp4i;
							sum += avgPoolingOutput[⁠g_idx_i];
						}
					}
					avgPoolingInput[g_idx_o] = sum * poolingWindowAreaInverse;
				}
			}
		}
	}
}

void maxPooling(vector<float> &maxPoolingOutput, vector<float> &maxPoolingInput, int input_n, int input_c, int input_h, int input_w, int poolingWindow, int poolingStride, int poolingOutputHeight, int poolingOutputWidth) {
	// cout << "===== maxPooling ===== \n";
	//maxPoolingOutput.resize(input_n*input_c*poolingOutputHeight*poolingOutputWidth);
	int temp1i = input_h * input_w *input_c;
	int temp1o = poolingOutputHeight * poolingOutputWidth * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2i = ⁠n_idx * temp1i;
		int temp2o = ⁠n_idx * temp1o;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3i = ⁠c_idx * input_w * input_h + temp2i;
			int temp3o = ⁠c_idx * poolingOutputHeight * poolingOutputWidth + temp2o;
			for (int rowStride = 0; rowStride < poolingOutputHeight; rowStride++) {
				int temp4o = rowStride * poolingOutputWidth + temp3o;
				for (int colStride = 0; colStride < poolingOutputWidth; colStride++) {
					float maxValue = 0;
					int g_idx_o = colStride + temp4o;
					for (int x = rowStride * poolingStride; x < rowStride * poolingStride + poolingWindow; x++) {
						int temp4i = x * input_w + temp3i;
						for (int y = colStride * poolingStride; y < colStride * poolingStride + poolingWindow; y++) {
							int ⁠g_idx_i = y + temp4i;
							if (maxValue < maxPoolingInput[⁠g_idx_i])
							{
								maxValue = maxPoolingInput[⁠g_idx_i];
							}
						}
					}
					maxPoolingOutput[g_idx_o] = maxValue;
				}
			}
		}
	}
}

void fullyConnected(vector<float> &fullyConnectedOutput, vector<float> &fullyConnectedInput, vector<float> &fullyConnectedWeight, int fullyConnectedSize, int input_n, int input_c, int input_h, int input_w ) {
	 
	 int temp1i = input_w * input_h * input_c;
	 int temp1k = input_w * input_h * input_c;
	 for (int n_idx = 0; n_idx < input_n; n_idx++)
	 {
		 int temp2i = n_idx * temp1i;
		 int temp2o = n_idx * fullyConnectedSize;
		for (int f_idx = 0; f_idx < fullyConnectedSize; f_idx++)
		{
			float sum = 0;
			int g_idx_o = f_idx + temp2o;
			int temp2k = f_idx * temp1k;
			for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
			{
				int temp3i = ⁠c_idx * input_w * input_h + temp2i;
				int temp3k = ⁠c_idx * input_w * input_h + temp2k;
				for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
				{
					int temp4i = ⁠h_idx * input_w + temp3i;
					int temp4k = ⁠h_idx * input_w + temp3k;
					for (int w_idx = 0; w_idx < input_w; w_idx++)
					{
						int ⁠g_idx_i = w_idx + temp4i;
						int g_idx_k = w_idx + temp4k;
						sum += fullyConnectedInput[⁠g_idx_i] * fullyConnectedWeight[g_idx_k];
					}
				}
			}
			fullyConnectedOutput[g_idx_o] = sum;
		}
	}
}

// 발산 방지 수정 필요
void softMax(vector<float> &softMaxOutput, vector<float> &softMaxInput, int input_n, int lastNodeNumber)
 {
	 for (int n = 0; n < input_n; n++) {
		 float sum = 0.0;
		 for (int i = 0; i < lastNodeNumber; i++)
		 {
			 int ⁠g_idx_i = n * lastNodeNumber + i;
			 sum += exp(softMaxInput[⁠g_idx_i] );
		 }
		 for (int i = 0; i < lastNodeNumber; i++)
		 {
			 //int ⁠g_idx_i = n * lastNodeNumber + i;
			 int ⁠g_idx_o = n * lastNodeNumber + i;
			 softMaxOutput[⁠g_idx_o] = exp(softMaxInput[⁠g_idx_o])/sum;
		 }
	 }
 }

int main()
{
	
	cout << "4D([N][C][H][W]) Convolutions ! \n\n";
	cout << "===== 1. Image loading ===== \n";
	vector<pair<Mat, string>> imgBox; // 이미지 데이터, 이미지 이름
	imgBox = TraverseFilesUsingDFS("C:\\cifar\\test10");// 이미지가 저장되어 있는 폴더 경로

	//입력변수
	int input_n = imgBox.size(); // 10
	int input_c = imgBox[0].first.channels(); // 3
	int input_h = imgBox[0].first.rows;//H 32 
	int input_w = imgBox[0].first.cols;//W 32
	int inputDataSize = input_n * input_c * input_h * input_w;
	vector<float> inputVec(inputDataSize);

	cout << "===== 2. zeroPadding and mat -> vector  ===== \n";
	// mat 형식 - > 4차 행렬
	int temp1 = input_w * input_h * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		unsigned char* temp = imgBox[⁠n_idx].first.data;
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				int temp5 = input_c * input_w * ⁠h_idx;
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int ⁠g_idx = w_idx + temp4;
					inputVec[⁠g_idx] = temp[temp5 + input_c * w_idx + ⁠c_idx];
				}
			}
		}
	}


	cout << "===== Input Value check  ===== \n";
	valueCheck(inputVec, input_n, input_c, input_h, input_w, 1);

	// 커널, window, weight
	//cout << "===== 3. weight(filter) generation ===== \n";
	//InitWeightsXavier(4, 3, 3, 3);
	//  임시 커널
	vector<float> kernel(4 * 3 * 3 * 3); // [oc, ic, h, w]
	for (int i = 0; i < kernel.size(); i++) {
		kernel[i] = 1;
	}


	cout << "===== 4. zeroPadding ===== \n";

	int topPadingSize = 1;
	int bottomPadingSize = 1;
	int leftPadingSize = 1;
	int rightPadingSize = 1;
	int zeroPaddingOutputSize = input_n * input_c * (input_h + topPadingSize + bottomPadingSize)*(input_w + leftPadingSize + rightPadingSize);

	vector<float> inputVecWithZeroPadding(zeroPaddingOutputSize);
	zeroPadding(inputVecWithZeroPadding, inputVec, input_n, input_c, input_h, input_w, topPadingSize, bottomPadingSize, leftPadingSize, rightPadingSize);
	valueCheck(inputVecWithZeroPadding, input_n, input_c, input_h + topPadingSize + bottomPadingSize, input_w + leftPadingSize + rightPadingSize, 1);
	
	//cout << "===== activation ===== \n";
	//activation(inputVec_ZeroPading);
	//valueCheck(inputVec_ZeroPading, input_n, input_c, input_h + topPadingSize + bottomPadingSize, input_w + leftPadingSize + rightPadingSize, 1);


	cout << "===== 5. Convolution ===== \n";
	int outputCh = 4;
	int kernelSize = 3;
	int stride = 1;
	int outputHeight = ((input_h + topPadingSize + bottomPadingSize - kernelSize) / stride) + 1;
	int outputWidth = ((input_w + leftPadingSize + rightPadingSize - kernelSize) / stride) + 1;
	int convOutputSize = input_n * outputCh * outputHeight * outputWidth;

	vector<float> conv1Output(convOutputSize);
	convolution(conv1Output, inputVecWithZeroPadding, kernel, kernelSize, stride, input_n, input_c, input_h + topPadingSize + bottomPadingSize, input_w + leftPadingSize + rightPadingSize, outputCh);
	valueCheck(conv1Output, input_n, outputCh, outputHeight, outputWidth, 1);


	cout << "===== 6. Pooling ===== \n";
	int poolingWindow = 4;
	int poolingStride = 4;
	int poolingOutputHeight = ((outputHeight - poolingWindow) / poolingStride) + 1;
	int poolingOutputWidth = ((outputWidth - poolingWindow) / poolingStride) + 1;
	int poolingOutputSize = input_n * outputCh * poolingOutputHeight * poolingOutputWidth;

	vector<float> poolingOutput(poolingOutputSize);
	//avgPooling(poolingOutput, conv1Output, input_n, outputCh, outputHeight, outputWidth, poolingWindow, poolingStride, poolingOutputHeight, poolingOutputWidth);
	maxPooling(poolingOutput, conv1Output, input_n, outputCh, outputHeight, outputWidth, poolingWindow, poolingStride, poolingOutputHeight, poolingOutputWidth);
	valueCheck(poolingOutput, input_n, outputCh, poolingOutputHeight, poolingOutputWidth, 1);
	

	cout << "===== 7. fullyConnected_1  ===== \n";
	int fullyConnectedSize = 100;
	int fullyWeightSize = fullyConnectedSize * outputCh * poolingOutputHeight * poolingOutputWidth;
	vector<float> fullyWeight(fullyWeightSize);
	for (int i = 0; i < fullyWeight.size(); i++) {
		fullyWeight[i] = 1;
	}
	int fullyOutputSize = input_n * fullyConnectedSize;
	vector<float> fullyOutput(fullyOutputSize);
	fullyConnected(fullyOutput, poolingOutput, fullyWeight, fullyConnectedSize, input_n, outputCh, poolingOutputHeight, poolingOutputWidth);
	valueCheck(fullyOutput, input_n, fullyConnectedSize);
	

	cout << "===== 8. fullyConnected_2  ===== \n";
	int fullyConnectedSize2 = 10;
	int fullyWeightSize2 = fullyConnectedSize2 * fullyConnectedSize;
	vector<float> fullyWeight2(fullyWeightSize2);
	for (int i = 0; i < fullyWeight2.size(); i++) {
		fullyWeight2[i] = 1;
	}

	int fullyOutputSize2 = input_n * fullyConnectedSize2;
	vector<float> fullyOutput2(fullyOutputSize2);
	fullyConnected(fullyOutput2, fullyOutput, fullyWeight2, fullyConnectedSize2, input_n, fullyConnectedSize, 1, 1);
	valueCheck(fullyOutput2, input_n, fullyConnectedSize2);
	
	
	cout << "===== 9. softmax_test  ===== \n";

	int fullyConnectedSize2 = 10;
	int input_n = 11;
	vector<float> softmaxTestOut(fullyConnectedSize2 * input_n);
	vector<float> softmaxTestIn(fullyConnectedSize2 * input_n);

	for (int i = 0; i < softmaxTestOut.size(); i++) {
		softmaxTestIn[i] = (i+1)*0.1 ;
	}

	softMax(softmaxTestOut, softmaxTestIn, input_n, fullyConnectedSize2);
	//valueCheck(softmaxTestIn, input_n, fullyConnectedSize2);cout << endl;
	valueCheck(softmaxTestOut, input_n, fullyConnectedSize2);

	return 0;
}