#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"
#include <time.h>

using namespace cv;
using namespace std;

/***************************************************************************
	filterTransform 함수,  U = G g Gt 
*****************************************************************************/

void filterTransform(vector<float> & Output_U, vector<float> filter, int out_ch, int in_ch) { 

	int temp1u = in_ch * 4 * 4;
	int temp1f = in_ch * 3 * 3;
	for (int ⁠n_idx = 0; ⁠n_idx < out_ch; ⁠n_idx++)
	{
		int temp2u = ⁠n_idx * temp1u;
		int temp2f = ⁠n_idx * temp1f;
		for (int ⁠c_idx = 0; ⁠c_idx < in_ch; ⁠c_idx++)
		{
			int u_idx = ⁠c_idx * 4 * 4 + temp2u;
			int f_idx = ⁠c_idx * 3 * 3 + temp2f;

			float *g1 = &filter[f_idx];
			float *g2 = &filter[f_idx + 1];
			float *g3 = &filter[f_idx + 2];
			float *g4 = &filter[f_idx + 3];
			float *g5 = &filter[f_idx + 4];
			float *g6 = &filter[f_idx + 5];
			float *g7 = &filter[f_idx + 6];
			float *g8 = &filter[f_idx + 7];
			float *g9 = &filter[f_idx + 8];

			Output_U[u_idx] = *g1;
			Output_U[u_idx + 1] = (*g1 + *g2 + *g3) / 2.f;
			Output_U[u_idx + 2] = (*g1 - *g2 + *g3) / 2.f;
			Output_U[u_idx + 3] = *g3;

			float temp1 = *g1 + *g4 + *g7;
			float temp2 = *g2 + *g5 + *g8;
			float temp3 = *g3 + *g6 + *g9;

			Output_U[u_idx + 4] = (temp1) / 2.f;
			Output_U[u_idx + 5] = (temp1 + temp2 + temp3) / 4.f;
			Output_U[u_idx + 6] = (temp1 - temp2 + temp3) / 4.f;
			Output_U[u_idx + 7] = (temp3) / 2.f;

			float temp4 = *g1 - *g4 + *g7;
			float temp5 = *g2 - *g5 + *g8;
			float temp6 = *g3 - *g6 + *g9;

			Output_U[u_idx + 8] = (temp4) / 2.f;
			Output_U[u_idx + 9] = (temp4 + temp5 + temp6) / 4.f;
			Output_U[u_idx + 10] = (temp4 - temp5 + temp6) / 4.f;
			Output_U[u_idx + 11] = (temp6) / 2.f;

			Output_U[u_idx + 12] = *g7;
			Output_U[u_idx + 13] = (*g7 + *g8 + *g9) / 2.f;
			Output_U[u_idx + 14] = (*g7 - *g8 + *g9) / 2.f;
			Output_U[u_idx + 15] = *g9;
						 
		
		}
	}
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

int main()
{
	cout << "Winograd Convolutions filterTransform function! \n\n";
	
	int FeatureCh = 3;
	int out_ch = 4;

	// h[O_Ch][ln_Ch][KernelSize_h][KernelSize_w] 
	// h[4][3][3][3]
	// 임시 filter 값 
	vector<float> h(out_ch*FeatureCh*3*3);
	float count = 1.f;
	int temp1 = FeatureCh * 3 * 3;
	for (int ⁠n_idx = 0; ⁠n_idx < out_ch; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < FeatureCh; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * 3 * 3 + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < 3; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * 3 + temp3;
				for (int w_idx = 0; w_idx < 3; w_idx++)
				{
					int g_idx = w_idx + temp4;
					h[g_idx] = count;
					count++;
				}
			}
		}
	}

	// filater 값 체크
	valueCheck(h, out_ch,FeatureCh , 3 ,3);

	// U[O_Ch][ln_Ch][4][4] 
	vector<float> U(out_ch*FeatureCh * 4 * 4);
	
	// filterTransform 수행
	filterTransform(U, h, out_ch, FeatureCh);

	// U 값 체크
	valueCheck(U, out_ch, FeatureCh, 4, 4);




	return 0;
}