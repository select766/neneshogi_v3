
// シリアライズされた実行エンジンを実行する際に必要な情報(POD型)
struct DNNEngineInfo
{
	int serializedEngineSize; //エンジンファイルのサイズ(bytes)
	int inputDims[4];			 //batch size (dummy), channel, height, width
	int inputSizePerSample;		 //1サンプルあたりの要素数(バイト数は*sizeof(float))
	int outputPolicyDims[2];	 //batch size, channel
	int outputPolicySizePerSample;
	int outputValueDims[2]; //batch size, channel
	int outputValueSizePerSample;
	int batchSizeMin;
	int batchSizeMax;
	char inputTensorName[256];
	char outputPolicyTensorName[256];
	char outputValueTensorName[256];
	int profileForBatchSize[32769]; //バッチサイズごとのprofile番号(可変長にできないため大きめに取っている)
};
