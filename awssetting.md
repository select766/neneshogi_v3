AWSクラウド上でのエンジン動作のための環境構築メモ

再現が取れるような清書はできてないのでご了承ください。

ローカルlinuxで
ssh-keygen -t rsa
=> id_rsa.wcsc30, id_rsa.wcsc30.pub が生成された

AWSマネジメントコンソールから
キーペアのインポート＝＞id_rsa.wcsc30.pubをインポート

Deep Learning AMI (Ubuntu 18.04) Version 26.0 - ami-07729b5941107618c
をベースイメージとして利用。pytorchが最初からcuda有効で入っている。
p3.2xlargeインスタンスを立ち上げ。ディスク容量90GB。

TensorRTとそれに必要なcuda, cudnnの設置（デフォルトで入ってるものは10.1/7.6.2で、TensorRTを動かすと警告が出る）
cudaは
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork
に従ってインストールコマンドを実行。
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

ライブラリ展開・パスを通す
```sh
sudo bash
mkdir /usr/local/mycudnn
cd /usr/local/mycudnn
tar zxvf cudnn-10.2-linux-x64-v7.6.5.32.tgz
tar zxvf TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz
echo /usr/local/mycudnn/cuda/lib64 >> /etc/ld.so.conf.d/00cudnn.conf
echo /usr/local/mycudnn/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib >> /etc/ld.so.conf.d/00cudnn.conf
ldconfig
```

local: neneshogiのルートディレクトリで

```sh
AWS_IP=13.231.167.233
rsync -av -e "ssh -i ~/.ssh/id_rsa.wcsc30" neneshogi/ ubuntu@$AWS_IP:neneshogi_python/
rsync -av -e "ssh -i ~/.ssh/id_rsa.wcsc30" build data ubuntu@$AWS_IP:
```

ここまで設定して、インスタンスを停止。イメージ（AMI）を作成。

スポットインスタンスを上記で作成したAMIイメージをもとに作成。p3.16xlargeインスタンスタイプ。

ローカル(windows)のssh設定(Gitをインストールしたときに入るsshコマンドを活用)
`awsip.bat`

```
@set AWS_IP=123.45.67.89
```

このIPアドレスは、インスタンスを立ち上げるたびに変更される。

`awsyaneuraou.bat`

```bat
@call awsip.bat
@"C:\Program Files\Git\usr\bin\ssh.exe" -i "C:\Users\Public\Documents\shogi\key\id_rsa.wcsc30" ubuntu@%AWS_IP% "cd ./build/user; NENESHOGI_NODE_HASH_SIZE=160000 ./YaneuraOu-user-linux-clang-avx2"
```

`awsssh.bat`
```bat
call awsip.bat
"C:\Program Files\Git\usr\bin\ssh.exe" -i "C:\Users\Public\Documents\shogi\key\id_rsa.wcsc30" ubuntu@%AWS_IP%
```

将棋所にはawsyaneuraou.batをエンジンとして設定。
インスタンスを立ち上げたときは、awsip.batを書き換えた後awsssh.batを起動してインスタンスの起動が完了していることおよび鍵の確認でyesを答えること。
