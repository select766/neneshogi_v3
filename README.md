# ねね将棋 (v3.1; 2020/05 世界コンピュータ将棋オンライン大会向け)

ねね将棋は、深層学習(Deep Learning)を用いた将棋AIです。

環境構築が面倒なので棋譜検討用にはお勧めできません。

2019/05 第29回世界コンピュータ将棋選手権 一次予選10位（一次予選参加40チーム中、別に二次予選からのシード16チーム）
2020/05 世界コンピュータ将棋オンライン大会1日目12位（1日目参加27チーム中）、2日目22位（2日目からのシード12チーム含む28チーム中）

# 対局プログラムのビルド
## Windows
子プロセスとしてpythonを呼び出し、pytorchモデルを動作させる。学習用のpython環境セットアップが必要。

Visual Studio 2019を利用。

`YaneuraOu.sln`でソリューション構成`Release-user`でビルド。`build/user/Yaneuraou-user.exe`が出来る。

同じディレクトリに`nenefwd.bat`を以下のような内容で設置。`Yaneuraou-user.exe`からはこのバッチファイルが実行され、副次的にpythonが呼び出される。

```
@echo off
call C:\path\to\Anaconda3\Scripts\activate.bat C:\path\to\Anaconda3\envs\neneshogi2020
python -m neneshogi.nenefwd.nenefwd %*
```

2行目はAnacondaの仮想環境を有効化する設定。環境により異なる。

定跡を使用する場合(デフォルト)、[https://github.com/yaneurao/YaneuraOu/releases/tag/v4.73_book](https://github.com/yaneurao/YaneuraOu/releases/tag/v4.73_book)から`standard_book.zip`をダウンロード・解凍して`build/user/book/standard_book.db`に設置。

## Linux

TensorRTを用い、ONNXフォーマットのモデルを実行する。TensorRTを使用する都合上、NVIDIA GPU上でしか動かせない。(プリプロセッサ`DNN_EXTERNAL`の定義により、Windowsと同様外部pythonプロセスによる評価も可能)

Ubuntu 18.04環境を想定する。

```
sudo apt-get install build-essential clang-7 g++-8
```

このほか、CUDA、cuDNN、TensorRTのパスを通す必要あり。

```
cd script
./linux_build.sh
```

これで実行バイナリ`build/user/YaneuraOu-user-linux-clang-avx2`が生成できるはず。

定跡の設置はWindowsと同様。

エンジンを起動し、ONNXモデルをTensorRTエンジンに変換する。
```
usi
user tensorrt_engine_builder /path/to/model.onnx /path/to/dst 1 126 1-1-15-15-126-126 16
```

(意味 `onnxModelPath dstDir batchSizeMin batchSizeMax profileBatchSizeRange fpbit`)

`/path/to/dst`（ディレクトリ）が作成されその中にTensorRT関係のファイルが生成される。このディレクトリを将棋所設定のEvalDirに指定する。

クラウド設定例は [awssetting.md](awssetting.md) 参照

# 主な設定
将棋所のオプションから設定できる。(デフォルトは省略)

大会時はAWS p3.16xlarge使用のため、CPU64コア、GPU8台だった。

|項目|意味|大会時設定|1GPU設定|
|---|---|---|---|
|Threads|CPUの探索スレッド数(3以上が必要)|58|4|
|SlowMover|探索予定時間の引き延ばし率[%]|200|200|
|MaxMovesToDraw|引き分けまでの手数|320|320|
|EvalDir|下記参照|?|?|
|BatchSize|GPU評価バッチサイズ|126|126|
|GPU|使用するGPU番号(-1=CPU)|0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7|0,0|
|DNNFormatBoard|DNNの入力形式|1|1|
|DNNFormatMove|DNNの方策出力形式|1|1|
|LeafMateSearchDepth|探索木の末端で詰み探索をする際の深さ|5|5|
|MCTSHash|MCTSのハッシュテーブルサイズの上限(MB)|80000|10000|
|PvInterval|読み筋出力時間間隔[ms]|1000|1000|
|RootMateSearch|ルート局面からの詰み探索をするかどうか|true|true|
|PolicyOnly|方策関数での即指し|false|false|
|LimitedBatchSize|探索局面数が少ない間のバッチサイズ|15|15|
|LimitedUntil|探索局面数が少ないと判定する局面数|10000|10000|
|EarlyStopProb|今後指し手が変化する確率がこの値[%]未満になったら指す|5|5|

EvalDirは、TensorRTを使う場合はONNXモデルから生成したエンジンの出力ディレクトリ、nenefwdを使う場合はpytorchの学習スナップショットディレクトリ(`model.pt`がある)。

定跡をやねうら王標準定跡と定跡なしで自己対局したところ、若干定跡なしのほうが勝率が良かったため、大会2日目では定跡なし(`no_book`)とした。

ハッシュテーブルサイズを環境変数`NENESHOGI_NODE_HASH_SIZE`で指定すると起動直後からメモリ確保・ゼロクリアを開始する。確保には1GB/sec程度かかるため。必ず設定値の`MCTSHash`と同じ値を指定すること。大会時、対局サーバ上で相手とマッチングする前にメモリ確保することで、スムーズに対局開始可能となる。2局以上連続して行う場合には使えないかもしれない。

エンジンクラッシュ・回線切断時のバックアップとして用いる即指しエンジン設定(デフォルトは省略)は以下の通り。[shogi-usi-failover](https://github.com/select766/shogi-usi-failover)を用いてクラッシュ時に切り替える。

|項目|大会時設定|
|---|---|
|EvalDir|C:\\..\\resnetaz_192ch_19block\\checkpoints\\train_029700192|
|DNNFormatBoard|1|
|DNNFormatMove|1|
|MCTSHash|32|
|PolicyOnly|true|


# 学習環境セットアップ
やねうら王のPackedSfenValue形式の棋譜から教師あり学習する機能のみ搭載されている。

やねうら王の一部を切り出したpython moduleのコンパイルが必要で、Windowsにのみ対応。

## 環境
- Visual Studio 2017 (C++)
- Python 3.7 (Anaconda)
- PyCharm

## ビルド
```
git submodule init
git submodule update
```

`YaneuraOu.sln`を開きソリューション構成`Release-user-py`をビルド。

`neneshogi`ディレクトリで`python addpath.py`を実行。ビルドされたpythonモジュール`neneshogi_cpp`をimportパスに入れる。

`neneshogi`ディレクトリで`python setup.py develop`を実行。pythonのモジュール`neneshogi`をimportパスに入れる。

## 方策学習
やねうら王形式(`PackedSfenValue`)の学習棋譜が必要。
[例](http://yaneuraou.yaneu.com/2018/01/23/%E6%9C%88%E5%88%8A%E6%95%99%E5%B8%AB%E5%B1%80%E9%9D%A2-2018%E5%B9%B41%E6%9C%88%E5%8F%B7/)

各局面に対して、指し手(policy)および勝敗(value)を学習する。

棋譜データを`shuffle_sfen[12].bin`とする。

適当なディレクトリ(以下`traindir`)を作成。

モデル構造を定義する`model.yaml`を設置

```
model: ResNetAZ
kwargs:
  ch: 192
  depth: 19
  block_depth: 2
  move_hidden: 256
```

学習方法を定義する`train.yaml`を設置

```
optimizer:
  kwargs:
    lr: 0.01
    momentum: 0.9
lr_scheduler:
  kwargs: {}
loss:
  policy: 1.0
  value: 1.0
dataset:
  train:
    data:
      path: path/to/shuffle_sfen1.bin
      count: 167000000
      skip: 0
    loader:
      batch_size: 256
  val:
    data:
      path: path/to/shuffle_sfen2.bin
      count: 10000
      skip: 0
    loader:
      batch_size: 256
manager:
  batch_size: 256
  val_frequency: 1000000
  exit_lr: 0.9e-6
```

学習実行
```
python -m neneshogi.pt_train traindir
```

損失・正解率の推移をtensorboardで可視化できる。

```
tensorboard --logdir traindir/log
```

validationのたびに、モデル及び学習再開用のデータが`traindir/checkpoints/train_<iteration>`に出力される。`model.pt`がモデル本体。

学習を任意のタイミングで中断したいときは、`traindir/deletetostop.tmp`ファイルを削除する。直ちにモデル及び学習再開用のデータが`traindir/checkpoints/train_<iteration>`に出力される。

学習を再開する際は、学習再開用のデータがあるディレクトリを指定して次のようなコマンドを実行する。

```
python -m neneshogi.pt_train traindir --resume traindir/checkpoints/train_<iteration>
```

## TensorRTを用いたモデル実行
TensorRTでモデルを実行する場合、まずpytorchのモデルをONNXフォーマットに変換する必要がある。

```
python -m neneshogi.export_onnx traindir/checkpoints/train_123456/model.pt traindir/checkpoints/train_123456/model.onnx
```

ここから先はやねうら王エンジン上での操作になる。

====

以下、やねうら王のREADME

# About this project

YaneuraOu mini is a shogi engine(AI player), stronger than Bonanza6 , educational and tiny code(about 2500 lines) , USI compliant engine , capable of being compiled by VC++2017

やねうら王miniは、将棋の思考エンジンで、Bonanza6より強く、教育的で短いコード(2500行程度)で書かれたUSIプロトコル準拠の思考エンジンで、VC++2017でコンパイル可能です。

[やねうら王mini 公式サイト (解説記事、開発者向け情報等)](http://yaneuraou.yaneu.com/YaneuraOu_Mini/)

[やねうら王公式 ](http://yaneuraou.yaneu.com/)

# お知らせ

- 2018/4/28 23:00 WCSC28に向けて探索部のアップデート作業中です。
- 2018/4/30 06:00 探索部のアップデート完了。やねうら王2018 OTAFUKU V4.82。実行ファイルを公開しました。
- 2018/5/ 3 12:00 置換表サイズの2^N制限を撤廃する改良をしましたが、まだテストあまりしていないのでWCSC28参加者はマージしないことを強く推奨。


# やねうら王エンジンの大会での戦績

- 2017年 世界コンピュータ将棋選手権(WCSC27) 『elmo』優勝
- 2017年 第5回将棋電王トーナメント(SDT5) 『平成将棋合戦ぽんぽこ』優勝

# 現在進行中のサブプロジェクト

## やねうら王2018 OTAFUKU (やねうら王2018 with お多福ラボ)

今年は一年、これでいきます。


## やねうら王詰め将棋solver

《tanuki-さんが開発中》

長手数の詰将棋が解けるsolverです。

# 過去のサブプロジェクト

過去のサブプロジェクトである、やねうら王nano , mini , classic、王手将棋、取る一手将棋、協力詰めsolver、連続自己対戦フレームワークなどはこちらからどうぞ。

- [過去のサブプロジェクト](/docs/README2017.md)

## やねうら王評価関数ファイル

- やねうら王2017Early用 - Apery(WCSC26)、Apery(SDT4)＝「浮かむ瀬」の評価関数バイナリがそのまま使えます。
- やねうら王2017 KPP_KKPT型評価関数 - 以下のKPP_KKPT型ビルド用評価関数のところにあるものが使えます。
- やねうら王2018 Otafuku用 KPPT型　→ やねうら王2017Early(KPPT)と同様
- やねうら王2018 Otafuku用 KPP_KKPT型　→ やねうら王2017Early(KPP_KKPT)と同様

### 「Re : ゼロから始める評価関数生活」プロジェクト(略して「リゼロ」)

ゼロベクトルの評価関数(≒駒得のみの評価関数)から、「elmo絞り」(elmo(WCSC27)の手法)を用いて強化学習しました。従来のソフトにはない、不思議な囲いと終盤力が特徴です。
やねうら王2017Earlyの評価関数ファイルと差し替えて使うことが出来ます。フォルダ名に書いてあるepochの数字が大きいものほど新しい世代(強い)です。

- [リゼロ評価関数 epoch 0](https://drive.google.com/open?id=0Bzbi5rbfN85Nb3o1Zkd6cjVNYkE) : 全パラメーターがゼロの初期状態の評価関数です。
- [リゼロ評価関数 epoch 0.1](https://drive.google.com/open?id=0Bzbi5rbfN85NNTBERmhiMGZlSWs) : [解説記事](http://yaneuraou.yaneu.com/2017/06/20/%E5%BE%93%E6%9D%A5%E6%89%8B%E6%B3%95%E3%81%AB%E5%9F%BA%E3%81%A5%E3%81%8F%E3%83%97%E3%83%AD%E3%81%AE%E6%A3%8B%E8%AD%9C%E3%82%92%E7%94%A8%E3%81%84%E3%81%AA%E3%81%84%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0/)
- [リゼロ評価関数 epoch 1から4まで](https://drive.google.com/open?id=0Bzbi5rbfN85NNWY0RTJlc2x5czg) : [解説記事](http://yaneuraou.yaneu.com/2017/06/12/%E4%BA%BA%E9%96%93%E3%81%AE%E6%A3%8B%E8%AD%9C%E3%82%92%E7%94%A8%E3%81%84%E3%81%9A%E3%81%AB%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AE%E5%AD%A6%E7%BF%92%E3%81%AB%E6%88%90%E5%8A%9F/)
- [リゼロ評価関数 epoch 5から6まで](https://drive.google.com/open?id=0Bzbi5rbfN85NSS0wWkEwSERZVzQ) : [解説記事](http://yaneuraou.yaneu.com/2017/06/13/%E7%B6%9A-%E4%BA%BA%E9%96%93%E3%81%AE%E6%A3%8B%E8%AD%9C%E3%82%92%E7%94%A8%E3%81%84%E3%81%9A%E3%81%AB%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AE%E5%AD%A6%E7%BF%92/)
- [リゼロ評価関数 epoch 7](https://drive.google.com/open?id=0Bzbi5rbfN85NWWloTFdMRjI5LWs) : [解説記事](http://yaneuraou.yaneu.com/2017/06/15/%E7%B6%9A2-%E4%BA%BA%E9%96%93%E3%81%AE%E6%A3%8B%E8%AD%9C%E3%82%92%E7%94%A8%E3%81%84%E3%81%9A%E3%81%AB%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AE%E5%AD%A6%E7%BF%92/)
- [リゼロ評価関数 epoch 8](https://drive.google.com/open?id=0Bzbi5rbfN85NMHd0OEUxcUVJQW8) : [解説記事](http://yaneuraou.yaneu.com/2017/06/21/%E3%83%AA%E3%82%BC%E3%83%AD%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0epoch8%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F%E3%80%82/)

### やねうら王 KPP_KKPT型ビルド用評価関数

やねうら王2017 KPP_KKPT型ビルドで使える評価関数です。

- [リゼロ評価関数 KPP_KKPT型 epoch4](https://drive.google.com/open?id=0Bzbi5rbfN85NSk1qQ042U0RnUEU) : [解説記事](http://yaneuraou.yaneu.com/2017/09/02/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E3%80%81kpp_kkpt%E5%9E%8B%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AB%E5%AF%BE%E5%BF%9C%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/)

### Shivoray(シボレー) 全自動雑巾絞り機

自分で自分好みの評価関数を作って遊んでみたいという人のために『Shivoray』(シボレー)という全自動雑巾絞り機を公開しました。

- [ShivorayV4.71](https://drive.google.com/open?id=0Bzbi5rbfN85Nb292azZxRmU0R1U) : [解説記事](http://yaneuraou.yaneu.com/2017/06/26/%E3%80%8Eshivoray%E3%80%8F%E5%85%A8%E8%87%AA%E5%8B%95%E9%9B%91%E5%B7%BE%E7%B5%9E%E3%82%8A%E6%A9%9F%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%97%E3%81%9F/)

## 定跡集

やねうら王2017Earlyで使える、各種定跡集。
ダウンロードしたあと、zipファイルになっているのでそれを解凍して、やねうら王の実行ファイルを配置しているフォルダ配下のbookフォルダに放り込んでください。

- コンセプトおよび定跡フォーマットについて : [やねうら大定跡はじめました](http://yaneuraou.yaneu.com/2016/07/10/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E5%A4%A7%E5%AE%9A%E8%B7%A1%E3%81%AF%E3%81%98%E3%82%81%E3%81%BE%E3%81%97%E3%81%9F/)
- 定跡ファイルのダウンロードは[こちら](https://github.com/yaneurao/YaneuraOu/releases/tag/v4.73_book)

## 世界コンピュータ将棋選手権および2017年に開催される第5回将棋電王トーナメントに参加される開発者の方へ

やねうら王をライブラリとして用いて参加される場合、このやねうら王のGitHub上にあるすべてのファイルおよび、このトップページから直リンしているファイルすべてが使えます。

## ライセンス

やねうら王プロジェクトのソースコードはStockfishをそのまま用いている部分が多々あり、Apery/SilentMajorityを参考にしている部分もありますので、やねうら王プロジェクトは、それらのプロジェクトのライセンス(GPLv3)に従うものとします。

「リゼロ評価関数ファイル」については、やねうら王プロジェクトのオリジナルですが、一切の権利は主張しませんのでご自由にお使いください。
