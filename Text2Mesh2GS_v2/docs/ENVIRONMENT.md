# WSL実行環境

## このPCでの使い方

WindowsのターミナルでUbuntu-22.04を開きます。

```powershell
wsl -d Ubuntu-22.04
```

Ubuntu内で実行します。

```bash
cd /mnt/c/Users/81808/Documents/ChatGPT/2025_YAMAMOTO/Text2Mesh2GS_v2
source activate_wsl.sh
python scripts/verify_runtime.py
python pipeline.py prepare --scene examples/scene_v2.json --output outputs/room_001
```

環境は`/home/dpc7/.venvs/text2mesh2gs-v2`です。`source activate_wsl.sh`は端末を開くたびに実行してください。仮想環境はWSL用なのでWindowsのPythonでは利用できません。

PyTorch、Diffusers、gsplatと前処理の依存関係をまとめた環境です。TRELLIS本体の環境とUnity Editorは別です。TRELLISの外部コマンドには既存のTRELLIS専用Pythonを絶対パスで指定してください。モデル重みのダウンロードや本学習は環境作成に含めません。

## 構成

- Python 3.10系列のvenv（system-site-packagesを使わない）
- PyTorch 2.7.1 / torchvision 0.22.1、CUDA 12.8版
- gsplat 1.5.3
- Diffusers 0.35.1 / Transformers 4.56.2
- NumPy 1.26.4
- 背景除去はrembg + CPU版ONNX Runtime
- CUDAコンパイラ：`/usr/local/cuda-12.8`
- このPCのBlender：`/home/dpc7/software/blender-5.1.2-linux-x64`

主要依存は`requirements-runtime.txt`に固定します。今回の全依存の実際のバージョンは`requirements-runtime-lock.txt`に保存し、setupはこのファイルがある場合に優先します。CUDA ToolkitとBlenderは仮想環境外の既存インストールを使用します。activateスクリプトでPATHを設定するため、システム全体の設定は変更しません。gsplatの初回実行ではCUDA拡張のビルドに時間がかかることがあります。

## 作り直す場合・別PCの場合

Python 3.10とvenv機能、対応するNVIDIAドライバー、CUDA Toolkit 12.8、C++コンパイラを用意します。PyTorchのCUDAライブラリだけではgsplatのコンパイルに必要なnvccは用意されません。

```bash
bash setup_wsl.sh
source activate_wsl.sh
python scripts/verify_runtime.py
```

このPCではvenv作成機能を持つ既存のPythonを使用しました。

```bash
T2M_PYTHON=/home/dpc7/miniforge/envs/gsplat/bin/python bash setup_wsl.sh
```

既存gsplat環境のパッケージは共有せず、新規環境にインストールします。ただしvenvのPython標準ライブラリは作成元Pythonに依存するため、作成元のConda環境を削除すると作り直しが必要です。

環境の配置先は`T2M_ENV_DIR`、CUDAの配置先は`T2M_CUDA_HOME`で変更できます。別PCではBlenderがPATHにあることも確認してください。仮想環境そのものはGitへ保存せず、再作成スクリプトを保存します。

## 検証の意味

`verify_runtime.py`はパッケージのimport、CUDA上の小さなGaussian描画と逆伝播、Blenderコマンドの起動を検査します。画像生成モデル・TRELLIS推論・実アセットでの学習・Blender各スクリプトの互換性までは保証しません。

2026-09-16、このPCで以下を確認しました。

- Python 3.10.20、RTX 5070 Ti、PyTorch 2.7.1+cu128。
- pip check：依存関係の不整合なし。
- gsplat 1.5.3：CUDA拡張のコンパイル、32×32描画、逆伝播、有限値チェックが成功。
- Diffusers・Transformers・rembg・ONNX Runtime等のimportが成功。
- Blender 5.1.2のコマンド起動が成功。
- 新環境でPython回帰テスト14件が成功。

## 参考

- [PyTorch公式のバージョン別インストール手順](https://pytorch.org/get-started/previous-versions/)
- [gsplat公式のインストール説明](https://github.com/nerfstudio-project/gsplat#installation)
