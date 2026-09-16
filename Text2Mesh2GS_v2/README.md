# Text2Mesh2GS — ローカル画像認識版

最初に生成した部屋画像をローカルのQwen2.5-VLで読み取り、家具インスタンスと窓背景を抽出します。同じ画像と認識結果からLLMが意味制約を生成し、Unityの幾何探索で配置します。APIキーは不要です。

- [実行手順](docs/USAGE.md) — 初期設定からUnity配置までのコマンド
- [仕組み・スコア](docs/ARCHITECTURE.md)
- [Python/CUDA環境](docs/ENVIRONMENT.md)
- [検証範囲](docs/VALIDATION.md)

## 物体別レビューへの更新

[更新後の実行手順と確認待ちの解決方法](docs/OBJECT_REVIEW.md)を参照してください。23から再実行が必要です。固定棚・窓は確認済みの位置を登録します。

## これまでの変更

- 23：画像パスを記録するだけの処理から、実画像を送るローカルVLM認識へ変更。
- 34：家具カテゴリから固定関係を選ぶ処理から、画像と物体一覧に基づくLLM生成へ変更。
- 家具の個数、見た目、推定寸法、Mesh/Gaussian、生成プロンプトをモデルが出力。
- 初回認識結果を同じ画像で再照合するLLMレビューを既定で実行し、漏れ・重複を見直す。
- bbox・根拠・信頼度・不確かな点を保存。信頼度はモデルの自己申告値で、校正済み確率ではありません。
- 不正出力は検証エラーをモデルへ返し、最大2回の修復まで。失敗時は停止し、ルール方式で穴埋めしません。
- 25/26の変換をprepareに統合。未使用・比較用の旧スクリプト13本と重複requirementsを削除。

## 最短の手順

Ollamaの準備・起動後、仮想環境を有効にして実行します。詳しくは[実行手順](docs/USAGE.md)を参照してください。

```bash
python scripts/23_build_scene_graph_from_text_image.py --scene_id room_001 --image_path outputs/scene_images/room_001/candidate_00.png
python scripts/34_generate_spatial_constraints.py --scene_graph outputs/scene_graphs/room_001/scene_graph.json --output outputs/scene_graphs/room_001/spatial_constraints.json
python pipeline.py prepare --scene outputs/scene_graphs/room_001/scene_graph.json --constraints outputs/scene_graphs/room_001/spatial_constraints.json --output outputs/room_001
```

`examples/scene_v2.json`は変換・Unity確認用の従来サンプルです。新しい画像認識結果ではありません。画像から生成する際は上記のscene_graphを使います。

## 残したファイル

| ファイル | 役割 |
|---|---|
| pipeline.py | 正規化・検証・各ジョブ出力・Gaussian実行 |
| scripts/22 / 23 / 34 | 部屋画像生成 → 家具認識 → 制約生成 |
| scripts/local_scene.py | ローカル通信、スキーマ、検証・修復 |
| scripts/28 / 29 / 30 | 家具画像 → TRELLISジョブ → 外部実行器 |
| scripts/31 / 03 | 背景画像 → 背景板GLB |
| scripts/04 / 10 / 12 | 必要時のGLB整理・テクスチャ変換・寸法正規化 |
| scripts/05 / 07_5 / 08 / 09 | 多視点画像 → Gaussian学習 → 検証 → PLY |
| scripts/render_settings.py / verify_runtime.py | 設定共有・CUDA環境確認 |
| unity/ | 家具・背景配置、Bounds、ID、補助部屋生成 |
| configs/ | 生成・推論・学習・外部TRELLISの設定 |
| setup_wsl.sh / activate_wsl.sh | Python環境の作成・起動 |
| setup_local_llm.py / serve_local_llm.sh | ローカルモデル基盤の導入・起動 |

## 重要な前提

単眼画像から正確な3D位置・実寸を復元する方式ではありません。部屋寸法は設定値、家具寸法はモデルの推定値です。家具の配置は意味制約と実アセットBoundsで決まります。認識漏れ・誤認識はあり得るため、生成前に画像と一覧を照合してください。

床・天井・壁の矩形部屋は設定された参照構造として作り、画像認識した家具と区別します。補助Unity部屋ビルダーは背面1窓の矩形部屋用で、画像の建築形状を自動復元しません。複数窓・任意IDはUnity側で参照を指定してください。

モデルの取得時はインターネットを使います。認識時は127.0.0.1のOllamaのみを使い、serveスクリプトはクラウド機能を無効化します。推論終了時にモデルをGPUから解放し、後続のSDXL/TRELLIS/gsplatとVRAMを共有します。

生成アセット・モデル重みはGitに含めません。TRELLIS本体とその実行コマンド設定は別途必要です。
