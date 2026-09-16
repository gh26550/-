# 実行手順 — ローカル画像認識版

## 0. 更新とPython環境

Ubuntu-22.04内で実行します。このPCでクローン済みの場合：

```bash
cd ~/Text2Mesh2GS-delivery
git pull --ff-only
cd Text2Mesh2GS_v2
source activate_wsl.sh
python -m pip install -r requirements-runtime-lock.txt
python scripts/verify_runtime.py
```

新規取得・環境作成は[環境説明書](ENVIRONMENT.md)を参照してください。作業ディレクトリは常に`Text2Mesh2GS_v2`です。旧00/01/02/17/18/19/20/21/24/25/26/27/35は削除しました。旧手順の25/26を呼ぶ必要はありません。

## 1. ローカルモデルの準備・起動

このPCにはOllamaとQwen2.5-VL 7Bを導入します。別PC・再導入時は：

```bash
python setup_local_llm.py
bash serve_local_llm.sh
```

`serve_local_llm.sh`の端末は開いたままにします。別のUbuntu端末で同じフォルダへ移動し、`source activate_wsl.sh`を実行してからモデルを取得します（初回のみ）。

```bash
~/.local/share/text2mesh2gs/ollama/bin/ollama pull qwen2.5vl:7b
```

以降はサーバーを起動した状態で実行します。既に11434番ポートでサーバーが動いていれば二重起動しません。推論設定は`configs/scene_pipeline.yaml`のlocal_llmです。APIキーは不要です。モデル本体はPythonのvenvとは別に`~/.local/share/text2mesh2gs/`へ保存します。

## 2. 最初の部屋画像を生成・選択

```bash
python scripts/22_generate_scene_images.py --config configs/scene_pipeline.yaml --scene_id room_001
```

入力テキストは`examples/scene_prompts.csv`です。`outputs/scene_images/room_001/`の候補を確認して1枚選びます。既存の部屋画像がある場合は22を省略し、その画像を23へ渡せます。

## 3. 画像を認識して家具を抽出

```bash
python scripts/23_build_scene_graph_from_text_image.py --scene_id room_001 --image_path outputs/scene_images/room_001/candidate_00.png
```

既存の元プロジェクトの画像を使う例：

```bash
python scripts/23_build_scene_graph_from_text_image.py --scene_id room_001 --image_path /home/dpc7/Text2Mesh2GS/outputs/scene_images/room_001/candidate_00.png
```

`outputs/scene_graphs/room_001/scene_graph.json`に個々の物体、bbox、見た目、生成用プロンプト、推定寸法、信頼度、根拠を保存します。椅子が2脚なら2インスタンスです。クッションや本などはソファ・棚のアセットに含め、同じ物体の重複を避けます。天井・壁に固定された照明など、現在の家具配置で扱えない物体はuncertaintiesに記録して手動対応します。画像が存在しなければ停止します。テキストから家具一覧を補完しません。

画像と認識一覧を照合し、必要ならJSONを修正します。推定寸法は計測値ではありません。床・壁・天井は設定寸法の参照構造です。家具の初期座標はゼロで、配置の答え座標を生成しません。

23は既定で初回認識後にもう一度画像を見直し、認識漏れ・重複・表現形式を点検します。初回とレビューの監査JSONを別々に保存します。`local_llm.vision_review: false`で省略できますが、人による画像照合は必要です。

画像端の座標が寸法の2%以内だけ超過した場合は画像端で切り詰め、同じカテゴリ・表現形式・可動性・元のbboxが完全一致する別IDの検出は最初のIDに統合します。元出力は監査JSONのraw_result、補正内容はcorrectionsに保存し、uncertaintiesにも確認を促す注記を追加します。似た位置の別物体や異なるカテゴリは自動統合しません。大きな座標超過・同一ID重複・無効な矩形は引き続きエラーです。

`Model output invalid after bounded repair`の場合は、続けて表示される対象ID・理由と`.vision_audit.json`または`.vision_review.json`を確認します。サーバー接続失敗とは異なり、モデル応答の検証エラーです。

## 4. LLMで意味制約を生成

```bash
python scripts/34_generate_spatial_constraints.py --scene_graph outputs/scene_graphs/room_001/scene_graph.json --output outputs/scene_graphs/room_001/spatial_constraints.json
```

34も同じ画像を読みます。画像を移動した場合は`--image_path`で新しいパスを指定できますが、SHA256が一致する必要があります。

根拠がある場合にon/near/face to/部屋中心/窓背景のcenter・parallel・cover・visibility等を出力します。欠落ID、未対応関係、支持循環、方向矛盾、制約のない家具などを検査し、問題はLLMへ返して修復します。上限まで失敗したら停止します。監査JSONに試行結果とエラーを保存します。

## 5. Unity入力と生成ジョブへ変換

```bash
python pipeline.py prepare --scene outputs/scene_graphs/room_001/scene_graph.json --constraints outputs/scene_graphs/room_001/spatial_constraints.json --output outputs/room_001
python pipeline.py validate --scene outputs/room_001/scene.json
```

prepareはUnity用JSONのほか`mesh_object_prompts.csv`と`gaussian_background_prompts.json`も作成します。旧25/26は不要です。生成プロンプトはモデルの出力を引き継ぎ、カテゴリ別の定型家具判断は行いません。GLB/PLY本体はまだ生成されません。

## 6. 家具を生成

```bash
python scripts/28_generate_object_front_images.py --config configs/scene_pipeline.yaml --mesh_csv outputs/room_001/mesh_object_prompts.csv --remove_bg
python scripts/29_prepare_trellis_jobs.py --mesh_csv outputs/room_001/mesh_object_prompts.csv --output_json outputs/room_001/trellis_jobs.json --output_csv outputs/room_001/trellis_jobs.csv
python scripts/30_run_trellis_jobs.py --jobs outputs/room_001/trellis_jobs.json --command-config configs/trellis.command.json --output outputs/room_001/trellis_run.json
```

`configs/trellis.command.example.json`をコピーして、実際の外部TRELLISコマンドを設定してください。例示パスのままでは動きません。計画を確認してから30に`--execute`を付けます。29の`--candidate_index`で使用画像を選べます。

必要に応じてBlenderで整理・正規化します。以下の入出力名・高さは例です。

```bash
blender --background --python scripts/04_clean_mesh_blender.py -- --input outputs/raw_chair.glb --output outputs/clean_chair.glb
blender --background --python scripts/10_convert_glb_webp_to_png.py -- --input outputs/clean_chair.glb --output outputs/png_chair.glb
blender --background --python scripts/12_normalize_glb_for_unity.py -- --input outputs/png_chair.glb --output outputs/mesh_assets/room_001/chair_01/chair_01.glb --target_height 0.9 --yaw_deg 0
```

Blenderスクリプトは作業シーンをクリアするため、独立したbackgroundプロセスで実行してください。実アセットの寸法・正面を確認します。既存アセットがある場合は期待GLBパスへ配置して生成を省略できます。

## 7. 背景を生成してGaussian化

```bash
python scripts/31_generate_background_images.py --prompts outputs/room_001/gaussian_background_prompts.json
```

計画を確認して`--execute`を追加します。モデルが認識した背景のIDに合わせて、選んだ画像からGLBを作ります。以下の`outside_view_01`は例です。

```bash
blender --background --python scripts/03_create_background_plate_glb.py -- --image outputs/background_images/room_001/outside_view_01/candidate_00.png --output outputs/background_assets/room_001/outside_view_01/background_plate.glb
```

`outputs/room_001/scene.json`内の該当背景の`asset.source_path`を上記GLBパスへ設定して再変換します。

```bash
python pipeline.py prepare --scene outputs/room_001/scene.json --output outputs/room_001
python pipeline.py gaussian --jobs outputs/room_001/gaussian_import_jobs.json --config configs/gaussian.json --output outputs/gaussian_run
```

plan.jsonを確認し、同じgaussianコマンドへ`--execute`を追加すると05→07_5→08→09が動きます。背景板は平面表現であり、真の奥行き復元ではありません。背景が認識されていないシーンではこの工程を省略します。

## 8. Unity配置

1. `unity/`のC#とprepare出力のlayout_for_unity.json / spatial_constraints.jsonをAssets内へコピー。
2. 既存インポーターで家具GLB・Gaussian PLYをシーンへ導入。
3. 床中心を原点とするroomFrame（world scale=1）、家具のmovableRoot、固定構造のreferenceRootを用意。
4. 各対象のSceneObjectIdV2を今回認識されたIDと一致させる。モデル正面が+Z以外ならforwardAxisを設定。
5. SemanticScenePlacerV2にJSONと参照を設定。ContextMenuからApply→Audit→Export。
6. 背景はWindowBackgroundPlacerV2へ実際のgaussianId・windowId・wallIdとTransformを指定。窓基準+Zは屋外。
7. RendererからBoundsを取得できないGaussianにはPlacementBounds.localBoundsを設定。Apply→Audit→Export後、実カメラで見た目を確認。

補助ReferenceRoomBuilderV2は背面1窓の矩形部屋専用です。画像の窓位置・数・建築形状を自動再現しません。任意の窓IDや複数窓は固定構造を手動で用意し、背景ごとにコンポーネントを指定します。正規化済み家具はapplyJsonScale=falseのまま使います。

家具配置は意味制約と物理条件を満たす候補を探します。背景のcoverを求める場合はfitInsideとの目的の違いに注意してください。visible throughはカメラ検証待ちとして報告します。

```bash
python pipeline.py assets --scene outputs/room_001/scene.json
```

GLB/PLYが未配置ならこの検査は失敗します。画像認識やJSON生成の失敗とは別です。Unityレポートは`Assets/GeneratedLayoutsV2/`へ保存されます。
