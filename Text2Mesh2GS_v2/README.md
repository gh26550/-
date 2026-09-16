# Text2Mesh2GS V2

## 説明書の入口

- [導入・実行手順](docs/USAGE.md)：GitHubから取得してUnityで実行するまで。
- [動作原理とスコアの説明](docs/ARCHITECTURE.md)：データの流れ、配置探索、評価式、制約の意味。
- [検証記録](docs/VALIDATION.md)：確認済みの範囲と未検証の工程。

下記には各工程の具体的なコマンドも掲載しています。

既存コードを残したまま使える、自動空間生成・Unity配置の改良版です。Pythonの共通データ変換と実行計画、Gaussianの設定共有、Unityの家具配置・窓背景配置を含みます。

**このフォルダが新しい実行単位です。元のWSLプロジェクトやUnityプロジェクトには上書きしません。** Unityコンポーネントの名前空間は`Text2Mesh2GS.V2`です。

## 変更の要点

| 課題 | V2の対応 |
|---|---|
| JSON形式が工程ごとに異なる | `pipeline.py prepare`でschema 2.0へ変換し、同一データからUnity JSON・manifest・生成/importジョブを出力 |
| 欠落物体が点数から消える | 家具制約を分母に残し、スコア・評価カバー率・未達・欠落を分離 |
| 高得点でも物理違反が残る | シーン全体を再検査し、物理違反数→違反量→意味スコアで配置を比較 |
| テーブルと花瓶を一緒に動かせない | onの支持関係を辿り、支持物移動時に子の姿勢もまとめて移動 |
| 初期姿勢や世界座標に依存 | 単位scaleのroomFrameを基準に配置。既定は共通の初期姿勢から開始 |
| GaussianのRenderer.boundsが取れない | 明示的な`PlacementBounds.localBounds`を指定できる。未取得時は失敗して復元 |
| FitInsideでも窓より拡大される | fitは内側余白のみ、coverは外側余白のみ。式を分離 |
| visible throughの成功が不明確 | `requires_camera_validation`として未検証を明示 |
| 学習・描画・PLYで条件が違う | checkpointにrender_settings保存。08/09が継承し、PLYにも有効スケール範囲を適用 |
| 引数を変えても損失に効かない | alpha/projectionの独立した損失項を追加 |
| 手順の抜け | 05・12を同梱。背景画像生成31、外部TRELLIS呼び出し30を追加 |

## フォルダ

```text
pipeline.py                共通形式・検証・ジョブ作成・Gaussian実行
scripts/                   既存の実行スクリプトを引き継いだコピーと改良版
  05_...                   多視点画像・カメラJSON・整列NPZ
  07_5_... / 08_... / 09_... 設定を共有する学習・検証・出力
  12_...                   GLBの底面原点・寸法・yaw正規化
  20_...                   pipelineのGaussian実行を呼ぶ互換入口
  30_run_trellis_jobs.py    利用中のTRELLIS実行器を設定に従って呼ぶ
  31_generate_background_images.py 背景画像生成
unity/                     新しいUnityコンポーネント
configs/                   画像生成・Gaussian設定、TRELLISコマンド例
examples/                  受領したJSONの参照例、V2シーン例、入力CSV
tests/                     Python回帰テスト、C#数値テスト、Unityスモークテスト
```

生成物は`outputs/`へ出し、Gitには含めません。モデル重み、GLB、PLY、学習checkpoint、Unity Library、一時テストプロジェクトも含めません。

## 1. 最初にできる軽い確認

WSL/bashまたはPython 3.10以上がある端末で、このフォルダへ移動します。`prepare`・`validate`・実行計画の作成は、PyTorchやモデルの読み込みを行いません。

```bash
python pipeline.py prepare --scene examples/scene_v2.json --output outputs/room_001
python pipeline.py validate --scene outputs/room_001/scene.json
python -m unittest discover -s tests -v
```

元形式を変換する場合：

```bash
python pipeline.py prepare --scene examples/layout_legacy.json --constraints examples/constraints_legacy.json --output outputs/legacy_import
```

重複したsource/relation/targetは同じ重みなら統合します。受領した19行はbehindの重複があるため18行になります。重みが異なる重複・欠落ID・不正値・明らかな方向矛盾・onの循環はエラーです。room中心のtargetのみ仮想IDを許可します。

legacy入力の誤った座標は勝手に推測して書き換えません。`examples/scene_v2.json`は別のサンプルとして左右壁を±X、背面壁を部屋奥へ修正済みです。Unityで固定構造を作る場合にはReferenceRoomBuilderV2も使えます。

### 出力

- `scene.json`：正規化済みの共通データ
- `layout_for_unity.json`・`spatial_constraints.json`：V2のUnity入力
- `mesh_manifest.json`・`gaussian_manifest.json`
- `mesh_jobs.json`・`gaussian_jobs.json`：25/26用
- `gaussian_import_jobs.json`：20/pipeline gaussian用
- `validation.json`：検証結果。アセット実体を生成したという意味ではありません

`scene_id/object_id`をパスの識別子として使います。アセットの期待パスは次に統一します。

```text
outputs/mesh_assets/<scene_id>/<object_id>/<object_id>.glb
outputs/gaussian_assets/<scene_id>/<object_id>/<object_id>.ply
```

## 2. テキストから家具の準備まで

画像生成には対応するPyTorch/CUDA環境とモデルが必要です。依存候補は`requirements-images.txt`。既存の動作環境を利用してください。依存ファイルはバージョン固定の動作保証ではありません。

```bash
python scripts/22_generate_scene_images.py --config configs/scene_pipeline.yaml --scene_id room_001
python scripts/23_build_scene_graph_from_text_image.py --config configs/scene_pipeline.yaml --scene_id room_001 --image_path outputs/scene_images/room_001/candidate_00.png
python scripts/34_generate_spatial_constraints.py --scene_graph outputs/scene_graphs/room_001/scene_graph.json --output outputs/scene_graphs/room_001/constraints_initial.json
```

23は引き続きキーワードによるルールベースです。画像認識/LLMは実装していません。34の制約は初期案なので、必要な関係を確認してから共通データへ変換します。窓背景のcenter/parallel/cover/visibilityや部屋中心は必要に応じて追加してください。付属V2サンプルには受領した制約を含めています。

```bash
python pipeline.py prepare --scene outputs/scene_graphs/room_001/scene_graph.json --constraints outputs/scene_graphs/room_001/constraints_initial.json --output outputs/room_001
python scripts/25_prepare_mesh_jobs.py --jobs outputs/room_001/mesh_jobs.json --output_csv outputs/room_001/mesh_object_prompts.csv
python scripts/28_generate_object_front_images.py --config configs/scene_pipeline.yaml --mesh_csv outputs/room_001/mesh_object_prompts.csv --remove_bg
python scripts/29_prepare_trellis_jobs.py --mesh_csv outputs/room_001/mesh_object_prompts.csv --output_json outputs/room_001/trellis_jobs.json --output_csv outputs/room_001/trellis_jobs.csv
```

画像を確認し、必要なら29の`--candidate_index`で候補を指定します。

### TRELLIS実行

TRELLIS本体とそのCLIは環境依存で、今回その実行器は元プロジェクトに見つかっていません。30は架空のモデルAPIを実装せず、利用中の実行器を引数配列で呼び出します。

1. `configs/trellis.command.example.json`を別名でコピーする。
2. `command`を実際のTRELLIS実行コマンドに変更する。`{input_image}`と`{expected_glb}`は必須。
3. まず計画を出して確認し、その後`--execute`を付ける。

```bash
python scripts/30_run_trellis_jobs.py --jobs outputs/room_001/trellis_jobs.json --command-config configs/trellis.command.json --output outputs/room_001/trellis_run.json
# 実行する場合は同じコマンドに --execute を追加
```

サンプルの`/absolute/path/to/your/trellis_runner.py`は説明用です。そのままでは実行できません。

### Meshの整理・正規化

Blenderスクリプトは必ず独立したbackgroundプロセスで実行してください。スクリプトはBlenderシーンをクリアします。

```bash
blender --background --python scripts/04_clean_mesh_blender.py -- --input outputs/raw_chair.glb --output outputs/clean_chair.glb
blender --background --python scripts/10_convert_glb_webp_to_png.py -- --input outputs/clean_chair.glb --output outputs/png_chair.glb
blender --background --python scripts/12_normalize_glb_for_unity.py -- --input outputs/png_chair.glb --output outputs/mesh_assets/room_001/chair_01/chair_01.glb --target_height 0.9 --yaw_deg 0
```

高さ0.9は例です。各家具の実寸・正面を指定してください。12は依存グラフを更新してBoundsを取り、複数ルートのyawでは位置も一緒に回すよう修正しています。正規化済みアセットを使う場合、Unityの`applyJsonScale`はfalseのままにして二重スケーリングを避けます。

## 3. 背景画像からGaussianまで

```bash
python scripts/26_prepare_gaussian_jobs_from_scene.py --jobs outputs/room_001/gaussian_jobs.json --output_dir outputs/room_001
python scripts/31_generate_background_images.py --prompts outputs/room_001/gaussian_background_prompts.json
# 内容を確認して --execute を付ける（CUDAが必要）
```

背景候補を選び、03で板GLBを作成します。

```bash
blender --background --python scripts/03_create_background_plate_glb.py -- --image outputs/background_images/room_001/outside_view_01/candidate_00.png --output outputs/background_assets/room_001/outside_view_01/background_plate.glb
```

これは画像付き平面であり、窓外の本物の奥行きを復元する工程ではありません。立体背景GLBがあればそちらを使えます。

共通scene.json内の背景の`asset.source_path`を生成したGLBに合わせ、**GLBが存在してからprepareを再実行**します。ジョブの存在状態はprepare時に判定されます。サンプルscene_v2.jsonは上記の背景板パスを指定済みです。

```bash
python pipeline.py prepare --scene outputs/room_001/scene.json --output outputs/room_001
python pipeline.py gaussian --jobs outputs/room_001/gaussian_import_jobs.json --config configs/gaussian.json --output outputs/gaussian_run
# plan.jsonを確認後、同じコマンドに --execute を追加
```

この入口は05→07_5→指定複数視点の08→09を実行し、ステップごとにログと終了コードを保存します。失敗したら停止します。画像だけのジョブをGLB用の05へ渡すことは拒否します。

標準設定は10,000点・1,000ステップの確認用です。正しくUnityへ表示できることを確認してから増やしてください。

### 学習設定の一貫性

- 07_5は乱数seedと`render_settings`を保存。
- 08/09はcheckpointの設定を継承。CLIで明示した値のみ上書き。
- `alpha_weight`はアルファMSE、`projection_weight`は投影マスク外のアルファ抑制へ実際に適用。
- 09は学習時に描画された有効スケール範囲へclampしてPLYを出力。`.metadata.json`も保存。
- Z-buffer可視性を有効にした場合、実行計画は可視性なしの比較も作成。PLYには視点ごとのマスクを焼き込めません。
- 最終的な見た目は出力PLYをUnityで確認します。

既存PLYを使う場合は、レイアウトの期待PLYパスに配置してください。runnerは別パスのファイルを無断で移動せず、不一致をエラーにします。

## 4. Unityへ入れる

`unity/`内のC#をUnityの新しい`Assets/Text2Mesh2GS_V2/`へコピーします。既存コンポーネントと同時に同じ家具を動かさないでください。V2は旧SceneObjectIdを自動変換せず、`SceneObjectIdV2`を使います。

### 家具

1. 空のroomFrameを作成。部屋の床中心を原点とし、world scale=(1,1,1)。回転・移動は可能。
2. movableRootに家具GLB、referenceRootに固定構造を用意。家具ルート同士を入れ子にしない。
3. 家具・壁・窓にSceneObjectIdV2を付け、IDをJSONと一致させる。
4. 正面が+Z以外ならmarkerのforwardAxisを指定。
5. SemanticScenePlacerV2にroomFrame、2つのルート、prepare出力のlayout/constraintsを割り当てる。
6. `Apply Semantic Layout V2`、`Audit Current Layout V2`、`Export Placement Report V2`を実行。

壁・床がまだない場合は`ReferenceRoomBuilderV2`を追加し、layoutJsonとroomFrameを指定して`Build New Reference Room V2`を実行できます。矩形の部屋と背面の1窓を作り、既存オブジェクトは削除しません。前面は開放です。窓サイズ・位置はbuilderで指定してください。

### 家具の判定

```text
候補生成：部屋ローカルの格子 + 入出力両方向の関係先周囲 + 支持面候補
順序：onの深さ → 他の家具から参照される数 → ID（再試行では同順位をランダム化）
比較：物理違反数 → 違反量 → 意味制約スコア
支持物移動：onの子孫を同時移動・回転
最終監査：欠落・床/天井/部屋外・壁・家具重複・全家具制約を再評価
```

意味制約のスコアは0～100、評価カバー率も別に出します。欠落sourceも分母に含みます。全制約の達成と物理判定が揃った場合だけ`success`です。score=100でも未評価・物理違反があればsuccessにはしません。

配置候補に「JSONの答え座標」は使いません。deterministicStartは既定trueで、家具を共通の初期位置へ戻して探索します。固定構造とアセットの寸法は入力条件です。AABBによる近似・有限候補探索なので、大域最適解や複雑な凹形状の厳密衝突は保証しません。

### Gaussian背景

1. `WindowBackgroundPlacerV2`を追加し、gaussianとwindowBasisを明示的に指定。
2. windowBasisはworld scale=(1,1,1)、+Zが屋外、+Xが横、+Yが上。
3. 窓の4辺を指定するか、useExplicitOpeningで開口Boundsを入力。BoundsはwindowBasisのローカル座標。
4. 通常Rendererを持たないGaussianには`PlacementBounds`を追加して、アセットに合ったlocalBoundsを指定。
5. prepare出力のconstraintsを割り当ててApply/Audit/Exportを実行。

fitInsideは内側余白のみ、coverは外側余白のみを使います。behindは中心ではなくGaussianの最前面を窓の外へ出します。制約なしのApplyはscaleも変えません。Bounds取得に失敗した場合は元の姿勢へ戻します。

**fitInsideとcoverの完全達成は縦横比や余白によって両立しません。** fitInsideで余白を取ればcoverはunmetになるのが正常です。窓からのはみ出し防止と全面被覆のどちらを求めるかを決めて設定・制約を選んでください。visible throughはカメラ検証待ちとして扱い、幾何学的調整だけで視認性成功とは報告しません。

回転のparallel評価は投影厚みの近似です。Gaussianの画像が上下逆かどうかなど、画素内容の向きは判定しません。必要ならautoAxisCorrectionを切り、manualAxisCorrectionを指定します。

### 出力

`Assets/GeneratedLayoutsV2/placement_report_v2.json`と`window_report_v2.json`を出します。家具レポートには現在の部屋ローカル姿勢も含みます。背景の見た目の確認結果は自動統合せず、検証待ち状態を残します。

UnityコンポーネントはContextMenuまたはpublicメソッドで実行し、Start/Updateで自動実行しません。元のplacement_report.jsonは更新しません。

## 5. 完了前の確認

```bash
python pipeline.py assets --scene outputs/room_001/scene.json
```

これは期待GLB/PLYの存在検査です。メッシュの見た目や寸法まで保証するものではありません。

推奨順は、入力検証→家具生成・正規化→固定部屋→家具配置→背景の短い学習→PLYのUnity表示→品質調整→最終レポート保存です。

## 検証結果と残る条件

- Python：形式変換、欠落、重み、循環、矛盾、パス整合、設定継承、計画フラグの14テスト。
- C#：境界点、fit/cover分離、実行可能性優先など7つの純粋数値テスト。
- Unity 2022.3.62f3：一時プロジェクトで回転/移動した部屋、支持配置、再現性、欠落sourceの採点、窓内サイズ、屋外側距離、明示制約なしの無変更を確認。
- 画像生成・実アセットによるGPU学習は未実行。追加した05/12のBlender処理も、この環境では実行確認していません。
- TRELLIS本体・モデル重み・GLB/PLYは含めません。実際のTRELLIS呼び出し設定とGaussianのBounds入力は利用環境で必要です。
- 旧スクリプトの全挙動を置き換えていません。01/02/21/35などは互換・比較用です。V2の推奨経路では35による固定初期座標を使いません。

GitHubへ保存するのは実行コード・設定例・ドキュメント・テストです。元データの参照例はexamplesに保持し、生成アセットや認証情報は含めません。
