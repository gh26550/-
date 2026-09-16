# 導入・実行手順

## 1. GitHubから取得

```bash
git clone https://github.com/gh26550/-.git Text2Mesh2GS-delivery
cd Text2Mesh2GS-delivery/Text2Mesh2GS_v2
```

以降はこのフォルダを作業ディレクトリにします。パスはここからの相対パスです。Pythonの軽い確認はWindowsでも可能です。画像生成・学習は既存のWSL/CUDA環境を利用することを推奨します。

## 2. モデル不要の動作確認

Python 3.10以上で実行します。

```bash
python pipeline.py prepare --scene examples/scene_v2.json --output outputs/room_001
python pipeline.py validate --scene outputs/room_001/scene.json
python -m unittest discover -s tests -v
```

サンプルは14物体・18制約です。`outputs/room_001`にUnity用JSONや生成ジョブが作成されます。この段階では家具モデルやGaussianは生成されません。

独自の旧形式JSONは次のように変換します。

```bash
python pipeline.py prepare --scene YOUR_LAYOUT.json --constraints YOUR_CONSTRAINTS.json --output outputs/my_scene
```

## 3. 家具と背景を用意

既存アセットがある場合は新規生成を省略できます。生成する場合の詳細コマンドは[READMEの工程2・3](../README.md)を参照してください。

| 順序 | 操作 | 次へ進む条件 |
|---|---|---|
| 1 | 22で部屋画像、23でscene graph、34で制約の初期案を作る | 物体IDと制約を確認 |
| 2 | prepareで共通形式とジョブを作る | validate成功 |
| 3 | 25→28→29で家具画像とTRELLISジョブを作る | 画像候補を確認 |
| 4 | 30で利用中のTRELLIS実行器を呼ぶ | 外部コマンド設定済み、GLB生成成功 |
| 5 | 必要に応じ04→10→12でGLBを整理・寸法調整 | 原点・大きさ・正面を確認 |
| 6 | 26→31で背景画像、03で背景板GLBを作る | 選んだ背景GLBが存在 |
| 7 | scene.jsonのasset.source_pathを合わせprepareを再実行 | GaussianジョブがGLBを参照 |
| 8 | pipeline gaussianで計画を確認し、--executeで実行 | 05→07_5→08→09が完了 |

TRELLIS本体は同梱しません。`configs/trellis.command.example.json`の説明用パスを実環境のコマンドへ変更した設定が必要です。画像生成用とGaussian学習用のrequirementsは依存候補であり、GPU環境ごとのバージョン調整が必要です。

生成アセットは`outputs/mesh_assets/<scene_id>/<object_id>/<object_id>.glb`と`outputs/gaussian_assets/<scene_id>/<object_id>/<object_id>.ply`に揃えます。

```bash
python pipeline.py assets --scene outputs/room_001/scene.json
```

Blender工程は独立した`blender --background`で実行します。処理中にシーンをクリアするため、編集中のBlenderシーン上で実行しないでください。

## 4. Unityに追加

1. `unity/`内のC#を、新しい`Assets/Text2Mesh2GS_V2/`へコピーします。
2. prepareで生成した`layout_for_unity.json`と`spatial_constraints.json`をAssets内へコピーします。
3. 既存のGLBインポーターとGaussian表示機能でアセットをインポートし、シーンへ置きます。このパッケージ自体はインポーターではありません。
4. 空オブジェクト`roomFrame`を作り、床の中心を原点、ワールドscaleを(1,1,1)にします。+Xが右、+Yが上、-Zが手前です。
5. 家具の親`movableRoot`と、壁など固定構造の親`referenceRoot`を用意します。家具ルート同士は入れ子にしません。
6. 各対象に`SceneObjectIdV2`を追加し、JSONのidと一致させます。モデルの正面が+Z以外ならforwardAxisを指定します。
7. `SemanticScenePlacerV2`を追加し、2つのJSON・roomFrame・2つのルートをInspectorで指定します。
8. コンポーネントのメニューから`Apply Semantic Layout V2`を実行します。続いて`Audit Current Layout V2`と`Export Placement Report V2`を実行します。

固定構造がない場合は`ReferenceRoomBuilderV2`で矩形の部屋と背面窓を作れます。生成された固定構造の親をreferenceRootとして指定してください。窓の寸法・位置はInspectorで設定します。既存の配置スクリプトが同じ家具を動かしている場合は、その自動実行を無効にします。

## 5. 窓背景を調整

1. `WindowBackgroundPlacerV2`へGaussian、窓基準windowBasis、制約JSONを指定します。
2. windowBasisは単位scale、+Zを屋外にします。窓の4辺または明示的な開口Boundsを指定します。
3. Rendererから寸法を取得できないGaussianには`PlacementBounds.localBounds`を実アセットに合わせて設定します。
4. fitInside（内側に収める）とcover（全面を覆う）の目的を選び、Apply/Audit/Exportを実行します。
5. 実際に使うカメラからPLYの見た目を確認します。`visible through`は自動で成功扱いになりません。

## 6. レポートを読む

`Assets/GeneratedLayoutsV2/`へ家具と窓背景のJSONレポートを保存します。

- score_100：意味制約の重み付き達成度。
- coverage_100：期待する制約のうち評価できた割合。
- physical_violations：物理違反の件数。
- constraints：どの関係がok/unmet/missingになったか。
- objects：家具の最終姿勢。

家具のstatusがsuccessになるには、意味制約の全達成と物理判定の両方が必要です。点数だけで完了を判定しないでください。

## よくある問題

| 症状 | 確認箇所 |
|---|---|
| missing_source / missing_target | ID、対象ルート、Boundsの取得可否 |
| roomFrameのscaleエラー | 親を含めワールドscaleが1か |
| 家具の大きさが不自然 | 12での正規化とapplyJsonScaleの二重適用 |
| againstが不自然 | 壁参照のforwardが壁の法線方向か |
| Gaussianジョブが準備未完了 | source_pathのGLBが存在するか、prepareを再実行したか |
| scoreが高いのにincomplete | 物理違反、未達の制約、欠落の有無 |
| fitInsideでcoverがunmet | 余白・縦横比による目的の衝突。設定と制約を見直す |
| TRELLISが動かない | サンプルの架空パスを実際の実行器へ変更したか |
