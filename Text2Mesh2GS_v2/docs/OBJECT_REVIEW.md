# 物体別レビュー版の実行手順

## この更新について

23は「候補検出 → 全体の見直し → 全体画像と切り抜きによる物体別確認」、34は「物体別の制約生成 → 画像との再照合 → 全体検証」を行います。既存のコマンド名・Python環境は変わりません。物体数に応じて推論回数が増えるため、従来より時間がかかります。

旧版のscene_graph.jsonは34で拒否します。植物19個を未確認のまま引き継がず、23から再実行してください。Ollamaを起動した端末は閉じず、別端末で実行します。

```bash
cd ~/Text2Mesh2GS_v2
source activate_wsl.sh
python scripts/23_build_scene_graph_from_text_image.py --scene_id room_001 --image_path outputs/scene_images/room_001/candidate_00.png
# review status: complete になった場合に次へ進む
python scripts/34_generate_spatial_constraints.py --scene_graph outputs/scene_graphs/room_001/scene_graph.json --output outputs/scene_graphs/room_001/spatial_constraints.json
# status: complete になった場合に変換
python pipeline.py prepare --scene outputs/scene_graphs/room_001/scene_graph.json --constraints outputs/scene_graphs/room_001/spatial_constraints.json --output outputs/room_001
```

## 23の確認待ちを解決する

終了コード2は確認待ちです。`scene_graph.json`の`generation.unresolved`に対象ID・理由を保存します。候補一覧は`generation.detections`、個別判断は`generation.decisions`です。未確認の物体を黙って落として完成扱いにはしません。

- `scene_graph.review/detection.json`：短い物体一覧とbboxの検出。
- `scene_graph.review/detection_review.json`：全体の見直し。
- `scene_graph.review/<ID>.json`：各物体の推論・修復ログ。
- `*.cache.json`：同一入力で成功した結果。入力画像・設定・プロンプト・スキーマが変わると再推論します。
- 同じ入力で再実行すると確認済み部分を再利用します。`needs_review`の応答は再利用しません。誤った成功結果を推論し直すには該当キャッシュだけを別名へ移してください。

物体は独立した室内物体、他の物体の一部・重複、室外の緑、確認待ちに区別します。窓越しの背景プレートはGaussianとして扱えます。重なりは確認候補を選ぶ条件であり、自動削除の条件ではありません。候補の上限64件に到達した場合も確認待ちです。

寸法は`estimated_dimensions_m.width/height/depth`で保存し、互換用`dimensions_m`は同じ値を幅・高さ・奥行きの順で保持します。単眼画像からの推定であり、Unityの実Boundsや測定値と照合してください。互換用のconfidence=0.5は未採点の仮値で、`confidence_kind=unscored_placeholder`を付けています。採否の基準には使いません。

### 棚・窓などの固定物体

画像だけでは部屋内の固定位置を確定できないため、検出された壁付け棚・窓は、確認済みTransformを指定するまで確認待ちになります。次はファイル形式の例で、位置・IDは実際の確認結果に置き換えます。

```json
{
  "shelf_01": {
    "position": [-2.8, 1.8, 0.0],
    "rotation_euler": [0, 0, 0],
    "scale": [1, 1, 1]
  }
}
```

これを`fixed_anchors.json`などに保存し、23へ`--anchors fixed_anchors.json`を追加します。単位はm、roomFrameローカル座標です。固定物体はmovable=falseとなり、棚上の植物はそのIDをsupport_idに持ちます。窓はcategory=window_frameです。

Unityでは固定物体をreferenceRoot下に置き、同じSceneObjectIdV2と確認済みTransformを設定してください。家具ソルバーは固定物体を移動しません。既存のReferenceRoomBuilderV2は背面窓1枚の簡易構造であり、複数窓や棚を自動構築するものではありません。

### 人が認識内容を訂正する場合

`generation.decisions`から対象IDのレコードをコピーし、画像を見て訂正したID→レコードのJSONを保存します。23に`--review-decisions reviewed_decisions.json`を追加してください。各レコードにはstatus、related_id、reason、representation、support_id、mounting、estimated_dimensions_m、dimensions_plausible、description、asset_promptが必要です。reasonには確認根拠を記載します。

statusはconfirmed / part_or_duplicate / outdoor_background / needs_reviewです。支持先が不明なままconfirmedに変更したり、未解決一覧だけを削除したりしないでください。参照先のIDが確認済みでない場合は引き続き停止します。人の訂正は`<ID>.manual.json`に記録されます。固定物体には別途anchorsも必要です。

## 34の確認待ちと再実行

各物体について、全体画像・物体の切り抜き・支持先の切り抜き（存在する場合）を渡します。sourceを当該IDに限定し、確認済みsupport_idがあれば対応するon関係を要求します。画像で観察した関係はbasis=observed、推定の設計意図はdesign_intentで区別します。

`spatial_constraints.objects/`に物体別の生成・画像レビュー・修復ログ、`.draft.json`に途中結果を保存します。未解決が残る最終JSONはstatus=incompleteで、終了コード2となります。prepareはこれを拒否します。

再実行では同じ入力の成功結果を再利用し、未解決項目を再問い合わせします。先行する制約が変われば、依存する後続の入力も変わるため再確認します。ルールによってすべての植物を床へ配置する補完は行いません。物体一覧を修正した場合は34を再実行してください。シーンのハッシュが異なる古い制約はprepareで拒否します。

## 検証範囲

今回の自動テストは、19個の植物への個別要求、未解決時の出力拒否、キャッシュ、固定支持面、シーン変更の検出などを確認します。実モデルでは提供画像のソファ1個について個別生成・画像レビューが完了し、制約2件を生成しました。全家具の再認識から実アセット生成・Unity表示までの一括検証は未実施です。同じVLMによる再照合は誤認識を完全には排除しません。
