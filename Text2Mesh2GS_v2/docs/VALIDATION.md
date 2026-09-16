# 物体別レビュー更新の検証（2026-09-16）

Python回帰テスト35件成功。19物体の個別要求、未解決時の出力拒否、固定支持面、入力ハッシュ、キャッシュを検証。実画像のソファ1個ではローカルQwen2.5-VLによる生成・画像レビューが完了し、制約2件を保存。全物体とUnityでの最終表示は未検証。以下は過去の版の記録です。

# 検証記録

## 画像端と重複検出の修正

- ユーザー実行ログでは1024px幅の画像にx=1035のbboxと、同じ植物の繰り返しが含まれていました。
- 2%以内の画像端クリップと、完全一致の別ID検出統合を監査付きで追加しました。
- 当該失敗応答は24検出から12対象に整理され、画像範囲・ID・形式の検証に成功しました。
- 回帰テスト29件が成功。大きな超過・同一ID重複は拒否し、元出力を変更しないことも確認しました。

## ローカル画像認識版の検証（2026-09-16）

- Python 3.10.20の専用WSL環境：27件のテスト成功、pip check成功。
- Ollama 0.34.1 + Qwen2.5-VL 7B (Q4_K_M)、RTX 5070 Tiで実画像推論を実行。
- 元プロジェクトのroom_001/candidate_00.png（1344×768）で23の認識・レビュー→34の制約生成→prepareまで成功。
- 認識結果：椅子2脚、ソファ、テーブル、ラグ、植物2つ、窓の8対象。設定された部屋の固定構造5件を加えて13物体、生成制約7件。
- 画像内の本棚などの検出漏れは残りました。これは検出精度100%の主張ではありません。実アセット生成前に画像と認識一覧を照合してください。
- 追加テスト：複数インスタンス、重複bbox、画像寸法での座標変換、根拠の保持、明示Meshの保持、画像不一致、欠落画像、ローカル画像リクエスト、スキーマ制限、有限修復、失敗時停止、CSV出力。
- 初回検証で座標単位不一致・重複列挙・存在しないIDの制約が見つかったため、ピクセル契約・重複検証・実在IDに制限した生成スキーマを追加しました。

認識出力はoutputsに保存し、モデル重みと部屋画像をGitへ追加していません。初期画像生成、TRELLIS、実アセットでのGaussian学習・最終Unity見た目は今回実行していません。

## 確認した範囲

- 既存のPython回帰テスト14件：形式変換、重複、欠落ID、不正重み・寸法、支持循環、方向矛盾、出力パス、Gaussian計画、設定継承。上記27件に含まれます。
- C#：7件の数値チェック。満足度の境界、fit/cover、候補比較。
- Unity 2022.3.62f3：使い捨てテストプロジェクトで`PlacementV2Smoke.Run`を実行し、`V2_SMOKE_PASS`を確認。
- Unity確認内容：移動・回転した部屋、支持関係、決定的な再配置、欠落制約の採点、背景の窓内サイズと屋外側配置、制約なしでの無変更、左右壁の分離。

## 再実行

```bash
python -m compileall -q pipeline.py scripts tests
python -m unittest discover -s tests -v
python pipeline.py prepare --scene examples/scene_v2.json --output outputs/check
python pipeline.py validate --scene outputs/check/scene.json
```

Unityテストは独立した空プロジェクトのAssetsへunityのC#を、Assets/Editorへtests/PlacementV2Smoke.csをコピーして実行します。JsonSerializeとPhysicsの組み込みモジュールが必要です。既存の制作シーンを使わないでください。

```bash
Unity -batchmode -nographics -projectPath /absolute/path/to/test_project -executeMethod PlacementV2Smoke.Run -logFile smoke.log
```

Unityプロセス完了後、smoke.logの`V2_SMOKE_PASS`を確認します。実行ファイル名・パスは環境に合わせて変更してください。

## 未検証・利用環境で必要な確認

画像生成モデルの実推論、TRELLIS実行器、Blenderの05/12、実アセットによるGPU学習とUnityでのPLY表示は未検証です。コードのテスト成功は、これらの環境での動作保証ではありません。依存バージョン、モデル重み、アセット、外部コマンド設定は別途用意してください。

生成アセット、checkpoint、UnityのLibraryと一時プロジェクトはGitHubへ含めません。

## 2026-09-16 支持先修復の実データ検証
Python回帰テスト42件成功。DPC7のroom_001で34 --estimate-fixedを実行し、complete・未解決0件、制約17件。家具5件の支持先はfloor_01、窓枠2件は固定推定配置。続くprepareもvalid=true（物体12件、部屋アンカー5件を含む）。元scene_graphは日時付きバックアップに保存。検出漏れ・全制約の視覚的正しさ・実アセットでのUnity配置の成功を保証する検証ではありません。
