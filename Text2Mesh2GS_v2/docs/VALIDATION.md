# 検証記録

## 確認した範囲

- Python 3.12.14：14件のunittest。形式変換、重複、欠落ID、不正重み・寸法、支持循環、方向矛盾、出力パス、Gaussian計画、設定継承。
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
