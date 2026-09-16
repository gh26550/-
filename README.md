# Text2Mesh2GS・ローカル画像認識とUnity自動空間配置

生成した部屋画像をローカルQwen2.5-VLで認識し、家具抽出・レビュー・LLM制約生成から、アセット生成とUnity配置へつなげるスクリプト集です。外部APIキーは不要です。

- [導入・実行手順](Text2Mesh2GS_v2/docs/USAGE.md)
- [動作原理・スコアの説明](Text2Mesh2GS_v2/docs/ARCHITECTURE.md)
- [全工程のコマンドと設定](Text2Mesh2GS_v2/README.md)
- [検証範囲](Text2Mesh2GS_v2/docs/VALIDATION.md)
- [変更点](Text2Mesh2GS_v2/CHANGELOG.md)

実行ファイルは`Text2Mesh2GS_v2/`にまとめています。生成アセットとモデル重みは含みません。TRELLISの外部実行コマンド、GPU環境、Unityへのアセット導入は利用環境で設定してください。
