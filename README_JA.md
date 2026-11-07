<div align="center">

<img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/header.png" alt="" />

<h1>SoMark｜O<span style="color: #DE7802;">X</span>R</h1>

ポストトレーニングで RSI ループを採用し、SOTA の精度と速度を実現したオープンソースのドキュメント解析モデル。

---

**ドキュメントを Agent-Friendly なデータへ。**

[🌐 オンラインデモ](https://free-oxr.somark.ai) ｜ [🔥 🆓 無料 API](#free-api)

[![GitHub | OXR](https://img.shields.io/badge/GitHub-OXR-181717?style=flat-square&logo=github)](https://github.com/SoMarkAI/OXR)   [![Hugging Face | OXR-1.0](https://img.shields.io/badge/Hugging%20Face-OXR--1.0-F59E0B?style=flat-square)](https://huggingface.co/SoMarkAI/OXR-1.0)   [![ModelScope | OXR-1.0](https://img.shields.io/badge/ModelScope-OXR--1.0-624AFF?style=flat-square)](https://modelscope.cn/models/SoMark/OXR-1.0)
![arXiv | Coming Soon](https://img.shields.io/badge/arXiv-Coming%20Soon-B31B1B?style=flat-square)

[English](README.md) · [简体中文](README_CN.md) · **日本語**

</div>

OXR（Optical Everything Recognition）は、SoMark チームが開発したオープンソースモデルです。PDF やドキュメント画像を、レイアウト座標付きの Markdown または JSON 形式の Agent-Friendly なデータに変換します。

OXR はスクラッチから事前学習されており、ポストトレーニングにおいてドキュメント解析分野で初めて再帰的自己改善（RSI: Recursive Self-Improvement）ループを採用しました。わずか 1.3B のパラメータで、olmOCR-Bench、OmniDocBench、DocParsingBench の 3 つのベンチマークで 1 位を獲得し、業界トップクラスの解析速度を実現しています。

OXR は、レイアウト解析モジュールを独立・プラガブルにした初めてのモデルです。この設計により解析が高速化されるとともに、内蔵のレイアウトモデルを独自のモデルに置き換えることが可能になります。その場合でも、コンテンツ認識と読み順の処理は引き続き OXR が担うため、最小限の統合作業で多様なシナリオに適応できます。

![](https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/three-benchmark-leaderboards-olm-main.png)

![](https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/speed-score-overview.png)

---

> このプロジェクトが役に立ったら、右上の ⭐ Star をクリックして応援してください。開発者にとって何よりの励みになります！



## なぜ OXR なのか？

- 🎯 SOTA の精度。3 つのベンチマークで 1 位。

  - OXR は olmOCR-Bench、OmniDocBench、DocParsingBench で 🏆1 位を獲得しており、特に表の解析では大きな差をつけています。
  - 数式や表から化学反応式、化学構造式、コードブロックまで、13 種類のドキュメント要素をバウンディングボックス座標付きで認識できます。A3 の試験問題、新聞、縦書き、手書き文字など、難易度の高いドキュメントやレイアウトにも強みを発揮します。
  - 詳細な結果と評価の再現手順は [docs/benchmark.md](docs/benchmark.md) を参照してください。
- 🚀 1.3B パラメータ。トップクラスの解析速度。

  - わずか 1.3B のパラメータで、RTX PRO 6000 での速度テストにおいてオープンソースのドキュメント解析モデルの中で 🏆1 位を獲得しています。
  - タスクレベルで処理アーキテクチャを再設計することで、精度を維持しながら解析速度を向上させています。
  - 詳細な結果と評価の再現手順は [docs/benchmark.md](docs/benchmark.md) を参照してください。
- 🧩 プラガブルなレイアウト解析。ドキュメント構造を思いどおりに。

  - レイアウト解析の要件は分野によって大きく異なります。同じドキュメントであっても、ビジネスのユースケースによって要素の種類、領域の境界、グルーピングのルールが変わることがあります。
  - OXR はレイアウト解析を独立・プラガブルにした初めてのモデルです。OXR の認識機能と読み順の機能を維持したまま内蔵のレイアウトモデルを置き換えられるため、最小限の統合作業でさまざまなドキュメントシナリオに適応できます。
  - 詳細は下記の[仕組み](#how-to-works)セクションを参照してください。


<details>
<summary>展開して 13 種類の要素をすべて表示</summary>

| 名称 | JSON の type 値 | 内容または出力形式 |
|-|-|-|
| キャプション | Caption | プレーンテキスト |
| 脚注 | Footnote | プレーンテキスト |
| 数式 | Formula | LaTeX |
| テキスト | Text | プレーンテキスト |
| フッター | Page-footer | プレーンテキスト（デフォルトでは除外） |
| ヘッダー | Page-header | プレーンテキスト（デフォルトでは除外） |
| 図 | Picture | 画像 |
| タイトル | Title | プレーンテキスト |
| 表 | Table | HTML / Markdown |
| 印影 | Stamp | プレーンテキスト / 画像 |
| 化学反応式 | Chemical Equation | LaTeX（\ce で始まる mhchem 構文） |
| コードブロック | Code Block | Markdown コードブロック |
| 化学構造式 | Chemical Structure | SMILES / 画像 |


</details>


## 解析例

以下のアニメーションは OXR の解析プロセスを示しています。

![OXR の解析例のアニメーション](https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/oxr_parsing_demo.gif)

以下は、OXR がさまざまな種類のドキュメントを解析した例です。画像をクリックすると元のサイズで表示されます。

<table>
  <tr>
    <td align="center" valign="top"><strong>A3 試験問題</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/a3_exam_paper.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/a3_exam_paper.png" alt="A3 試験問題" width="280" /></a></td>
    <td align="center" valign="top"><strong>中国語の手書き文字</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/chinese_handwriting.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/chinese_handwriting.png" alt="中国語の手書き文字" width="280" /></a></td>
    <td align="center" valign="top"><strong>コードブロック</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/code_block.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/code_block.png" alt="コードブロック" width="280" /></a></td>
  </tr>
  <tr>
    <td align="center" valign="top"><strong>化学反応式</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/chemical_equation.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/chemical_equation.png" alt="化学反応式" width="280" /></a></td>
    <td align="center" valign="top"><strong>化学構造式</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/chemical_structure.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/chemical_structure.png" alt="化学構造式" width="280" /></a></td>
    <td align="center" valign="top"><strong>複雑な数式</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/complex_formula.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/complex_formula.png" alt="複雑な数式" width="280" /></a></td>
  </tr>
  <tr>
    <td align="center" valign="top"><strong>複雑な表</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/complex_table.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/complex_table.png" alt="複雑な表" width="280" /></a></td>
    <td align="center" valign="top"><strong>学術論文</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/academic_paper.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/academic_paper.png" alt="学術論文" width="280" /></a></td>
    <td align="center" valign="top"><strong>新聞</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/newspaper.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/newspaper.png" alt="新聞" width="280" /></a></td>
  </tr>
  <tr>
    <td align="center" valign="top"><strong>調査レポート</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/research_report.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/research_report.png" alt="調査レポート" width="280" /></a></td>
    <td align="center" valign="top"><strong>履歴書</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/resume.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/resume.png" alt="履歴書" width="280" /></a></td>
    <td align="center" valign="top"><strong>英語の手書き文字</strong><br/><a href="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/english_handwriting.png"><img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/parsing_examples/english_handwriting.png" alt="英語の手書き文字" width="280" /></a></td>
  </tr>
</table>

## 仕組み

<a id="how-to-works"></a>

### 技術アーキテクチャ

このシステムは、レイアウト解析モデルと OXR の 2 つのモデルで構成されています。読み順（Reading Order）の推定を認識（Recognition）ステージに統合し、両タスクを同一のモデルで処理しています。この設計により、速度と精度の両方が向上しました。

また、このような設計にしたのはカスタマイズを容易にするためでもあります。私たちの経験上、レイアウト解析はドキュメントの分野によるばらつきが他のどのモジュールよりも大きく、ドキュメント解析を提供する事業者が最もカスタマイズを必要とする部分です。既存のドキュメント解析ツールは、レイアウト解析を認識モデルに担わせるか、読み順も処理する専用モデルを使用するかのいずれかです。OXR の設計はレイアウト解析を最もシンプルに組み込める方式を実現しており、どのようなレイアウト解析モデルでも OXR モデルとうまく連携できます。

![](https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/oxr_pipeline.png)



### RSI ループによる改善

ドキュメント解析のように正解が存在するタスクにおいて、私たちは長い間、コアモデルが自らを改善するのを支援するエンジニアリングシステムを求めてきました。下図は OXR の RSI アプローチを示しています。モデルが誤った、あるいは苦手としているサンプルを対象とし、人手によるアノテーションなしで性能を向上させます。これにより、これまでにない速さでの反復開発が可能になりました。

![](https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/oxr_rsi_loop.png)



## クイックスタート

<a id="free-api"></a>

### 0. 🔥 🆓 無料 API

OXR は無料の API を提供しており、ユーザー登録や API キーの申請なしで、お手持ちのドキュメントの解析を試すことができます。

運用コストを負担できなくなるまで、無料 API は公開し続ける予定です。

⚠️ 注意：無料の OXR API を本番環境で使用することは推奨しません。サービスの安定性は保証されず、リクエストはキューで処理されるため、処理速度が多少変動する場合があります。

| オンラインデモ | https://free-oxr.somark.ai | https://free-oxr.somark.cn |
|-|-|-|
| API URL | https://free-oxr.somark.ai/api/v1 | https://free-oxr.somark.cn/api/v1 |
| API ドキュメント | [https://docs.somark.ai/api-reference](https://docs.somark.ai/api-reference) | https://docs.somark.cn/api-reference |

> somark.cn は somark.ai のミラーです。お住まいの地域で最も快適に利用できるドメインを選択してください。

OXR のドキュメント解析 API は、SoMark 公式のドキュメント解析 API と互換性があります。**Base URL** を変更するだけで、SoMark 公式のドキュメント解析 CLI ツールである somark-py および somark-js を利用できます。対応するエンドポイントと出力形式については 1.4 節を参照してください。

```Bash
# using python
pip install somark
# or using js
npm install -g somark-js

somark --base-url https://free-oxr.somark.ai/api/v1 --api-key dummy \
  parse document.pdf --formats md,json --out ./parsed-output/
```



### 1. ローカルで実行する

```Plain Text
OS: Linux
GPU: NVIDIA（BF16 対応）
ドライバー: NVIDIA ドライバー（CUDA 13.0 互換）
Python: 3.12（モデルのダウンロード / ソースからの実行用）
最小 GPU メモリ: 単一 GPU で 24GB（低同時実行数でのテスト用）
推奨 GPU メモリ: 単一 GPU で 48GB 以上（長いドキュメント / 高同時実行数向け）
```

#### 1.1 コードとモデルのダウンロード

| モデル | ダウンロードリンク | 精度 | 軽量レイアウト解析モデルを含むか |
|-|-|-|-|
| OXR-1.0 | [🤗 Hugging Face ](https://huggingface.co/SoMarkAI/OXR-1.0)  <br/>[🤖 ModelScope](https://modelscope.cn/models/SoMark/OXR-1.0) | BF16 | はい（layout_analysis フォルダ内） |

まず、コードを取得し、モデルをダウンロードするための環境を準備します。

```Bash
git clone https://github.com/SoMarkAI/OXR.git
cd OXR

# pip install -U huggingface_hub
hf download SoMarkAI/OXR-1.0 --local-dir ./models/OXR-1.0

# or using ModelScope
# pip install -U modelscope-hub
modelscope download SoMark/OXR-1.0 --local_dir ./models/OXR-1.0
```

#### 1.2 サービスの起動

##### 1.2.1 Docker で実行する（推奨）

すべてのサービスをまとめて設定・起動できる Docker の利用を推奨します。

```Bash
docker build -f docker/Dockerfile.allinone -t oxr-1.0 .
docker run --rm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/models/OXR-1.0:/models/oxr-model:ro" \
  oxr-1.0
```

起動ログにサービスの準備完了が表示されたら、http://localhost:8000/demo または http://localhost:8000/docs を開いてください。
初回の起動には時間がかかる場合があります。デフォルトの起動タイムアウトは 900 秒です。
より高度なオプションや使用例については「How to use OXR」を参照してください。


##### 1.2.2 ソースから実行する

コードとモデルをダウンロードしたら、以下の 3 つのサービスをそれぞれ起動します。各サービスは独立したプロセスと仮想環境を使用します。すべてのコマンドは OXR リポジトリのルートで実行してください。ドキュメント解析 API は OXR モデルサービスとレイアウト解析モデルサービスの両方に依存するため、先に 2 つのモデル API を起動してください。

**OXR モデルサービス（ポート 8080）**

OXR は vLLM に対応していますが、vLLM の公式リリースはまだ OXR の推論をサポートしていません。そのため、Hugging Face 上のモデルウェイトフォルダに out-of-tree プラグインと読み込み用スクリプトを同梱しています。再コンパイルは不要で、公式の vLLM Docker イメージや CLI でそのままプラグインを使用できます。

```Bash
# pip install vllm==0.25.1 transformers==5.13.1

MODEL_PATH="$PWD/models/OXR-1.0" \
  bash services/oxr/start_vllm.sh \
  --gpu-memory-utilization 0.70 --max-num-seqs 16
```

**レイアウト解析モデルサービス（ポート 8081）**

レイアウト解析モデルのウェイトは、先ほどダウンロードした OXR モデルのウェイトフォルダに含まれています。

```Bash
# pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
# pip install -r services/layout_analyze/requirements.txt

MODEL_PATH="$PWD/models/OXR-1.0/layout_analysis/last.pth" \
DFINE_DEVICE=cuda \
  python -m uvicorn dfine_la.api:app \
  --app-dir services/layout_analyze \
  --host 127.0.0.1 --port 8081 --workers 1
```

**ドキュメント解析 API（ポート 8000）**

同梱の `config.example.yaml` を `config.yaml` としてコピーします。`layout_analyze_model` と `oxr_model` の下にある url フィールドを、上で起動した 2 つのサービスのアドレスに設定してください。

```Bash
# pip install -c docker/requirements.allinone.txt .

oxr serve --config config.yaml \
  --host 127.0.0.1 --port 8000 --workers 1
```

3 つのサービスがすべて起動したら、Docker の場合と同じようにデモと解析 API を利用できます。1.3 節と 1.4 節を参照してください。

#### 1.3 ビジュアルデモを開く

http://localhost:8000/demo を開き、PDF または画像をアップロードします。元のファイル、Markdown のプレビュー、ソース、レイアウトのバウンディングボックス、読み順を確認でき、Markdown をダウンロードすることもできます。

デモでは一時ファイルキャッシュを使用しており、デフォルトでは 1 時間で期限切れになります（処理中は更新されます）。キャッシュの期限が切れるかクリアされると結果にアクセスできなくなるため、早めにダウンロードしてください。

#### 1.4 解析リクエストを送信する

OXR の API は SoMark API と互換性があります。OXR は一部の特殊な機能には対応していませんが、SoMark のエコシステムはデプロイ済みの OXR サービスを完全にサポートしています。[SoMark の Web サイト](https://somark.ai)にアクセスし、[API ドキュメント](https://docs.somark.ai/api-reference)を参照してください。

解析エンドポイントを直接呼び出す場合：

```Bash
curl --fail-with-body http://localhost:8000/v1/parse/sync \
  -F 'file=@document.pdf' \
  -F 'api_key=dummy' \
  -F 'output_formats=markdown' \
  -F 'output_formats=json'
```

SoMark は、Python と JavaScript 向けにドキュメント解析用の CLI および SDK ツール一式を提供しています。OXR はこれらのドキュメント解析インターフェースと互換性があるため、baseURL を変更するだけで利用できます。詳しくは https://docs.somark.ai/en/cli-sdk を参照してください。

```Bash
pip install somark

somark --base-url http://localhost:8000/v1 --api-key dummy \
  parse document.pdf --formats md,json --out ./parsed-output/
```

## エコシステム

SoMark チームは、ドキュメント解析のためのさまざまなツールを開発しています。役に立ったら ⭐ Star をお願いします。

|  | 説明 | プロジェクト URL |
|-|-|-|
| SoMarkDown | SoMark および OXR 公式の Markdown 解析・レンダリングライブラリ | https://github.com/SoMarkAI/SoMarkDown |
| SoMarkDownViewer | SoMarkDown のライブ編集アプリケーション | https://github.com/SoMarkAI/SoMarkDownViewer |
| SoPDF | 寛容なオープンソースライセンスの高性能 PDF 処理ライブラリ | https://github.com/SoMarkAI/SoPDF |
| Docision | MultiModal Jev for Document. | https://github.com/SoMarkAI/docision |
| somark-py | SoMark および OXR 公式の Python SDK & CLI | [https://github.com/SoMarkAI/somark-py](https://github.com/SoMarkAI/somark-py) |
| somark-js | SoMark および OXR 公式の JS / TS SDK & CLI | https://github.com/SoMarkAI/somark-js |

---

<div align="center">
<img src="https://raw.githubusercontent.com/SoMarkAI/image_bed/main/OXR_README/assets/OXR_Star_on_GitHub.gif" width="800" />
</div>

## 引用

現時点では、OXR-1.0 オープンソースプロジェクトを引用してください。技術レポートの公開後に論文の引用情報を追加します。

```bibtex
@misc{somark2026oxr,
  author = {{SoMark}},
  title = {{OXR-1.0}},
  url = {https://github.com/SoMarkAI/OXR},
  note = {GitHub repository}
}
```

## ライセンス

OXR は、Apache 2.0 をベースに追加条項を加えた独自ライセンスである [OXR Open Source License](./LICENSE) の下で公開されています。

その他のサービスについては、[somark.ai](http://somark.ai) から SoMark チームにお問い合わせください。

## コミュニティ

OXR コミュニティに参加して、ドキュメント解析について議論したり、フィードバックを共有したり、最新情報を入手したりしましょう。

- [Discord](https://discord.gg/msCPj2AtNN) — OXR について議論し、ノウハウを交換し、開発中のものを共有できます。
- [X / Twitter](https://x.com/somarkai) — OXR のリリース情報や SoMark のニュースをお届けします。
- [GitHub Issues](https://github.com/SoMarkAI/OXR/issues) — バグ報告、機能提案、進捗の確認ができます。
- **WeChat** — 下の QR コードをスキャンして OXR グループに参加し、他のユーザーと交流しましょう。

<p>
  <img src="./assets/wechat-community.png" alt="OXR WeChat グループの QR コード" width="200" />
</p>
