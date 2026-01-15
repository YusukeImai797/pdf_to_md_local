# データ設計書（Draft）
## 論文レビュー支援のための前処理パイプライン
**（PDF → 構造化Markdown＋sidecar＋監査可能な段落位置）**

Version: 0.1（Draft）  
Date: 2026-01-15

---

## 0. 出力物（ファイル単位）

- `paper.md`：復元本文（人間可読・AI入力の主）
- `paper.bundle.yaml`：sidecar（必須、リンクラベル・メタ・QCを含む）
- `paper.ann.jsonl`：リンクラベル（分割出力、機械可読・監査用）
- `paper.meta.json`：処理条件・座標系・バージョン等
- `paper.qc.json`：品質ゲート結果

> sidecar は `paper.bundle.yaml` を必須とする。大規模運用では JSONL への分割出力も許容する。

---

## 1. 参照の基本：段落と段落位置

### 1.1 段落（本文参照）
**最小要件**：mdの行番号で参照できること（安定していること）

- `span.md_line_start`（int, 1-index）
- `span.md_line_end`（int, inclusive）
  - sidecar内のフィールド名は `span` を用いるが、意味は「段落」参照とする

拡張（任意）：
- `span.char_start` / `span.char_end`（md全体の文字オフセット）
- `span.sentence_ids`（文分割を導入した場合）

### 1.2 段落位置（原典参照）
リンクラベルは必ずPDFへ遡れる（監査可能）こと。

- `pdf_ref[]`：複数可
  - `page`（int, 1-index）
  - `bbox`（float[4]）：`[x0,y0,x1,y1]`
  - `rot`（int, optional）：回転がある場合
  - `text_hash`（string, optional）：照合用

---

## 2. IDと参照関係

### 2.1 ID（推奨）
- `paper_id`：論文単位のID（例：`smith2023_psychsafety`）
- `ann_id`：リンクラベル単位のID（UUID推奨）
- `claim_id`：主張単位ID（例：`C_H1`）
- `rel_id`：関係エッジID（例：`REL_C1_C2_support`）
- `construct_id`：構成概念ID（例：`K_psych_safety`）
- `ref_id`：参考文献ID（例：`R_edmondson1999`）

### 2.2 参照（外部キー）
- リンクラベルは `id` で独立し、必要に応じて `attrs` 内のID参照で結合する。
- 参照整合性（存在チェック）をQCで検証する。

---

## 3. リンクラベルレコード（Annotation）共通スキーマ

全リンクラベルレコードの共通フィールド：

- `id`（string）
- `type`（enum）
- `span`（object）※必須
- `pdf_ref`（array）※必須
- `confidence`（float 0–1）
  - 推定リンクラベルは必須
  - 確定情報は 1.0 でも可
- `attrs`（object）※typeごとの属性
  - `candidate_roles`（array<string>, optional）：広く抽出した候補ラベル（5要点への整理用）

---

## 4. type別スキーマ

### 4.1 `heading`
- attrs:
  - `level`（int）：1..n
  - `title`（string）
  - `numbering`（string, optional）：`2.1`等

### 4.2 `section`
（Introduction/Methods等の“章種別推定”）
- attrs:
  - `label`（enum）：`introduction|theory|methods|results|discussion|conclusion|references|appendix|unknown`

### 4.3 `discourse`
（文意：ギャップ、RQ、仮説、結果、限界など）
- attrs:
  - `role`（enum）：
    `background|prior_work|gap|purpose|rq|hypothesis|method|result|interpretation|implication|limitation|contribution|other`
  - `signals`（array<string>, optional）：キーフレーズ等（監査用）
  - `reason`（object, optional）：
    - `status`：`supported|missing`
    - `evidence_spans`（array<object>）
    - `evidence_text`（array<string>）

### 4.4 `citation`
（本文内引用→参考文献対応）
- attrs:
  - `in_text`（string）
  - `style`（enum）：`author_year|numeric|mixed`
  - `ref_id`（string, optional）：references側のID（照合できた場合）
  - `ref_span`（object, optional）：referencesセクション内のmd行範囲
  - `citation_role`（enum, optional）：`support|contrast|definition|method_borrowing|boundary_setting|other`
  - `role_confidence`（float, optional）

### 4.5 `reference`
（参考文献エントリ）
- attrs:
  - `ref_id`（string）
  - `raw`（string）：エントリの生文字列
  - `parsed`（object, optional）：著者・年・タイトル等（後段でパースする場合）

### 4.6 `figure` / `table`
- attrs:
  - `label`（string）：`Figure 2`等
  - `caption_span`（object）
  - `body_span`（object, optional）※tableで本体がmdにある場合
  - `quality`（enum）：`high|medium|low`

### 4.7 `claim`
（主張単位：仮説、発見、解釈など）
- attrs:
  - `claim_id`（string）
  - `claim_type`（enum）：
    `hypothesis|finding|interpretation|theoretical_proposition|definition|implication|limitation|other`
  - `statement`（string）※原文抜粋でも正規化文でも可（spanが原典）
  - `constructs`（array<string>, optional）：関係する構成概念ID
  - `modality`（enum）：`assert|suggest|speculate`
  - `modality_confidence`（float）
  - `reason`（object, optional）：
    - `status`：`supported|missing`
    - `evidence_spans`（array<object>）
    - `evidence_text`（array<string>）

### 4.8 `relation`
（主張間リンク：支持・反駁・限定・因果など）
- attrs:
  - `rel_id`（string）
  - `source_claim_id`（string）
  - `target_claim_id`（string）
  - `relation_type`（enum）：
    `support|rebut|qualify|motivates|entails|defines|operationalizes|other`
- `evidence_spans`（array<object>）※どの段落が根拠か（監査用）

#### 4.8.1 主張趣旨（Causal Status）
`relation` が因果・関連の統合レビューに参加する場合、追加で保持する。

- `causal_status.kind`（enum）：`causal_claim|associational|interpretive|theoretical`
- `causal_status.identification`（enum or array）：
  `experimental|quasi_experimental|longitudinal|cross_sectional|qualitative_process|theory_only|unknown`
- `causal_status.modality`（enum）：`assert|suggest|speculate`
- `causal_status.confidence`（float）

---

## 5. ルール（バリデーション）

- すべてのリンクラベルは `span` と `pdf_ref` を必須（監査性）
- `citation.ref_id` があるなら、必ず `reference.ref_id` が存在する（参照整合性）
- `relation` は `source_claim_id` / `target_claim_id` が存在する
- `bbox`座標系は `paper.meta.json` の `coordinate_system` に明記

---

## 6. summaryブロック（5要点整理）

sidecarに `summary` を追加し、RQ/Hypothesis/Method/Result/Claim の整理結果を保持する。欠損は許容し、エラーとしない。

- `summary.rq` / `summary.hypothesis` / `summary.method` / `summary.result` / `summary.claim`
  - `primary_span`：中心段落（md行範囲）
  - `supporting_spans`（array<object>, optional）
  - `confidence`（float 0–1）
  - `reason.status`：`supported|missing`
  - `reason.evidence_spans`（array<object>）
  - `reason.evidence_text`（array<string>）

---

## 7. LLM出力JSON仕様（設計）

LLMは段落単位で候補を広く抽出し、5要点に束ねるための中間JSONを出力する。

必須フィールド:

- `paragraph_span`（md行範囲）
- `candidate_roles`（array<string>）
- `discourse`（object, optional）
  - `role` / `signals` / `confidence`
  - `reason.status` / `reason.evidence_text`
- `claim`（object, optional）
  - `claim_type` / `statement` / `modality` / `confidence`
  - `reason.status` / `reason.evidence_text`
- `causal_status`（object, optional）
  - `kind` / `modality` / `confidence`

ルール:
- 推測禁止（本文に根拠が無い場合は `reason.status: missing`）
- 欠損許容（role/claim/causal_status が空でも正常）

## 8. 運用メモ（推奨）

- `paper.ann.jsonl` は「1リンクラベル＝1行」で追記しやすく、差分管理に強い
- sidecar YAML は理解用途・小規模用途に向く（大規模では分割出力が推奨）
