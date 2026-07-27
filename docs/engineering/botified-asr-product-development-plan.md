# Botified ASR 产品开发计划
> 目标仓库：`botified-asr`
> 研究基线：2026-07-26
> 上游基线：FunASR `1.3.29`（`8a34247d`）、SenseVoiceSmall、FSMN-VAD、CAM++

## 1. 文档定位

本计划用于一次性交付 Botified 生态的独立 ASR 服务。开发团队应完成服务、长音频处理、已知人物注册和命名、OpenAI 兼容接口、在线安装、公开发布以及 Agent Skill 的完整闭环。

该服务是 Botified 的周边服务，不属于 Botified Core 或 Gateway：

```text
Matrix / other channel
        |
Botified Claw Gateway
        |
POST /v1/audio/transcriptions
        |
botified-asr
        |
FunASR + SenseVoice + FSMN-VAD + CAM++
```

本计划不要求修改 Botified、OpenClaw、第三方 channel plugin、FunASR 或 SenseVoice。服务通过稳定 HTTP 契约独立演进。

## 2. 产品结论

开发团队必须交付以下完整产品：

1. 一个可独立部署、兼容 OpenAI Transcriptions API 明确子集的 ASR 服务。
2. 基于 SenseVoiceSmall 的多语言转写、情感和音频事件识别。
3. 可逐请求选择的 VAD 和说话人 pipeline。
4. 支持最大 1 GiB、最长 12 小时的会议音频；这是首版发布验证过的硬上限，部署配置只能收紧。
5. 已知人物的声音样本注册、更新、删除和命名转写。
6. 普通同步请求和同一端点上的可靠异步长任务。
7. 一键在线安装脚本，并在 `lzjever/botified-releases` 公开。
8. 一份可单独安装、同时兼容 Codex、OpenClaw 和 Botified 的 Agent Skill。

产品不承诺：

- 实时通话或实时麦克风字幕；
- TTS；
- 翻译；
- 声纹训练或模型微调；
- 仅凭姓名、描述推断人物身份；
- 对重叠语音做到无误分离；
- 把余弦相似度表述成身份概率；
- 多租户账号、计费或管理后台。

## 3. 核心工程约束

### 3.1 KISS

- 使用一个 Python 服务进程和一个 SQLite 数据库。
- 不引入 Redis、Celery、Kafka、对象存储或外部调度器。
- 使用一个公开转写端点，不增加 `large-transcribe` 等第二套业务 API。
- 同步和异步请求必须复用同一个 canonical transcription processor。
- 使用固定模型 alias，不允许客户端提交任意 ModelScope/Hugging Face 模型 ID。

### 3.2 DRY

- 音频解码、VAD、SenseVoice、speaker embedding、结果投影各只有一个实现。
- HTTP 同步处理和后台 job 不得复制 pipeline。
- Skill、README 和 OpenAPI 的请求示例从同一公开契约维护，不各自定义字段。
- `botified-asr` 是 Skill 的唯一源码仓库；`botified-releases` 只发布构建产物。

### 3.3 YAGNI

- 首版只支持当前明确需要的模型和 pipeline。
- 不实现模型市场、插件系统、工作流编排或任意 pipeline DAG。
- 不提供没有可靠底层语义的“关闭 SenseVoice 原生标点”开关。
- 不为将来可能存在的多租户提前增加 organization、project、role 等资源。
- 不实现断点续传；1 GiB 单次上传是首版明确边界。

### 3.4 一个功能一种做法

| 功能 | Canonical 做法 |
|---|---|
| 普通转写 | `POST /v1/audio/transcriptions` |
| 长音频转写 | 同一端点 + `Prefer: respond-async` |
| VAD | OpenAI `chunking_strategy` |
| 说话人分离 | `model=sensevoice-diarize` + `chunking_strategy=auto` + `response_format=diarized_json` |
| 已知人物 | `/v1/speakers` 注册，转写时提交 `known_speaker_ids[]` |
| 情感/事件 | namespaced `include[]` |
| API 鉴权 | Bearer Token |
| 非敏感配置 | 单一 YAML 配置文件 |
| API secret | 单一环境变量 `BOTIFIED_ASR_API_KEY` |
| 安装服务 | `install-asr.sh` |
| 安装 Skill | `install-asr-skill.sh` |

## 4. 上游能力和采用边界

### 4.1 采用

| 能力 | 上游组件 |
|---|---|
| ASR、语言、情感、事件、原生标点 | SenseVoiceSmall |
| VAD | FSMN-VAD streaming inference |
| Speaker embedding | CAM++ |
| Speaker embedding/window | CAM++ + FunASR `sv_chunk` 的固定 window 规则 |
| 模型加载和推理 | 三个固定的 FunASR model adapter |

### 4.2 不直接采用

不直接把当前 `funasr-server` 暴露为产品服务，原因包括：

- 当前实现会整体读取上传；
- 没有产品要求的鉴权和应用级限制；
- 动态 speaker 参数未形成完整稳定契约；
- 富标签被清除而没有结构化输出；
- response format 和 OpenAI error envelope 不完整；
- 长文件没有可恢复 job 语义；
- 已知人物数据库没有产品化接口。

实现应依赖上游公开模型能力，但由本仓库拥有 HTTP、持久化、长文件和响应契约。

首版明确不使用 FunASR `ClusterBackend`：它需要全量 embedding 和相似矩阵，不满足 12 小时有界内存目标。不得把它作为短音频或“质量更高时”的隐藏第二路径。

### 4.3 版本固定

- Python 固定为 `3.11.13`。FunASR 的上游源码基线固定为完整 commit `8a34247dc5ff71bea61b37e57f941680b456753f`；运行时只安装官方 PyPI `funasr==1.3.29` wheel `https://files.pythonhosted.org/packages/9c/10/0a43f6233db074e263c025718afff7e7960976ef5e545c40c92c5f59f1c9/funasr-1.3.29-py3-none-any.whl`，其 SHA-256 固定为 `bc022d3f80cab635227841a401cc872e5b863a207f8fa01262f15c42ed630137`，大小固定为 `956044` bytes；不得在运行时改从 Git tree、sdist 或其他同版本 artifact 安装。
- 该 wheel 发布的 425 个 `funasr/**` 文件与上述 commit 中的同路径文件逐字节一致；commit-only 的 28 个文件不属于当前 SenseVoice、FSMN-VAD、CAM++ 支持路径。启用 `EnglishTextNormalizer`、RWKV 或这些省略文件承载的其他能力前必须重新核对并更新 runtime artifact manifest。commit 固定源码基线，wheel URL、hash 和 size 固定实际安装物，二者不得相互替代。
- CPU release 的 `torch` 和 `torchaudio` 均固定为 `2.11.0+cpu`，只从 PyTorch 官方 CPU index `https://download.pytorch.org/whl/cpu` 解析和安装；不得让通用 PyPI 或其他额外 index 覆盖这两个 artifact。
- CUDA runtime、对应的 PyTorch artifact 和模型必须在各自 release 构建前精确固定；CPU pin 不自动成为 CUDA pin。
- 发布镜像不得在启动时执行 `pip install -U`。
- SenseVoice、FSMN-VAD、CAM++ 固定到下列 Hugging Face immutable commit；禁止解析 alias/master 后再记录“实际 revision”：

| 模型 | Hugging Face immutable revision | primary weight artifact | primary weight 预期 SHA-256 |
|---|---|---|---|
| SenseVoiceSmall | `FunAudioLLM/SenseVoiceSmall@3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` | `model.pt` | `833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea` |
| FSMN-VAD | `funasr/fsmn-vad@df20e6b30c653645fa4ff125cacfcabd1020a669` | `model.pt` | `b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5` |
| CAM++ | `funasr/campplus@e4b6ede7ce16997aff4ae69fbca1f0175e2afede` | `campplus_cn_common.bin` | `3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8` |

- 表中的 primary weight hash 只证明权重文件，不构成完整 snapshot attestation。SenseVoice runtime snapshot 还必须包含并校验 `configuration.json`、`config.yaml`、`am.mvn` 和 `chn_jpn_yue_eng_ko_spectok.bpe.model`；FSMN-VAD runtime snapshot 还必须包含并校验 `configuration.json`、`config.yaml` 和 `am.mvn`。这些文件的路径与逐文件 SHA-256 以 `src/botified_asr/model_artifacts.py` 的代码 manifest 为唯一真相，文档和 release manifest 从该来源生成或校验，不维护第二份 hash 长清单；CAM++ 接入 loader 前必须建立同等完整的 runtime manifest。
- 模型首次下载只从上述 Hugging Face immutable commit 进入按 revision 隔离的 cache；每次 ready/load 前逐文件校验完整 runtime manifest 的 SHA-256，全部通过后才允许加载并 warmup。
- 升级上游时运行本仓库的真实模型 smoke，不依赖“语义版本应当兼容”的假设。
- SenseVoice、FSMN-VAD、CAM++ 各加载一个单例 adapter；所有 model call 共用唯一串行 inference lane。
- 仓库代码采用 MIT license；每个 release 提供 `THIRD_PARTY_NOTICES`，README/manifest 记录模型名称、来源、revision 和 license URL。
- SenseVoice 权重许可与 FunASR toolkit 代码许可分别审核；只有许可明确允许再分发时才烘入 OCI，否则由 installer 按固定来源/hash下载，不以技术便利替代许可判断。
- SenseVoice 的预期 artifact hash 在 release 前必须由本地隔离下载重新计算并与 manifest 比较，不能只抄远端元数据。
- aarch64 CPU artifact 必须在原生 aarch64 runner 完成 fresh-install、模型加载和固定 smoke；x86_64 上的交叉构建不能替代。
- CUDA artifact 使用独立依赖锁、image digest，并在真实 NVIDIA/CUDA runner 验证；未经验证不得沿用 CPU 结论或标记 CUDA 受支持。

### 4.4 可复现 fingerprint

`processor_fingerprint` 是下列 version 1 canonical JSON manifest 的 UTF-8 bytes 的 SHA-256：

```json
{
  "model_snapshot_manifest_digests": {
    "campplus": "<sha256>",
    "fsmn_vad": "<sha256>",
    "sensevoice": "<sha256>"
  },
  "processor_compatibility_version": 1,
  "processor_policy_manifest_digest": "<sha256>",
  "result_envelope_version": 1,
  "speaker_snapshot_wire_version": 1,
  "version": 1
}
```

canonical JSON 使用 UTF-8、对象 key 字典序、无无意义空白、JSON array 保持声明顺序并拒绝 NaN/Infinity。

每个 `model_snapshot_manifest_digests` 值是 ready/loader 已验证的对应完整 runtime model snapshot canonical metadata 的 SHA-256。version 1 metadata 的 exact top-level keys 是 `files,immutable_revision,model_id,provider,version`；`version` 必须是 exact integer `1`。`files` entry 的 exact keys 是 `byte_size,relative_path,sha256`，entries 按 `relative_path` 严格升序且 path 唯一，不在每个 entry 重复 model ID 或 revision。metadata 不得包含 cache/host absolute path、mtime、inode 或其他部署时可变 metadata，也不得只绑定 primary weight。计算 fingerprint 复用 ready/loader 已验证的 canonical metadata，不为 fingerprint 二次读取模型大文件。

`processor_policy_manifest_digest` 是 canonical processor policy manifest 的 SHA-256。该 manifest 的 exact top-level keys 是 `policies,version`，`version` 是 integer `1`；`policies` 的 exact keys 是 `anonymous_clustering,asr_adapter,audio,join,result_projection,rich_label,segment,speaker_embedding,speaker_matching,text,vad`。每个 policy object 的 exact keys 是 `effective_parameters,version`，其中 `version` 是正整数，`effective_parameters` 是只含 JSON scalar/array/object 的 canonical object，并完整列出该 policy 当前生效的命名参数，不得用代码路径、类名或 Git revision 代替参数。

这些 policy object 分别覆盖：audio frontend、ffmpeg argv/protocol/demuxer、downmix、resample、PCM `s16le`/mono/16 kHz、9,600-sample block、EOF 与 sample-count；VAD block/cache、pre-padding 与边界换算；segment 最大长度、切分、padding 与 batching；ASR adapter 输入、语言与模型输出解析；clean text；rich-label parser；canonical join；result projection；anonymous clustering；speaker embedding window/normalization；known-speaker matching threshold/margin。任一 effective value 变化都必须改变对应 canonical policy object。

`processor_compatibility_version` 是兜底兼容版本：任何影响输出或旧 job/result 安全恢复、但不会由上述 model metadata、wire version 或 processor policy manifest 自动体现的代码、依赖、execution backend、adapter、parser、projection 或其他语义变化，都必须递增该值。不影响输出或恢复的重构、文件移动和 HTTP schema 变化不得递增。整个 Git commit 与 request/response/error schema hash 不进入 `processor_fingerprint`。

当前 release 的 `processor_fingerprint` 是进程常量，新 job row 使用该值。只有 queued/running job 的 claim、execute、crash recovery 和 running-result promotion 要求 job 值等于当前 release 值；不一致 fail closed `processor_fingerprint_mismatch`，不得用当前 processor 继续该 job。active job 由持久 metadata 重算出的 request fingerprint 不等于 job row 时 fail closed `request_fingerprint_mismatch`。读取或恢复任意 `.complete` 时，其 request fingerprint 和 processor fingerprint 必须分别等于 job row；request fingerprint 不一致是 result corruption，processor fingerprint 不一致使用 `processor_fingerprint_mismatch`。retained succeeded job 不要求其 processor fingerprint 等于当前 release，只要求 `.complete` 的两个 fingerprint 与该 job row 精确一致。

`request_fingerprint` 是下列 version 1 canonical JSON 的 UTF-8 bytes 的 SHA-256：

```json
{
  "canonical_options": {
    "chunking_strategy": null,
    "include": [],
    "known_speaker_ids": [],
    "language": "auto",
    "model": "sensevoice",
    "response_format": "json"
  },
  "input_sha256": "<sha256>",
  "speaker_snapshot_sha256": "<sha256>",
  "version": 1
}
```

`canonical_options` 是 §6.2 已验证的 canonical options JSON object，不是 JSON string，也不再包一层 hash；其数组顺序沿用 §6.2。`input_sha256` 是上传流中增量计算并在 sealed input 上固定的内容 SHA-256，不得为 fingerprint 二次读取输入。`speaker_snapshot_sha256` 是 §9.4 canonical snapshot bytes 的 SHA-256。fingerprint 只用于一致性，不作为公开资源 ID。

## 5. 用户故事

### 5.1 Botified 语音留言

Gateway 上传一个短音频，服务返回：

```json
{"text":"今天下午三点开会。"}
```

Gateway 不需要理解 FunASR 扩展字段。

### 5.2 富转写

客户端请求情感和音频事件，在不污染正文的前提下得到结构化标签。

### 5.3 匿名会议转写

客户端请求 diarization 且不提交已知人物候选，得到 `A`、`B` 等匿名说话人、绝对时间戳和正文。

### 5.4 已知人物会议转写

操作员预先为 Percy 等人物注册若干声音样本。转写会议时显式选择允许匹配的人物；匹配成功的 cluster 使用姓名，其他 cluster 使用 `Unknown A`、`Unknown B`。

### 5.5 500 MiB 长会议

客户端以异步模式上传文件，收到短 job ID；服务在重启后仍能继续或明确失败，客户端轮询结果且无需保持数小时 HTTP 连接。

### 5.6 Agent 自助使用

Codex、OpenClaw 或 Botified Agent 安装同一份 Skill，通过稳定脚本检查服务、注册人物、提交长会议、等待结果并生成会议记录。

## 6. 公开 API 契约

### 6.1 鉴权

发行运行时只注册产品 API 和 health 端点；`/health/live` 是唯一不要求鉴权的运行时端点，其余已注册端点均要求：

```http
Authorization: Bearer <BOTIFIED_ASR_API_KEY>
```

- 缺少或错误 token 返回 `401`。
- 比较使用 constant-time 实现。
- 不在日志、错误或 job 记录中保存 Authorization header。
- 首版只有一个部署级 API Key，不增加用户或租户概念。
- 发行运行时不注册 `/docs` 或 `/openapi.json`；OpenAPI JSON 在 release 构建时离线生成。

### 6.2 `POST /v1/audio/transcriptions`

“OpenAI 兼容”在首版只表示基础同步 transcription 的 wire-compatible 子集：

- 标准 OpenAI Python SDK 在自定义 `base_url` 下可提交受支持字段并读取基础 `{text}` 响应。
- `file`、`model`、`language`、`response_format`、`chunking_strategy` 沿用对应字段名和基础语义。
- 鉴权、成功响应和错误 envelope 在受支持组合内保持兼容。
- 不宣称兼容完整 OpenAI Audio API；异步 job、持久 speaker profile 和 `funasr` 字段均是 Botified 扩展。

兼容差异必须在 README、OpenAPI 和 Skill 中使用同一张表：

| 能力 | OpenAI 基础调用 | Botified ASR 首版 |
|---|---|---|
| 普通同步转写 | 支持 | SDK 可直接调用受支持子集 |
| model | OpenAI model name | 只接受固定 alias |
| `srt` / `vtt` | 可用 | 不支持 |
| `prompt` / 非零 `temperature` / `stream` | 依模型而定 | 不支持 |
| 异步 job | 非本兼容承诺 | `Prefer: respond-async` 扩展 |
| 已知人物参考 | 请求内 name + reference | 持久 speaker ID 扩展 |
| diarization format | 依模型和格式组合 | 固定 alias + 显式 `auto` chunking + `diarized_json` |
| 情感和事件 | 无 | namespaced `include[]` 扩展 |

标准 SDK 只保证基础同步路径；扩展能力由本仓库 helper 或直接 HTTP 调用。

请求类型：

```text
multipart/form-data
```

multipart parser 只接受一个 `file` part 和下表字段白名单；总 part 数最多 64、单 part header 最多 32 KiB。multipart overhead 定义为整个 multipart body 的实际字节数减去唯一 `file` payload 的实际字节数，包含 boundary、part header 和所有非 file part payload，最多 1 MiB。流式接收先执行 §7.1 的 raw body 联合安全上限；multipart 正常解析结束后再精确校验 `实际 body bytes - file payload bytes <= 1 MiB`。重复 scalar 字段、未知 part 和第二个 file 在进入模型前返回 `400 invalid_multipart`。

首版字段：

| 字段 | 必填 | 语义 |
|---|---:|---|
| `file` | 是 | 音频或含音频的视频 |
| `model` | 是 | `sensevoice` 或 `sensevoice-diarize` |
| `language` | 否 | `auto`、`zh`、`en`、`yue`、`ja`、`ko` |
| `response_format` | 条件 | `sensevoice` 可省略；`sensevoice-diarize` 必须显式提交 `diarized_json` |
| `chunking_strategy` | 条件 | `sensevoice` 可省略；`sensevoice-diarize` 必须显式提交 `auto` |
| `include[]` | 否 | `funasr.emotion`、`funasr.audio_events` |
| `known_speaker_ids[]` | 否 | 本次允许匹配的已注册人物 |

数组字段在 canonical request options 中使用稳定规则：

- `include[]` 先逐项验证；合法重复值去重后按 `funasr.emotion`、`funasr.audio_events` 的固定枚举顺序保存。
- `known_speaker_ids[]` 出现重复 ID 时返回 `400 invalid_known_speaker_ids`；全部 ID 验证存在且兼容后按 ID 排序，再用于 profile snapshot 和 request fingerprint。

首版明确拒绝：

- 非零 `temperature`；
- `prompt`；
- `stream=true`；
- 手工 VAD object；
- `srt`、`vtt`；
- 任意未知 model；
- 任意未知 namespaced include；
- `sensevoice-diarize` 未显式提交 `chunking_strategy=auto`；
- `sensevoice-diarize` 未使用 `diarized_json`；
- `diarized_json` 未使用 `sensevoice-diarize`；
- 已知人物参数与非 diarization model 组合。

上述拒绝项使用 OpenAI 风格 `400 invalid_request_error`，不得静默忽略。

### 6.3 Model 行为

#### `sensevoice`

- 无 `chunking_strategy`：直接转写，仅允许音频不超过 30 秒。
- `chunking_strategy=auto`：执行 streaming FSMN-VAD 后分段转写。
- `json` 默认只返回 `{text}`。
- `verbose_json` 返回语言、真实时长和真实 segments。

#### `sensevoice-diarize`

- 客户端必须显式提交 `chunking_strategy=auto`；缺省或其他值均返回 `400`，服务不得自动补齐。
- 客户端必须显式提交 `response_format=diarized_json`；缺省或其他值均返回 `400`，服务不得自动补齐。
- 在 VAD segment 上提取 CAM++ embedding。
- 使用同一套 speaker clustering 处理短音频和长会议。
- 已知人物只在匿名 cluster 完成后命名。

### 6.4 标点

SenseVoice 原生生成标点。首版不加载 CT-Punc，也不提供 punctuation 开关：

- 不叠加第二个标点模型；
- 不用字符删除伪装“关闭标点”；
- ITN 不与标点混为一项；
- `text` 必须清除 SenseVoice 控制 token，但保留正常标点。

### 6.5 同步响应

`response_format=json`：

```json
{
  "text": "识别后的文字"
}
```

`response_format=text`：

```text
Content-Type: text/plain; charset=utf-8
```

正文是纯文本，不是 JSON string。

`response_format=verbose_json`：

```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 3.2,
  "text": "识别后的文字。",
  "segments": [
    {
      "id": "0",
      "start": 0.24,
      "end": 3.08,
      "text": "识别后的文字。"
    }
  ]
}
```

禁止按字符数伪造时间戳。上游无法提供可靠时间信息时，省略对应 granular detail，而不是制造近似值。

解码后为 0 个 PCM sample 的输入是成功空结果，不调用 SenseVoice，也不生成 segment：

- `json` 返回 `{"text":""}`；
- `text` 返回 0-byte 正文，仍使用 `text/plain; charset=utf-8`；
- `verbose_json` 返回 `duration: 0`、`text: ""` 和 `segments: []`；显式请求语言时 `language` 回显该请求值，`language=auto` 时返回诚实的 `unknown`，不得为了填充语言而调用模型；
- 请求了 rich include 时只返回所请求的顶层 `funasr` 数组，数组为空。

### 6.6 Diarized response

遵循 OpenAI `diarized_json` 的基础形状：

```json
{
  "task": "transcribe",
  "duration": 18.4,
  "text": "大家好。今天讨论发布计划。",
  "segments": [
    {
      "id": "0",
      "type": "transcript.text.segment",
      "start": 0.62,
      "end": 3.14,
      "speaker": "Percy",
      "text": "大家好。",
      "funasr": {
        "speaker_id": "4X7K2M9Q",
        "anonymous_speaker": "A",
        "similarity": 0.78
      }
    },
    {
      "id": "1",
      "type": "transcript.text.segment",
      "start": 4.01,
      "end": 8.56,
      "speaker": "Unknown B",
      "text": "今天讨论发布计划。"
    }
  ]
}
```

规则：

- 已知 speaker 的 `speaker` 使用注册名称。
- 未提交 `known_speaker_ids[]` 时，匿名 speaker 使用 `A`、`B`。
- 提交候选后，未命中的 speaker 使用 `Unknown A`、`Unknown B`。
- 字母映射在同一 transcription/job 及其 crash recovery 前后稳定；不承诺跨请求稳定。
- `similarity` 是余弦相似度，不命名为 confidence。
- 不把人物 description 重复写入每个 segment。
- 默认 JSON 不暴露 embedding。

### 6.7 富标签扩展

仅当客户端请求对应 `include[]` 时返回：

```json
{
  "text": "谢谢大家。",
  "funasr": {
    "emotion": [
      {"label": "happy", "start": 0.4, "end": 3.0}
    ],
    "audio_events": [
      {"label": "speech", "start": 0.4, "end": 3.0}
    ]
  }
}
```

- SenseVoice 始终产生相关 token，因此 include 控制响应投影，不承诺减少推理成本。
- 每个 direct clip/VAD segment 最多投影一个 emotion 和一个 audio-event label；`start/end` 只能等于 enclosing segment 边界，因为 SenseVoice 不提供 rich token 内部时间。
- 顶层数组是逐 segment 标签的稳定聚合，不把分类标签伪装成独立事件 span。
- VAD 丢弃的纯掌声、背景音乐等 standalone non-speech 不保证检测；首版不增加第二条 non-speech 扫描 pipeline。
- `response_format=text` 与任意 `include[]` 组合返回 `400 incompatible_response_format`。
- 在 `json`、`verbose_json` 和 `diarized_json` 中，扩展都只位于响应顶层 `funasr`；segment 不复制全局数组。
- 正文不得注入 emoji。
- 未知上游标签映射为 namespaced `unknown` 并保留受限 raw tag，避免 silently drop。
- 标签解析规则由本服务拥有，不依赖上游私有 postprocess 输出 emoji。

### 6.8 OpenAI-compatible error envelope

```json
{
  "error": {
    "message": "chunking_strategy=auto is required for long audio",
    "type": "invalid_request_error",
    "param": "chunking_strategy",
    "code": "long_audio_requires_vad"
  }
}
```

错误分类：

| HTTP | 类型 |
|---:|---|
| 400 | 参数组合或媒体不可解码 |
| 401 | 鉴权失败 |
| 404 | job 或 speaker 不存在 |
| 409 | speaker name 冲突或资源状态冲突 |
| 413 | 超过部署最大上传或实际音频时长 |
| 422 | 可接受但需要异步/VAD 的请求模式 |
| 429 | storage、job 或推理 admission 饱和 |
| 500 | 未分类内部错误 |
| 503 | 模型未 ready 或运行时不可用 |

内部异常消息不得原样返回。

### 6.9 Models

```http
GET /v1/models
GET /v1/models/{model_id}
```

list 返回：

```json
{
  "object": "list",
  "data": [
    {"id":"sensevoice","object":"model","created":1785024000,"owned_by":"botified-asr"},
    {"id":"sensevoice-diarize","object":"model","created":1785024000,"owned_by":"botified-asr"}
  ]
}
```

- `created` 是该 release 的固定构建时间，不在每次请求变化。
- 单 model GET 返回同一 object；未知 ID 返回 `404 model_not_found`。
- 依赖和模型 revision 通过 `botified-asr --version` 与 release manifest 查看，不增加未定义的 `/version` API。

## 7. 长音频和异步 Job

### 7.1 边界

默认配置：

```yaml
limits:
  max_upload_bytes: 1073741824
  max_audio_duration_secs: 43200
  direct_max_audio_duration_secs: 30
  sync_max_upload_bytes: 67108864
  sync_max_audio_duration_secs: 3600
  max_active_uploads: 4
  max_queued_jobs: 16
  max_job_storage_bytes: 21474836480
  min_filesystem_free_bytes: 2147483648
  result_retention_hours: 24
```

- 部署者可以收紧这些边界，安装器使用以上发布上限。
- `max_upload_bytes` 不得高于 1 GiB，`max_audio_duration_secs` 不得高于 12 小时；服务启动时拒绝超过首版验证上限的配置。
- `max_upload_bytes` 只计算 transcription 请求中唯一 `file` part 的 payload 实际字节数；`file` payload 小于或等于上限时可接受。multipart overhead 另按 §6.2 的独立 1 MiB 上限计算。
- 每个请求先确定适用的 file limit `B`：带 `Prefer: respond-async` 或 `sync_max_upload_bytes == max_upload_bytes` 时为 `max_upload_bytes`，其余情况为 `sync_max_upload_bytes`。
- 应用在把 raw body byte 交给 multipart parser 分类前先累计 `R`，并施加联合安全上限 `R <= B + 1 MiB`；累计收到第 `B + 1 MiB + 1` 个 raw body byte 时固定返回 `400 invalid_multipart`。该错误优先于尚未分类的 file 或 overhead 单项错误。
- 不能只依据 Content-Length；应用层按实际流入的 raw body、`file` payload 和 multipart overhead 分别累计计数，Content-Length 是否存在或其声明值不得改变下述公开错误优先级。
- 上传完成后由 `ffprobe` 获取有限、非负的 duration，仅用于本次请求的 admission、VAD 要求和 sync/async 前置判断；该临时值不写入 job row，也不作为公开 progress total。
- 没有 `chunking_strategy=auto` 的请求超过 `direct_max_audio_duration_secs` 时，无论 sync/async 都返回 `422 long_audio_requires_vad`。
- 联合安全上限未先触发时，带 `Prefer: respond-async` 的请求累计收到第 `max_upload_bytes + 1` 个 `file` payload 字节便立即返回 `413`。
- 联合安全上限未先触发时，未带 `Prefer: respond-async` 且 `sync_max_upload_bytes < max_upload_bytes` 的请求累计收到第 `sync_max_upload_bytes + 1` 个 `file` payload 字节便立即停止接收，稳定返回 `422 async_required` 并幂等清理 upload lease、reservation 和 staging 文件；不得继续接收以推测最终是否超过硬上限。
- 联合安全上限未先触发时，未带 `Prefer: respond-async` 且 `sync_max_upload_bytes == max_upload_bytes` 的请求累计收到第 `max_upload_bytes + 1` 个 `file` payload 字节便立即返回 `413`，不得返回 `422`。
- 完成上传后，所有 POST 都先执行 `ffprobe` 可检查性、probe duration admission、VAD 要求和 sync/async 边界的前置判断；`ffprobe` 只负责 preflight。异步请求通过 preflight 后发布 queued job，不预解码，也不 spool 全量 PCM；实际 decode 错误和 actual PCM 时长越界由 job 执行阶段记录为稳定 terminal failure。实际 decode 的错误优先级按所选路径闭合：direct path 读到超过适用 `effective_direct_max_audio_samples` 的第一个 sample 时立即返回 `422 long_audio_requires_vad`；VAD path 只在超过适用 `effective_max_audio_samples` 或 12 小时发布硬上限时返回 `413 audio_too_long`。同步请求在 request 内遵循同一 actual path 顺序，不宣称 actual PCM 总时长硬上限总是先于 VAD 要求。流式接收期间继续按上述累计 byte 规则处理 byte 上限。

### 7.2 异步提交

```http
Prefer: respond-async
```

成功返回：

```http
HTTP/1.1 202 Accepted
Preference-Applied: respond-async
Location: /v1/audio/transcriptions/7K3M9Q2W
```

```json
{
  "id": "7K3M9Q2W",
  "status": "queued",
  "created_at": "2026-07-26T20:00:00Z"
}
```

### 7.3 Job ID

- 8 个 Crockford Base32 大写字符。
- 使用安全随机源，约 40 bit 空间。
- SQLite unique constraint 检测碰撞并重新生成。
- 不使用 UUID、时间戳路径或递增长整数。
- ID 只用于当前服务实例，不承诺跨实例全局唯一。

### 7.4 Job 状态

```text
queued -> running -> succeeded
                  -> failed
queued/running -> cancelled
```

公开 API 没有 `retrying`、`paused` 等额外状态。实现可使用不暴露的 `receiving`、`deleting` cleanup phase 和 `cancel_requested` 标记：

- 正常模型/媒体错误一次失败即 terminal，不自动重试。
- claim `queued -> running` 时生成 attempt token 并令 `attempt_no += 1`。
- claim 时将当前 process generation 持久化到 `job.owner_generation`。重启发现 running 且 `job.owner_generation == shutdown_marker.generation`（上一进程）时从头 requeue，不增加 `crash_recoveries`；其他异常遗留令 `crash_recoveries += 1`。
- 同一 job 最多一次 crash recovery；第二次异常则 `failed`。不实现 segment checkpoint resume。
- cancel 在当前有界 segment/batch 结束后生效。
- `effective_max_audio_samples` 在 job 提交时固定为当时适用的部署 `max_audio_duration_secs * 16000`，且不得超过 12 小时发布硬上限对应的 samples；它在 job 生命周期内不可变，重启后的配置变化不得改变旧 job 的限制。
- `effective_direct_max_audio_samples` 在 job 提交时固定为当时适用的部署 `direct_max_audio_duration_secs * 16000`，并受 `effective_max_audio_samples` 和 480,000 samples（30 秒）发布上限约束；它在 job 生命周期内不可变。同步请求不持久化 job 字段，但必须把本次请求开始时的当前 effective direct cap 传给同一个 processor，direct decoder 不得使用代码内固定的 480,000 threshold。
- `total_samples` 只表示 ffmpeg 实际输出的连续 mono 16 kHz PCM sample count，不由 `ffprobe` duration 换算。任一 attempt 首次成功到达 decoder EOF 前，`total_samples` 为 `NULL`；首次固定后在整个 job 生命周期内不可变。requeue 保留已经固定的 `total_samples`，只把 `processed_samples` 归零；新 attempt 到达 EOF 时必须得到完全相同的 actual count。succeeded 必须满足 `total_samples IS NOT NULL` 且 `processed_samples == total_samples`；failed/cancelled 可以没有 `total_samples`。

### 7.5 查询

```http
GET /v1/audio/transcriptions/{job_id}
```

queued/running：

```json
{
  "id": "7K3M9Q2W",
  "status": "running",
  "progress": {
    "processed_audio_secs": 1840.0,
    "total_audio_secs": 7200.0
  }
}
```

`processed_audio_secs = processed_samples / 16000`。`total_samples` 已固定时，`total_audio_secs = total_samples / 16000`；否则 `total_audio_secs = null`。queued/running 不返回 `ffprobe` estimate，也不把未知总时长伪装为 sample-exact duration；`progress` 是处理进度，不是预计完成时间。

succeeded 始终返回 job envelope：

```json
{
  "id": "7K3M9Q2W",
  "status": "succeeded",
  "result": {
    "text": "识别后的文字。"
  }
}
```

- `json`、`verbose_json` 和 `diarized_json` 的 canonical response 放在 `result`。
- `text/plain` 只用于同步请求；异步 `response_format=text` 使用 `result.text`。
- failed 返回 `{"id":"...","status":"failed","error":{...},"finished_at":"<RFC3339 UTC>"}`，不返回原始异常。
- cancelled 返回 `{"id":"...","status":"cancelled","finished_at":"<RFC3339 UTC>"}`。
- 所有时间使用 RFC 3339 UTC；queued/running 不返回不存在的 finished 字段，terminal 不返回 progress。

### 7.6 删除和取消

```http
DELETE /v1/audio/transcriptions/{job_id}
```

- queued：CAS 为 cancelled，异步清理输入，返回 `202 {"id":"...","status":"cancelled"}`；GET 在 retention 内可见 cancelled。
- running：原子设置 `cancel_requested=true`，返回 `202 {"id":"...","status":"running"}`；重复 DELETE 幂等返回相同响应，worker 在下一个安全点 CAS 为 cancelled 并清理。
- worker 只有在 `status=running AND attempt_token=<当前值> AND cancel_requested=false` 时才能提交 succeeded；DELETE 和结果提交竞争时只允许一个 CAS 获胜。
- succeeded/failed/cancelled：先转入内部 deleting phase，幂等删除 artifact 和 reservation 后删除记录，返回 `204`；deleting 对 GET 表现为 `404`。
- cancelled job 的再次 DELETE 进入上一条 terminal 删除语义；记录删除后的请求返回 `404`，不增加永久 tombstone。

### 7.7 持久化和恢复

SQLite 保存：

- job ID；
- 状态和 attempts；
- 安全的输入文件内部路径；
- 经过验证的 canonical request options；
- input size、`effective_max_audio_samples`、`effective_direct_max_audio_samples`、`processed_samples` 和 nullable actual `total_samples`；
- 当前 attempt、结果文件内部路径或稳定错误码；
- 已选择人物的 canonical job-private profile snapshot、snapshot SHA-256、request fingerprint 和 processor fingerprint；
- created/started/finished 时间。

SQLite 还保存 upload/job storage reservation、`cancel_requested`、attempt token、`owner_generation` 和 cleanup phase；内部字段不进入公开响应。

不保存：

- Authorization；
- 客户端原始文件名作为路径；
- 非本 job 已选择人物的 embedding；
- 原始异常 backtrace；
- 完整 transcript 到通用日志。

写入规则：

- 接收首个 body 字节前，先在 SQLite 原子创建不可见的 receiving 记录和初始 storage reservation，再写确定性 job 路径的 `.partial`。
- 完成 byte 与 `ffprobe` 前置校验后 `fsync` 输入文件、原子 rename 为 `.ready`、`fsync` 父目录；`ffprobe` duration 随本次请求结束丢弃，不持久化。随后在同一个 SQLite `BEGIN IMMEDIATE` 事务内依次：从 sealed input lease 取得 `input_sha256`；按 `canonical_options` 中升序的 speaker ID `SELECT` 当前 profiles 并验证存在性与 compatibility；serialize §9.4 snapshot 并计算 snapshot SHA-256；计算 §4.4 request fingerprint；写入 `canonical_options`、snapshot、request/processor fingerprint、`effective_max_audio_samples` 和 `effective_direct_max_audio_samples`，并以 `total_samples = NULL`、`processed_samples = 0` CAS 为 queued。commit 后才返回 `202`。该事务不预解码输入，也不创建或 spool PCM。
- 文件系统 rename 与 SQLite 不伪装成同一原子提交。启动时所有仍为 receiving 的记录，无论存在 `.partial` 还是 `.ready`，都转 deleting 并幂等清理；客户端尚未收到 job，不尝试从音频反推 metadata 或提升 queued。
- queued commit 后、`202` 返回前崩溃可能留下客户端不可见的有效 job，这是 HTTP 边界；它按正常 retention 处理，不引入 idempotency key 或额外查询协议。
- 每次 attempt 使用独立 JSONL 临时结果，从空文件逐 segment 写入。finalize 只用有界次数的顺序扫描生成内部 envelope：首行 manifest 含 `job_id`、attempt、request fingerprint 和 processor fingerprint，后续是 canonical result；不得整体 `json.dumps` 物化。
- `.complete` 落盘后才在一个 SQLite 事务中校验 attempt 并标记 succeeded，同时设置 `input_cleanup_pending`；实际 unlink 输入并 `fsync` 目录后才释放其 reservation，恢复器继续未完成 cleanup。
- 恢复 succeeded 必须同时看到 non-null `total_samples`、`processed_samples == total_samples`，以及 valid、exact-bound 的 `.complete` result；不得从 result body 或 `ffprobe` 值反推 `total_samples`。
- 若在结果 rename 后、状态提交前崩溃，恢复器验证 `.complete` envelope 后复用完全相同的 `status=running AND attempt_token=<当前值> AND cancel_requested=false` CAS；成功才提交 succeeded。取消已获胜时删除 `.complete` 并完成 cancelled cleanup，不重复推理或覆盖取消。
- 若 running job 没有有效 `.complete`，删除该 attempt 的 partial artifact，按 §7.4 的单次 crash recovery 重新排队。
- 恢复器只对 SQLite 引用路径和受控目录的严格文件名布局做对账；不从任意文件内容猜测或创建 job。
- 删除先持久化内部 deleting phase，再幂等删除文件和 reservation，最后删除 DB 记录；崩溃恢复继续 deleting，不能先丢记录再清文件。
- orphan、partial、cancelled 和 terminal 清理均幂等；每个 DB/file 边界有 fault-injection contract test。

### 7.8 单一 Job executor

- 一个进程内 worker 消费 SQLite queued jobs。
- queued job 按 `created_at, id` 稳定 FIFO 选择。
- 不增加第二个 broker。
- sync 和 async 共用同一 inference admission semaphore。
- 长 job 每处理一个有界 batch 释放 admission，让短同步请求有机会执行。
- 一个 inference unit 同时限制总 speech 时长（默认 60 秒）、segment 数（默认 32）和 decoded wall-audio span（默认 5 分钟）。
- inference lane 不跨 ffmpeg decode 持有；FSMN 每个 bounded PCM block 单独 acquire/release，SenseVoice/CAM++ 每个 unit acquire/release。
- unit 结束时若有 sync waiter，调度器先放行一个 sync waiter再继续 async job；不依赖普通 semaphore 的未定义公平性。
- admission 饱和时同步请求等待一个短有界时间，之后返回 `429`，不能无界堆积。

## 8. 大文件流式音频处理

### 8.1 单一处理路径

启用 VAD 的短音频和所有长音频使用同一条路径：

```text
bounded upload
  -> ffprobe
  -> ffmpeg streaming decode: mono PCM16 16 kHz
  -> streaming FSMN-VAD
  -> bounded speech segment buffer
  -> SenseVoice batch
  -> optional CAM++ embedding
  -> incremental result writer
  -> optional speaker naming
  -> final response projection
```

不得：

- 解码完整会议到单个 WAV；
- 将完整 PCM 放进内存；
- 为每个 VAD segment 从文件头重复解码；
- 将所有 segment 音频保留到 job 结束；
- 用固定 10 分钟切片直接截断句子；
- 因客户端断开而泄漏 ffmpeg 或临时文件。
- 把 file path 交给组合式 `AutoModel(model+vad+spk).generate` 或 `inference_with_vad`；
- 直接使用保存全历史的 `DynamicVAD.process/confirmed_segments`。

三个 adapter 的边界固定为：

- ffmpeg 只把 bounded PCM block 交给 FSMN-VAD `generate` 的 request-local cache，消费边界后立即丢弃已确认历史；
- 只有不超过 30 秒的 segment ndarray 可交给 SenseVoice 和 CAM++；
- adapter 不接受原始上传路径，不允许上游入口再次加载完整媒体；
- 12 小时测试采集 request-local VAD cache 的 tensor bytes；首个 10 分钟 warmup 后到结束增长不得超过 1 MiB，且 cache 中不得出现 confirmed-segment 历史列表。

公开入口只调用一个 mode-neutral processor：

```text
process(
  input_path,
  canonical_options,
  cancellation,
  progress_sink,
  segment_sink,
  *,
  effective_max_audio_samples,
  effective_direct_max_audio_samples
) -> result_artifact
```

processor 不知道 HTTP、job ID、SQLite 状态或 artifact 路径策略，只向注入的 `SegmentSink` 追加 sample-based canonical records，并接收 sink 完成后返回的 opaque artifact ref。API composition 在调用 processor 前从 storage 获取带 reservation 的 byte writer，再将其包装为 canonical JSONL `SegmentSink`；pipeline 不依赖 storage。sync 使用 request-owned artifact/progress sink，并传入本次请求的 current effective overall/direct sample caps；async 使用 durable attempt artifact/SQLite sink，并传入 job row 中 immutable 的 `effective_max_audio_samples` 和 `effective_direct_max_audio_samples`。direct/VAD 只是 processor 内的 segmentation policy，crash recovery 从头调用同一函数。只用一个 spy/structure test 证明两个入口调用该 processor，不复制两套行为测试。

### 8.2 Decoder

- 使用 argv 调用 ffmpeg，不经过 shell。
- 输入是服务生成的内部路径。
- 固定 `-nostdin`、只选 `0:a:0`，禁用 video/subtitle/data 输出；probe 和 decode 都有 wall timeout、bounded stdout/stderr。
- protocol 只允许本地 `file,pipe`，container 禁止网络；format 必须落在受支持音频/媒体 demuxer allowlist，playlist/concat-like 输入 fail closed。
- 输出固定为 little-endian signed PCM16（`s16le`）、mono、16 kHz。decoder 以 `DecodedBlock(start_sample, pcm)` 交付一维 C-contiguous `np.int16`，非末块固定为 9,600 samples，末块为 1–9,600 samples，`start_sample` 从 0 起严格连续。
- pipe read 可跨 read 保留至多 1 byte carry；EOF 仍剩 1 byte 是 malformed decoder output，必须失败，不能丢弃或补零。
- decoder 只有在成功读到 EOF、确认无残留半个 sample 且子进程成功退出后，processor 才调用 `progress_sink.update(processed_samples=actual_count, total_samples=actual_count)`；processor 不知道 job、attempt token 或 SQLite，也不执行 CAS 或 artifact cleanup。同步 request-local progress sink 只在本次请求内校验并固定 actual count。异步 durable progress sink 把 EOF update 实现为 attempt/cancel fenced CAS；CAS loser 的停止、partial artifact 清理以及禁止 finalize/commit 均由 async composition/storage 层负责。
- EOF 前的异步 progress update 同样必须要求 job 为 `visible/running`、attempt token 等于当前 token且尚未取消，并只允许 `processed_samples` 单调不减；`total_samples IS NULL` 时还必须满足 `processed_samples <= effective_max_audio_samples`，已有 non-null `total_samples` 时必须满足 `processed_samples <= total_samples`。EOF fenced CAS 还必须要求 `total_samples` 为 `NULL` 或已经等于 actual count、`processed_samples <= actual count` 且 `actual count <= effective_max_audio_samples`；成功时原子设置 `processed_samples = total_samples = actual count`。已有 non-null `total_samples` 与新 attempt actual count 不同必须 fail closed；任何 fence/CAS loser 都由 async composition/storage 停止当前 attempt 并幂等清理其 partial artifact，不得 finalize 或提交结果。
- processor 边界始终使用 `np.int16`；模型 adapter 只在当前有界 segment 内执行 `pcm.astype(np.float32) / 32768.0`，校验一维、finite 和长度后调用模型，不在 processor 额外保留一份 float32 全量副本。
- direct path 同样按输出 PCM sample 计数，并使用调用方提供的 `effective_direct_max_audio_samples`，不得在 decoder 内写死 480,000。1–effective direct cap samples 只产生一个 half-open segment `[0,n)`；读到第 effective direct cap + 1 个 sample 时立即终止 decoder 并返回 `422 long_audio_requires_vad`，不得调用模型、追加/完成成功 segment sink 或留下 partial artifact/reservation，也禁止把 cap 内前缀作为成功响应。异步执行使用 job 持久化的 immutable cap，同步执行使用本次请求的 current effective cap；首版发布 cap 上限仍为 480,000 samples（30 秒）。
- direct path 解码为 0 samples 时按 §6.5 返回成功空结果，模型调用数为 0。
- VAD path 按实际 PCM sample 数执行调用方提供的 `effective_max_audio_samples`；超过该值或 12 小时发布硬上限即停止并返回 `413 audio_too_long`，`ffprobe` 只做前置拒绝。异步执行使用 job 持久化的 immutable cap，同步执行使用本次请求的 current effective cap。
- stderr 有界捕获，错误不回传原始命令或宿主路径。
- 子进程跟随 request/job cancellation。
- 每个 PCM block 大小固定并有背压。

支持输入：

```text
flac mp3 mp4 mpeg mpga m4a ogg wav webm
```

实际是否可解码以打包的 ffmpeg 为准，不仅依赖扩展名或 Content-Type。

### 8.3 VAD buffer

- streaming VAD 输出绝对毫秒边界。
- ring/spool 只保留当前未闭合语音和少量前置 padding。
- 单段达到 30 秒时强制安全切分。
- segment 是唯一 ASR、情感、事件和 speaker embedding 输入单位。
- 空白音频返回空字符串和空 segments，不生成模型幻觉文本。

### 8.4 Incremental result

- 每个完成 batch 的 segment、文本和标签追加到唯一 JSONL 中间真相；canonical line 保存整数 `start_sample`/`end_sample`，公开响应只在 projection 时统一换算为秒。
- 内存只保留当前 batch 和有界 speaker state。
- 最终 `text` 由 canonical segment 顺序拼接。
- 所有输出格式共用一个 canonical text join helper。每个 segment text 只移除首尾 Unicode whitespace，不执行 NFC/NFKC 或其他正文 normalization；trim 后为空的 segment 跳过。连接相邻非空文本时，若左侧末字符 Unicode category 为 `Ps`/`Pi`、右侧首字符 category 为 `Pe`/`Pf`/`Po`，或任一边界字符属于 Han、Hiragana、Katakana，则不插入字符；其他情况恰好插入一个 U+0020。Hangul 按普通 letter 处理，因此默认插入 U+0020，segment 内已有空格不改写。
- join golden 至少固定：`你` + `好` = `你好`，`hello` + `world` = `hello world`，`Hello.` + `Next` = `Hello. Next`，`hello` + `,` = `hello,`，`中` + `English` = `中English`，`English` + `中` = `English中`，`(` + `hello` = `(hello`，`hello` + `)` = `hello)`，`안녕` + `하세요` = `안녕 하세요`；outer whitespace 被 trim、empty 被跳过，emoji 作为普通非 CJK 边界默认以 U+0020 分隔。
- job progress 由已完成的绝对音频位置计算。
- finalize 顺序扫描 JSONL，依次流式写 escaped canonical text、segments 和顶层 rich arrays并完成 `.complete`；允许为不同顶层数组 rewind 重读，不允许载入完整结果。恢复时用 streaming parser 校验 JSON 完整结束和 manifest fingerprint。
- succeeded GET 跳过私有 manifest line，写 job envelope prefix，将 canonical body 流式嵌入 `result` 后写 suffix；不得为返回 12 小时结果重新构造完整 dict/string。
- partial、JSONL、`.complete`、输入和 enrollment staging 各自记录实际 reservation；只有 artifact unlink 且父目录 `fsync` 后才释放对应 bytes。

### 8.5 Speaker state

首版使用有界的稳定 centroid 状态：

- 使用 FunASR `sv_chunk` 的固定 1.5 秒 window / 0.75 秒 shift 提取并归一化 CAM++ embedding。
- 按绝对时间顺序与最多 32 个 centroid 做 cosine nearest；达到随 embedding policy 固定的 anonymous-cluster threshold 时更新最近 centroid，否则按首次出现顺序创建稳定 ID。
- centroid 以累计 window count 做加权平均后重新归一化，永不 merge、split 或 renumber。
- 一个 VAD segment 由 window 时长加权多数决定 speaker；tie 选择最小稳定 ID。
- 第 33 个 unmatched centroid 直接失败 `too_many_speakers`，不合并成随机人物。
- 状态只保存 32 个 centroid、count 和 label，不保存全历史 embedding。

短音频和长会议不得使用两套 speaker label 算法。

## 9. 已知人物注册

### 9.1 数据模型

```text
SpeakerProfile
  id: 8-char Crockford Base32
  name: unique display name
  description: optional short metadata
  embedding: normalized averaged CAM++ embedding
  embedding_model_id
  embedding_model_revision
  embedding_dimension
  embedding_policy_fingerprint
  sample_count
  created_at
  updated_at
```

约束：

- `name` trim 后 1–80 个 Unicode 字符。
- name 按 Unicode casefold 唯一，避免同实例中两个不可区分名字。
- `A`–`Z` 后按 `AA`、`AB` 延伸；纯大写拉丁字母标签以及大小写不敏感的 `Unknown <该标签>` 是响应保留名称，不得注册，冲突返回 `400 reserved_speaker_name`。
- `description` 最多 500 个 Unicode 字符。
- 单实例固定最多 256 个 speaker profile；达到时 POST 返回 `409 speaker_profile_limit_reached`，首版因此不增加 pagination。
- `embedding_policy_fingerprint = SHA256(model commit + dimension + 16k/downmix + 1.5s/0.75s window/shift + padding + normalization policy version)`。
- profile fingerprint 必须与当前只读 model policy 一致；不匹配时返回 `409 speaker_profile_incompatible` 并要求重新 enrollment，不得静默复用。
- 首版不增加头像、邮箱、角色、组织等字段。

### 9.2 `POST /v1/speakers`

multipart：

```text
name=Percy
description=项目负责人
samples[]=voice-1.wav
samples[]=voice-2.wav
```

规则：

- 2–5 个样本。
- 单样本上传最多 20 MiB。
- 静音经同一 VAD 去除后，每个样本必须包含 5–30 秒有效人声，否则返回 `400 invalid_speaker_sample_duration`。
- 每个样本由同一人录制是操作员责任；首版不增加第二套“样本是否单人”检测模型，也不虚假承诺能自动证明身份。
- 不可解码、没有人声或出现非有限 embedding 分别返回稳定的 `invalid_audio`、`no_speech`、`invalid_speaker_embedding`。
- 每个 sample 先将其 window embedding 等权平均并归一化为 sample centroid；2–5 个 sample centroid 再等权平均并归一化，避免长样本因 window 更多而获得更大权重。
- 一致性只比较 sample centroid pair；任一 pair 低于随 policy 发布的 enrollment threshold 时拒绝 `speaker_samples_inconsistent`。这只拦截明显混杂，不宣称证明同一身份。
- 全部样本成功才创建 profile。
- 创建完成立即删除原始和解码样本。
- POST/PUT 复用 §11.3 generic storage ledger、typed upload lease、reservation、受控 staging 和启动对账；进程崩溃遗留样本按 writing lease cleanup 删除，不建立第二套清理器。
- PUT 只有新 embedding 完成并在单个 SQLite transaction 替换成功后才切换 profile；任一失败或崩溃保留旧 profile 并清理新样本。
- 不向客户端返回 embedding。

### 9.3 查询、更新、删除

```http
GET /v1/speakers
GET /v1/speakers/{speaker_id}
PUT /v1/speakers/{speaker_id}
DELETE /v1/speakers/{speaker_id}
```

- list 只返回 ID、name、description、sample_count、embedding model fingerprint 和时间。
- PUT 是唯一更新入口，使用 `multipart/form-data`，必须提交 `name`。
- `description` 未提交时保留，提交空字符串时清空，提交非空字符串时替换；不接受 `null` multipart 值。
- `samples[]` 未提交时保留 embedding 和 `sample_count`；提交时必须完整提供 2–5 个新样本并原子替换 embedding 和 `sample_count`；空 samples 返回 `400 invalid_speaker_samples`。
- metadata-only PUT 不改变 `sample_count`，成功更新均刷新 `updated_at`。
- DELETE 立即删除 metadata 和 embedding。
- job 在创建时已经完成 profile snapshot；之后删除或更新 profile 不改变 queued、running 或已完成 job。

### 9.4 转写时选择人物

- 客户端必须显式提交 `known_speaker_ids[]`。
- 不提交时不扫描全库。
- 单请求最多 32 个已知人物。
- 重复 ID 按 §6.2 返回 `400`；全部 ID 验证存在且与当前 embedding policy 兼容后按 ID 排序。
- sync 请求在进入 processor 前、async 请求在 job 创建事务中 snapshot profile ID、name 和 embedding，避免运行中更新造成同一会议前后不一致。
- async publish 在同一个 job 创建事务中按 ID 升序读取 profiles，并编码为 strict canonical UTF-8 JSON。version 1 wire 的唯一 shape 是 `{"speakers":[{"embedding":"...","id":"...","name":"..."}],"version":1}`；top-level exact keys 为 `speakers,version`，speaker entry exact keys 为 `embedding,id,name`，`version` 必须是 exact integer `1`，不得接受 boolean、其他 number 或 string。speaker array 最多 32 项，严格按 ID 升序且拒绝 duplicate ID；`name` 必须等于 speaker profile 的 canonical name。
- `embedding` 是 canonical little-endian float32 embedding bytes 的 RFC 4648 standard Base64，保留 `=` padding且不得含 whitespace。decode 后必须使用 `SpeakerEmbedding.from_bytes` 和当前 `speaker_embedding_policy_fingerprint` 固定的 expected dimension 验证 byte length、finite 与 normalization；parser 必须拒绝未知/缺失 key、非 canonical Base64，以及任何不能原样 reserialize 的 wire。empty snapshot 的唯一 bytes 是 `{"speakers":[],"version":1}`。整个 canonical snapshot UTF-8 wire 不得超过 64 KiB；它继续内联存储在现有 job 私有数据中，不新增 artifact、table 或 lease。
- snapshot SHA-256 对上述 exact bytes 计算；snapshot speaker ID 必须与 `canonical_options.known_speaker_ids` 完全一致。snapshot 只存储在 job 私有数据中，job 清理时一并删除；profile 后续更新或删除不改变该 job snapshot。

### 9.5 匹配规则

流程：

```text
anonymous clustering
  -> cluster centroid
  -> compare with selected profile embeddings
  -> threshold + top-two margin
  -> known name or Unknown
```

- threshold 和 top-two margin 是服务级固定模型策略，不由每个客户端随意调整。
- 阈值必须通过真实设备、普通话/英语和不同麦克风样本校准。
- enrollment threshold、anonymous-cluster threshold、match threshold 和 top-two margin 与 embedding policy fingerprint 一起固定；变更任一值需要重新跑同一 identity 测试。
- 相似度达到阈值但与第二名区分不足时保持 Unknown。
- 多个 anonymous cluster 可以匹配同一已知人物，以容忍长会议 cluster fragmentation。
- 匹配只使用声音；name 和 description 不进入模型。
- 未提交候选时跳过 profile matching 并输出 `A/B/...`；提交候选时只有未命中 cluster 使用 `Unknown A/B/...`。

### 9.6 OpenAI stateless known speakers

首版不同时实现 per-request data URL reference audio。

原因：

- Botified 需要的是可复用的持久人物；
- 500 MiB 会议不应反复携带参考音频；
- 同时实现持久 profile 和 stateless reference 会形成两套 enrollment 心智。

`known_speaker_names[]` / `known_speaker_references[]` 不在首版实现。只有出现明确第三方客户端需求时才重新立项；当前 OpenAI 兼容声明必须明确这一差异。

### 9.7 Speaker API wire contract

POST 成功返回 `201`，GET/PUT 成功返回 `200`，DELETE 成功返回 `204`。单资源形状固定为：

```json
{
  "id": "4X7K2M9Q",
  "object": "speaker",
  "name": "Percy",
  "description": "项目负责人",
  "sample_count": 2,
  "embedding_model": {
    "id": "cam++",
    "revision": "<pinned-revision>",
    "dimension": 192,
    "policy_fingerprint": "<sha256>"
  },
  "created_at": "2026-07-26T20:00:00Z",
  "updated_at": "2026-07-26T20:00:00Z"
}
```

list 返回 `{"object":"list","data":[...]}`，按 `created_at,id` 稳定排序，不在首版增加 pagination。DELETE 无响应体。speaker 不存在返回 `404 speaker_not_found`，名称冲突返回 `409 speaker_name_conflict`，job 提交引用不存在或不兼容的 profile 时整次请求失败，不忽略部分 ID。

## 10. 配置

### 10.1 单一配置文件

默认：

```text
${XDG_CONFIG_HOME:-$HOME/.config}/botified-asr/config.yaml
```

示例：

```yaml
server:
  listen: "127.0.0.1:8090"
  public_base_url: "http://127.0.0.1:8090"

runtime:
  device: "auto"
  model_cache_dir: "~/.cache/botified-asr/models"
  max_speakers: 32

storage:
  data_dir: "~/.local/share/botified-asr"

limits:
  max_upload_bytes: 1073741824
  max_audio_duration_secs: 43200
  direct_max_audio_duration_secs: 30
  sync_max_upload_bytes: 67108864
  sync_max_audio_duration_secs: 3600
  max_active_uploads: 4
  max_queued_jobs: 16
  max_job_storage_bytes: 21474836480
  min_filesystem_free_bytes: 2147483648
  result_retention_hours: 24
```

规则：

- 非敏感配置只从该文件读取。
- CLI 仅接受 `--config` 定位文件，不为每个字段再增加 CLI/env alias。
- `~` 在配置加载时由应用统一展开。
- 当前 CPU artifact 的 `runtime.device` 只接受精确值 `auto` 或 `cpu`，并将两者都 canonicalize 为 `cpu`；不得把 `auto` 直接传给 Torch/FunASR，也不得探测硬件后临时启用当前 artifact 未声明的 CUDA 路径。
- `storage.data_dir` 和 `runtime.model_cache_dir` 展开 `~` 后必须已经是 absolute path，再以 `resolve(strict=False)` 规范为 canonical path；两个 canonical root 相等、任一位于另一目录树内时均 fail closed。
- model cache 只归 model artifact resolver 管理，不进入 `Storage` ledger、job data reservation、retention 或 orphan cleanup。
- 未知字段 fail fast。
- `limits` 下所有值均为正整数；`runtime.max_speakers` 是 `1..32` 的整数。
- duration 必须满足 `direct_max_audio_duration_secs <= sync_max_audio_duration_secs <= max_audio_duration_secs <= 43200`。
- upload bytes 必须满足 `sync_max_upload_bytes <= max_upload_bytes <= 1073741824`。
- storage 必须满足 `max_job_storage_bytes >= ceil(max_upload_bytes / 8388608) * 8388608`，即将 `max_upload_bytes` 向上取整到 8 MiB reservation 量子的整数倍。
- 上一条只保证没有其他 reservation 且 filesystem free-space floor 满足时，单个最大 transcription 输入可获得 staging reservation；不保证并发上传或后续 intermediate、attempt、complete、terminal artifact 的空间，后者仍在每次写盘前执行 §11.3 的 SQLite 原子 reservation 和 free-space admission。

### 10.2 Secret

唯一 secret：

```text
BOTIFIED_ASR_API_KEY
```

- 应用只读取进程环境中的 `BOTIFIED_ASR_API_KEY`，不定位、打开或解析 `service.env`。
- installer 创建 mode `0600` 的 `${XDG_CONFIG_HOME:-$HOME/.config}/botified-asr/service.env`，systemd/container launcher 从该文件向服务进程注入唯一同名环境变量。
- YAML 不支持明文 `api_key` 字段。
- 应用不得 trim 或以其他方式改写该值；整个值必须符合 RFC 6750 `b64token` ASCII grammar：至少一个 `ALPHA`、`DIGIT`、`-`、`.`、`_`、`~`、`+` 或 `/`，之后只允许可选的尾随 `=` padding。
- 无论监听地址是什么，缺少、空、含空白、Unicode、控制字符、其他非法字符或中间 padding 的 secret 都拒绝启动。
- secret 校验错误只返回稳定的配置错误，不得回显 secret 的全部或部分内容。
- 本地开发在进程环境中显式设置测试 token，不提供隐式 no-auth。

### 10.3 唯一客户端连接配置

本机服务、远程服务和三个 Agent runtime 使用同一种客户端连接文件：

```text
${XDG_CONFIG_HOME:-$HOME/.config}/botified-asr/client.env
```

文件只接受以下两个键：

```text
BOTIFIED_ASR_BASE_URL=http://127.0.0.1:8090
BOTIFIED_ASR_API_KEY=<secret>
```

规则：

- fresh 本机 `install-asr.sh` 创建 `service.env` 和该客户端文件，初始 key 相同；已有 `client.env` 时只创建/保留服务 secret，不改 active target。systemd 从 `service.env` 注入服务进程环境，Skill helper 只读 `client.env`。
- 连接远程现有服务时，`install-asr-skill.sh` 接受显式 `BOTIFIED_ASR_BASE_URL` 和 `BOTIFIED_ASR_API_KEY`，并写入相同路径。
- 已存在 `client.env` 且安装命令未提供连接参数时保持现有 active target；同时显式提供远程 URL/Key 时先验证格式再原子切换。
- URL 与 Key 必须成对提供，拒绝半更新；任何 active target 切换都不得读取或修改 `service.env`。
- 只安装 Skill 且未给连接信息是允许的；helper 的 `health` 必须返回稳定的 `client_not_configured` 和该文件路径，其他命令同样 fail fast。
- helper 使用键白名单解析，不执行或 `source` 文件内容。
- 文件和原子替换临时文件权限均为 `0600`。
- token 不写入 Skill、YAML、shell profile、命令示例或日志。
- `service.env` 是服务启动 secret，不是客户端配置；远程 Skill 安装不得修改它。
- 不增加第二种客户端配置格式；进程环境中的同名变量只作为显式的一次性覆盖，便于 CI 和运维调用。

## 11. 健康、日志和数据生命周期

### 11.1 Health

```http
GET /health/live
GET /health/ready
```

- live 只证明进程事件循环可响应。
- ready 必须证明 SQLite 可用、模型加载完成、warmup 成功、job executor 已启动。
- health 不返回路径、token、speaker 名单或完整依赖版本。

### 11.2 结构化日志

每个请求/job 记录：

- request/job ID；
- model alias；
- 输入字节和 duration；
- sync/async；
- VAD、diarization、rich include；
- queue wait、decode、VAD、ASR、speaker、total duration；
- 状态和稳定错误码。

默认不记录：

- transcript；
- 原始文件名；
- 原始音频；
- speaker name/description；
- embedding；
- Authorization；
- multipart body。

### 11.3 清理

- sync 请求结束时删除输入和中间文件。
- async 成功后立即删除输入和 segment 文件，只保留结果到 retention。
- failed/cancelled job 删除输入和中间文件。
- 定时清理只处理 SQLite 中已过期 terminal job。
- 磁盘达到 admission 上限时拒绝新 job，不删除尚未过期的用户结果来腾空间。
- transcription sync/async 与 speaker enrollment upload 都在首字节前获得 SQLite storage lease；并发接收数由 `max_active_uploads` 限制。
- multipart file 从首块开始直接 spool 到受控文件，内存阈值固定为 8 MiB；不得由框架先聚合完整 body。
- storage accounting 包含 partial/ready 输入、sync 临时文件、running intermediate、attempt result 和 terminal result。
- 每个 lease 先预留 8 MiB，后续在写盘前按 8 MiB 增量原子扩展；实际完成时结算，所有拒绝、断连和取消路径释放。
- admission 同时检查 SQLite 总 reservation 与 `min_filesystem_free_bytes`；多个请求不能以“各自先检查”突破总配额。

所有 upload 和 artifact 使用同一张 generic ledger：

```text
storage_leases(
  id,
  owner_kind,
  owner_id,
  resource_kind,
  lease_type,       # upload | artifact
  phase,            # writing | sealed
  controlled_path,
  reserved_bytes,
  actual_bytes,
  content_sha256,
  created_at
)
```

- `owner_kind`、`owner_id` 和 `resource_kind` 是恢复与计费归属，不引入新的 owner service；`resource_kind` 由应用层 typed constructor 校验，不做每增加一种 artifact 都要迁移的封闭数据库枚举。
- backing ledger 共用不等于暴露万能 handle：writing upload、sealed input、writing artifact writer 和 sealed result artifact ref 使用不同的 typed handles，禁止跨 type/phase 调用 append、complete、resolve 或 release。
- `max_active_uploads` 只统计 `lease_type=upload AND phase=writing`。`complete_upload` 原子转为 `sealed` 后立即释放上传并发槽，但输入的实际字节 reservation 保留；artifact 不占上传并发槽，但 writing/sealed 都计入总 storage reservation。
- v1 `upload_leases` 到 generic ledger 的 v2 migration 在单一 SQLite transaction 中显式完成并更新 schema version；遇到未知更高 schema version fail closed，不静默删库或重建。
- reservation 和 free-space admission 在 begin 和每次扩展写盘前执行。容量上限或 `min_filesystem_free_bytes` floor 拒绝统一使用公开 `storage_capacity_exceeded`、HTTP `429`；实际 filesystem I/O、fsync 或 SQLite 故障保持内部 server error，不伪装为容量 admission。
- 只有受控文件已经 unlink（或确认不存在）且父目录 `fsync` 后才删除 ledger row 并释放字节 reservation；sealed 只改变可见生命周期，不释放字节。

## 12. 进程和并发模型

- 一个 ASGI 进程对应一个设备。
- GPU 默认一个 Uvicorn worker，禁止通过 `--workers N` 在同一卡重复加载模型。
- 阻塞模型调用必须离开 asyncio event loop。
- 首版 inference lane 固定为 1，不提供配置项；FunASR adapter 的共享 kwargs/cache 不允许并发调用。
- 未来并发推理必须每 lane 独立加载完整 model bundle，属于单独性能立项。
- ffmpeg decode 可以与 GPU inference 重叠，但受有界 channel 控制。
- API 上传、job worker 和模型推理不得各自建立无界队列。
- 收到 SIGTERM 后立刻停止接收新请求和 claim job，写入 shutdown marker，并给当前工作 30 秒 grace 到达安全点。
- marker 包含上一进程的唯一 generation；下一次启动在同一个恢复 transaction 中比较 `job.owner_generation`、分类 running rows并清除 marker。marker 的写入、分类、清除和中途崩溃使用 fault injection 覆盖，不能让后续真实 crash 永久绕过计数。
- ffmpeg 使用独立 process group；取消/关闭时先 TERM，5 秒后仍未退出则 KILL，并关闭 pipe。
- 已进入底层 CPU/GPU 推理的单次调用不承诺可抢占；超过 grace 由 systemd 在 `TimeoutStopSec=120` 后终止进程，启动恢复依据 shutdown marker 从头重排 job 且不计为 crash retry。
- 成功到达安全点的 running job 在退出事务中回到 queued；sync 请求在 grace 内未完成则断开并清理，不转换为 async。

## 13. 代码组织

- 不先拆成多个 Python package。
- 不创建通用 repository/service/use-case 分层。
- 文件超出清晰职责后再按真实边界拆分。
- API DTO、domain state 和 SQLite row 不必强行共用一个类型。

## 14. 一键安装和公开发布

### 14.1 发布物

`botified-asr` 负责构建：

- CPU OCI image；
- 通过真实 runner 验证时构建 CUDA OCI image；
- `botified-asr-skill.tar.gz`；
- `botified-asr-smoke.flac` 固定短转写音频；
- 版本和 image digest manifest；
- 离线生成的 OpenAPI JSON。

`botified-releases` 公开：

```text
install-asr.sh
install-asr-skill.sh
botified-asr-skill.tar.gz
botified-asr-smoke.flac
botified-asr-release.json
SHA256SUMS
```

OCI image 发布到固定公开 registry，并在 `botified-asr-release.json` 固定 digest。安装器不得只拉 mutable `latest`。

首版发布矩阵：

| 平台 | CPU | CUDA |
|---|---:|---:|
| Linux x86_64 | 必须发布并验证 | 仅在真实 runner 验证后发布 |
| Linux aarch64 | 必须发布并验证 | 不支持 |

CUDA image 的 PyTorch/CUDA runtime、最低 NVIDIA driver 和受支持 compute capability 由锁定依赖决定并写入 manifest；安装器在拉取大资产前校验。未验证的组合不得靠 `device=auto` 猜测启用。

### 14.2 `install-asr.sh`

必须：

1. 支持 `BOTIFIED_ASR_VERSION=vX.Y.Z`，默认最新稳定版。
2. 检测 Linux x86_64/aarch64；不支持的平台在下载大资产前失败。
3. 检测 Docker 或 Podman；不存在时给出前置条件，不擅自安装系统容器运行时。
4. 先下载 release manifest 和 SHA256SUMS，严格校验 checksums 后才解析 manifest，并验证其 schema、目标 platform 与 runtime/image matrix。
5. 只依据已验证的 runtime/image matrix 选择 device 和 image；CPU artifact 中 `device=auto` 与 `device=cpu` 都 canonicalize 为 CPU，只有 manifest 含目标平台 CUDA digest 且 NVIDIA runtime 验证通过时才可选择 CUDA artifact。
6. 按选定的 manifest digest 拉取对应 OCI image。
7. 创建配置、credential、data 和 model cache 目录；data 与 model cache 使用互不相等、互不嵌套的独立持久 mount。
8. 未提供 API Key 时生成安全随机 token，分别写入 mode `0600` 的 `service.env` 和 §10.3 `client.env`；不得回显。
9. 安装单个 systemd service；没有 systemd 时安装可执行启动 wrapper 并明确提示。
10. 启动服务、等待 ready，并用安装器内置的最小 HTTP smoke 执行一个随发布固定的短音频转写。
11. 输出 base URL、credential 路径、service status、upgrade 和 uninstall 命令。
12. 重复运行时执行幂等升级，不覆盖用户配置和 API Key。
13. 升级先拉取并验证新 image；新版本 ready/smoke 失败时恢复旧 image 和 service，保留失败日志。
14. 升级保留 config、`service.env`、`client.env`、speaker database、model cache、jobs 和未过期结果。
15. uninstall 默认只停止并移除 service/wrapper，保留上述数据；只有显式 `--purge` 且二次确认后才删除。
16. manifest 的 processor fingerprint 改变且存在 queued/running job 时拒绝切换，输出完成或取消这些 job 的命令；不让新 image 继续旧 job snapshot。
17. 停止旧服务后使用 SQLite backup API 创建一致备份；新版本 schema migration、ready 或 smoke 失败时同时恢复旧 DB 和旧 image。model cache 按 revision 隔离，不在原目录原地改写。
18. 每个 schema migration 必须声明旧 image 是否可读；没有备份/恢复验证的 schema-changing upgrade 直接阻断。

发布/安装细则：

- 首版只安装当前操作员的 rootless container 和 systemd user unit，不同时维护 system service 做法；拒绝以 root 执行。
- `device=auto` 只有在当前 manifest 含本平台 CUDA digest且 runtime 校验通过时选择 CUDA；否则确定性使用 CPU，不临时拼装未发布 GPU 路径。
- unit 位于 `~/.config/systemd/user/botified-asr.service`，通过 `systemctl --user enable --now` 管理。
- installer 检测 user manager 和 linger；无 user systemd 时仅生成 `~/.local/bin/botified-asr-service` wrapper，无 linger 时明确说明重启后不会自启并给出管理员应执行的 `loginctl enable-linger <user>`，不擅自 sudo。
- `SHA256SUMS` 覆盖 manifest、Skill tarball、两个 installer 和 `botified-asr-smoke.flac`；manifest 在校验 checksum 后才解析。
- manifest 为每个受支持平台给出 image digest、模型 ID/revision 和 runtime 要求，并记录包括固定短 smoke 音频在内的 artifact checksum。
- Skill tarball 解包前拒绝绝对路径、`..` traversal、device node 以及逃逸目标目录的 symlink/hardlink。
- checksum harness 向现有 `botified-releases/tests/installers.sh` 注入损坏 manifest/tarball，并断言拉 image、写 systemd 或替换当前安装前已失败；不另建 installer framework。
- 唯一卸载入口为已安装的 `botified-asr-uninstall [--purge]`。

不做：

- 自动安装 GPU driver；
- 自动修改防火墙；
- 默认绑定公网；
- 默认启用 TLS；
- 下载来源不明的模型或脚本。

### 14.3 版本关系

- `botified-asr` 使用独立 SemVer。
- `BOTIFIED_ASR_VERSION` 不复用 `BOTIFIED_VERSION`。
- `botified-releases` 使用 namespaced release tag `asr-vX.Y.Z`，避免与 Core/Gateway 的 `vX.Y.Z` tag 冲突。
- `BOTIFIED_ASR_VERSION=vX.Y.Z` 由安装器解析为 `asr-vX.Y.Z` release，并拉取 `ghcr.io/lzjever/botified-asr:vX.Y.Z` 的固定 digest。
- Botified Core、Gateway 和 ASR 不要求同版本发布。
- `botified-releases` README 说明各组件独立安装，避免“全家桶”心智。
- release manifest、image label 和 `botified-asr --version` 必须一致。

## 15. 通用 Agent Skill

### 15.1 唯一源码

```text
skills/botified-asr/
├── SKILL.md
├── agents/openai.yaml
├── scripts/botified-asr
└── references/api.md
```

`SKILL.md` frontmatter 只包含：

```yaml
---
name: botified-asr
description: ...
---
```

避免 OpenClaw 或 Codex 私有 frontmatter，保持 AgentSkills 兼容。

### 15.2 Skill 职责

Skill 指导 Agent：

- 从 §10.3 的唯一 `client.env` 使用服务；
- 先检查 ready；
- 判断普通同步还是长音频 async；
- 选择 VAD、diarization 和 rich include；
- 注册、列出、更新和删除人物；
- 显式选择本次 known speaker；
- 轮询 job，处理 terminal failure；
- 输出带 speaker/timestamp 的会议记录；
- 对 Unknown 和低 similarity 保持诚实；
- 不泄漏 token、原始音频或人物声音样本。
- “会议记录”在本产品内只表示带 speaker/timestamp 的转写投影；摘要、结论和行动项由下游 Agent 自行生成，不属于服务或 helper。

### 15.3 Deterministic helper

`scripts/botified-asr` 是 thin curl wrapper，只提供：

```text
health
transcribe
transcribe-long
job-get
job-wait
job-delete
speaker-add
speaker-list
speaker-get
speaker-put
speaker-delete
```

规则：

- 不在 Skill 中实现第二个 ASR client library。
- wrapper 不保存 API Key。
- wrapper 按 §10.3 的白名单规则读取 `client.env`，允许进程环境做显式覆盖。
- helper 启动前检查 `curl`；缺少时返回稳定错误和安装前置条件，不擅自安装。
- `job-wait` 有 timeout 和退避上限。
- 所有服务错误保留稳定 error code。
- 输出 JSON 供 Agent 处理，不替 Agent 生成总结。

### 15.4 独立安装

`install-asr-skill.sh --target <runtime>` 使用同一个 skill tarball；每次必须明确且只接受一个 target，不自动检测后静默选择：

| Runtime | 默认目录 |
|---|---|
| Codex | `~/.codex/skills/botified-asr` |
| OpenClaw | `~/.agents/skills/botified-asr` |
| Botified | `~/.local/share/botified/skills/botified-asr` |

- 同一 tarball 是唯一内容来源。
- target 只接受 `codex`、`openclaw`、`botified`。
- 安装前校验 SHA-256。
- artifact 解包和结构验证在独立 staging 完成；内容安装到版本目录，再以同文件系统临时 symlink + rename 原子切换 runtime discovery 路径。
- 旧版本在切换成功后清理；任一步失败由 trap 保持或恢复原 symlink。普通目录占用目标路径时 fail fast，不递归覆盖用户内容。
- 本地服务安装后的 Skill 自动使用既有 `client.env`；远程模式只把显式 URL/Key 写入同一文件。
- 不把服务 URL 或 token 写入 Skill 内容。

## 16. 测试策略

### 16.1 测试原则

- 测本服务拥有的契约，不复制 FunASR 单元测试。
- 一个行为在最接近实现的层级测试一次。
- API contract 用轻量 fake model adapter；真实模型路径用少量 integration/live smoke。
- 不将几百 MB fixture 提交进 Git。
- 性能和模型质量不放进每次普通 unit test。

### 16.2 Real model integration

使用固定小样本覆盖：

- 中文；
- 英文；
- 粤语；
- 日语；
- 韩语；
- 静音；
- 掌声或笑声；
- 两人对话。

只验证服务集成和输出结构，不把单句逐字结果写成脆弱 snapshot。

### 16.3 长音频和恢复测试

仓库只保存一个 checksum 固定、许可清晰的约 60 秒双人非静音 fixture。测试机使用固定命令重复编码：

```bash
ffmpeg -stream_loop -1 -i two-speaker-60s.flac \
  -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le -t 16384 long.wav
```

脚本断言输出可完整解码、duration 为 `16384 ± 0.1` 秒且文件大小在 500–501 MiB；生成文件不提交 Git。另用同一 source 生成恰好 12 小时、低于 1 GiB 的固定 codec 压缩媒体，并断言实际 PCM sample count 为 12 小时。

模型 cache 预热后使用以下固定资源上限：

- CPU container memory limit 8 GiB；
- CUDA runner GPU memory budget 8 GiB，host container memory limit 8 GiB；
- 相对 warm-ready baseline 的 host peak RSS 增量不超过 1 GiB；
- 按 `processed_audio_secs` 分桶：首个音频小时后任一 30 分钟音频窗口的 RSS median 不得比首个音频小时 median 高 256 MiB，不依赖任务实际 wall time；
- 不含预热 model cache 的 job data 磁盘增量不超过 2 GiB。

以下五个场景分别覆盖对应边界：

1. **500 MiB**：一个 job 完成 upload、decode 和 transcription，验证 byte/storage reservation、RSS/GPU/disk 上限，不注入状态机 fault。
2. **12 小时**：一个压缩媒体 job 只验证实际 PCM duration cap、VAD cache plateau、有界资源和最终结果；不重复承担 crash 测试。
3. **Crash/CAS**：1–5 分钟 fixture 使用 fake/fast adapter 精确覆盖 `.complete` rename 前后、cancel-vs-success、receiving/deleting、shutdown marker 生命周期；另用跨越至少五个 inference unit 的中型真实模型任务做一次进程 `SIGKILL` 和从头恢复。
4. **取消**：中型任务运行后 DELETE；五分钟内进入 cancelled，ffmpeg、partial、attempt artifact 和 reservation 清零，随后第二次 DELETE 删除 job。
5. **公平性**：足够跨越至少五个 inference unit 的中型 job；先用 20 次预热短请求建立 idle p95，再在 long job running 时提交固定 10 秒同步请求，要求在 long job terminal 前完成且延迟不超过 idle p95 三倍加 5 秒。

### 16.4 Skill 测试

同一 skill artifact：

1. helper 全部子命令用 fake HTTP fixture 各验证请求映射、退出码和响应透传；真实服务只保留一条 helper health + short-transcription smoke。
2. Codex、OpenClaw、Botified 的固定受支持版本分别在隔离 HOME 中显式 `--target` 安装同一 checksum artifact。
3. 三个 runtime 各验证实际 discovery，并执行相同的 `health + 固定短转写`。

三次 smoke 验证的是不同 runtime discovery 边界，不复制 async/speaker API 测试，也不维护三份 Skill 内容。

### 16.5 已知人物质量指标

- 使用未参与阈值拟合和 enrollment 的 held-out 中英文 fixture；至少两位已知人物、三位未知人物、两类录音设备。
- 每位已知人物使用 2–5 个 enrollment 样本，另有至少 10 个 held-out utterance；未知人物合计至少 30 个 utterance。
- 已知人物正确命名率至少 90%，未知 utterance 的 false-known assignment 必须为 0；低 margin 保持 `Unknown`。
- 至少一个“两位 known + 一位 Unknown”的中英文混合会议通过完整 VAD -> window embedding -> clustering -> centroid matching -> response projection。
- synthetic vector 只单测 threshold/margin 数学边界；真实 fixture 在 model policy/revision 变化时运行，不在每次普通 CI 重复。

## 17. 性能指标

基准覆盖以下指标，不在没有指定硬件时写死吞吐数字：

- cold start / warm start；
- 10 秒短音频 p50/p95；
- 1 小时会议 RTF；
- 500 MiB 测试总耗时；
- manifest 声明 device 的 peak host/device memory；
- diarization 开关前后成本；
- 1、2、4 并发短请求；
- 长 job 并行时短请求 p95。

首版只测量单 lane 下的排队和吞吐，不用基准数据临时打开共享模型并发。

## 18. 安全和隐私边界

- 默认只监听 loopback。
- 公网部署由反向代理提供 TLS；安装器不自动签发证书。
- 上传文件名不进入文件系统路径。
- ffmpeg 使用无 shell argv。
- SQLite、credential、job 和 speaker 数据目录权限最小化。
- OCI container 使用非 root user、read-only rootfs，不使用 host network，只发布配置的 loopback 端口；仅挂载受控 data、revision-isolated model cache 和只读配置，其中 data root 与 model cache root 必须互不相等、互不嵌套并使用独立持久 mount，服务 secret 由 launcher 注入进程环境。
- 原始音频和人物样本不进入日志、错误、metrics 或 crash report。
- API 不返回 embedding。
- 删除 profile 只立即删除主记录；已入队 job 的私有 embedding snapshot 保留到该 job 删除或 retention 清理，这是可恢复命名语义所需的最短生命周期。
- Skill 不将 token 放入命令历史示例。
- description 是不可信文本，只作为 metadata，不进入 prompt 或模型。
- 发行运行时始终不注册 `/docs` 和 `/openapi.json`；release 离线 OpenAPI artifact 不包含 speaker 数据或其他实例数据。

## 19. 明确非目标

- 修改 Botified Core/Gateway；
- 修改第三方 channel plugin；
- TTS；
- Matrix voice-note 格式；
- streaming websocket；
- 实时电话；
- ASR translation；
- hotword 管理；
- 任意模型选择；
- CT-Punc；
- 声纹模型训练；
- 人脸/视频 speaker identification；
- overlap detection 专用模型；
- 多租户；
- Web UI；
- S3；
- 分布式 worker；
- Redis/Celery；
- resumable upload；
- 自动摘要或会议行动项生成；
- 数据标注和质量管理平台。

## 20. 风险和处理

| 风险 | 处理 |
|---|---|
| 500 MiB 解码后内存膨胀 | streaming ffmpeg + bounded VAD segment |
| 长 job 阻塞语音留言 | bounded batch 后释放 admission |
| speaker 聚类随会议长度退化 | 有界 centroid/history，长会议基准 |
| 已知人物误匹配 | threshold + top-two margin，默认 Unknown |
| 不同麦克风影响声纹 | 多样本平均，真实环境校准 |
| 上游 mutable pipeline 状态 | 独立模型对象和本服务 orchestration，不逐请求改 AutoModel 配置 |
| 上游升级破坏 token/timestamp | 精确 pin + real model smoke |
| 异步结果与文件状态不一致 | receiving/deleting phase + per-attempt artifact + SQLite CAS 对账 |
| 磁盘占满 | SQLite 原子 reservation + free-space floor + terminal retention |
| Skill 三平台漂移 | 单一 tarball 和共同 SKILL.md |
| 安装器破坏宿主环境 | OCI image，不安装 driver/runtime |
