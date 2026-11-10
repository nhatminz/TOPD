# TOPD: Qwen3-4B → Qwen3-1.7B, Competition-MATH, 1 epoch

Triển khai độc lập Section 3 của [TOPD v2](https://arxiv.org/pdf/2606.00305v2),
thích ứng với pipeline CMT trong `BellmanOPD_analysis/`. Không import từ folder
gốc khi chạy và không sửa file của phương pháp gốc. Đây là bản tái lập có các
lựa chọn công khai, chưa phải mã chính thức của tác giả.

**Điểm cần biết:** CMT math launcher trong checkout mặc định **8B → 1.7B-Base,
3 epoch**. Không được so trực tiếp run đó với TOPD **4B → 1.7B, 1 epoch**.
Cần run CMT cùng cấu hình hoặc cung cấp `BASELINE_CONFIG` của run đã khớp.
Xem [audit](docs/BASELINE_AUDIT.md), [phương pháp](docs/TOPD_METHOD.md),
[các giả định](docs/PAPER_AMBIGUITIES.md), [fair comparison](docs/FAIR_COMPARISON.md)
và [kết quả kiểm thử](docs/TEST_RESULTS.md).

## Môi trường

Dùng môi trường train của BellmanOPD_analysis; thêm SciPy để giải OT. Nếu dựng
môi trường mới trên máy train, dùng Python 3.11/3.12 và CUDA/PyTorch tương thích:

```bash
python3 -m pip install -r TOPD/requirements.txt
```

`PYTHON_BIN=/path/to/venv/bin/python` chọn interpreter nếu không dùng `python3`.
Launcher gán lại `PYTHON_EXEC` cho torchrun và khởi động console entry point
vLLM bằng chính interpreter này; không dùng executable `vllm` từ một venv khác
trên `PATH`. Cần cài dependencies vào đúng Python đã chọn.
vLLM cần `>=0.17.1,<0.18`, Transformers `>=4.57,<5`. Production dùng BF16,
full-parameter FSDP và bắt buộc GPU B200, cùng quy tắc resolve path với bản gốc.
Model/data được đọc trực tiếp từ storage đã có trên B200. Kiểm thử CPU dùng model
Qwen3 nhỏ sinh ngẫu nhiên để kiểm thuật toán; không đổi cấu hình production.
Nếu gặp `Engine core initialization failed`, xem [hướng dẫn startup](docs/B200_STARTUP.md)
và lỗi engine gốc trong server log.
Log B200 đã xác nhận `flashinfer-cubin==0.6.6` lệch với
`flashinfer-python==0.6.4`. Sửa trong đúng venv trước khi train/eval:

```bash
PYTHON_BIN="$(command -v python3)" bash TOPD/scripts/fix_flashinfer_b200.sh
```

Script chỉ căn phiên bản cubin theo FlashInfer đang cài rồi kiểm tra import.
Thêm `--dry-run` để xem lệnh. Train/eval phát hiện lệch phiên bản và báo lỗi sớm;
các launcher không tự cài package.

## Chạy train trên GPU 0 và 1

Từ workspace chứa cả hai folder, trong môi trường đã cài dependencies:

```bash
CUDA_VISIBLE_DEVICES=0,1 TRAIN_SEED=42 \
STORAGE_ROOT=/workspace/storage-shared \
bash TOPD/scripts/train_compe_math_4b_17b_1epoch.sh
```

Script tự đếm GPU thật nhìn thấy bởi PyTorch và dùng một FSDP rank/GPU qua
`torch.distributed.run`. Không thay `CUDA_VISIBLE_DEVICES`. Nếu mask yêu cầu
hai GPU nhưng chỉ nhìn thấy một GPU, script dừng trước khi tải model.
Train chạy liên tục hết một epoch, không eval ở step 0, giữa các step hoặc
trong callback cuối train. Sau khi lưu `final/`, chạy eval multiseed riêng.
Quy tắc này cũng áp dụng khi dùng `BASELINE_CONFIG` hoặc resume config cũ.

Các đường dẫn mặc định dưới `STORAGE_ROOT`:

| Asset | Đường dẫn |
|---|---|
| Teacher | `models/Qwen3-4B` |
| Student | `models/Qwen3-1.7B` |
| Train | `nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet` |
| Competition-MATH eval | `nlp/minhpn19/data/competition_math/data/test-00000-of-00001.parquet` |
| MATH-500 / AIME24 / AIME25 | `nlp/minhpn19/data/eval/math500`, `aime24`, `aime25` |
| AMC23 | `nlp/minhpn19/data/amc23/test-00000-of-00001.parquet` |
| GPQA-Diamond (giữ suite gốc) | `nlp/minhpn19/data/GPQA-Diamond/gpqa_diamond.jsonl` |

Không tải model/data và không chuyển checkpoint sang Hugging Face cache. Mọi
asset tương đối được resolve dưới `STORAGE_ROOT=/workspace/storage-shared`
bằng đúng `b200_experiment.config.resolve_runtime_paths` của bản gốc. Model
local thiếu sẽ báo lỗi kèm path cụ thể. Các biến path của BellmanOPD_analysis
được nhận với cùng thứ tự ưu tiên:

- `TEACHER_MODEL` → `TEACHER_MODEL_PATH` → `TEACHER_PATH`.
- `STUDENT_MODEL` → `STUDENT_MODEL_PATH` → `STUDENT_PATH`.
- `TRAIN_DATA` → `TRAIN_DATA_PATH`; `PROMPT_KEY` → `TRAIN_PROMPT_KEY`.
- `TRAIN_DATA_SPLIT`, `TRAIN_PREFER_SOURCE_PROMPT`, `STORAGE_ROOT`.
- `OUTPUT_ROOT`, `RUN_NAME`, `OUTPUT_DIR`, `RESULTS_DIR`, `COMPARISON_NAME`.

Hai model path mặc định đã được bạn xác nhận: `models/Qwen3-4B` và
`models/Qwen3-1.7B`, tức `/workspace/storage-shared/models/Qwen3-4B` và
`/workspace/storage-shared/models/Qwen3-1.7B`. Khi chạy CMT đối chứng, dùng chính
hai path này. Đường dẫn tuyệt đối được giữ nguyên; đường dẫn tương đối dùng
storage root. Cặp 8B/Base trong mặc định CMT lịch sử không phải cặp đối chứng này.

Mặc định preflight hoạt động offline: kiểm architecture/tokenizer và ghi SHA256
của local weights/config/tokenizer. `VERIFY_OFFICIAL_WEIGHTS=true` bật so sánh
thêm với official Hugging Face LFS metadata, cần mạng nhưng vẫn giữ nguyên local
path và không tải weights. Không suy diễn local SHA256 thành bằng chứng official
identity nếu chưa bật kiểm này.

Để kế thừa chính xác cấu hình chung của CMT đối chứng:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
BASELINE_CONFIG=/absolute/path/cmt_run/resolved_config.yaml \
bash TOPD/scripts/train_compe_math_4b_17b_1epoch.sh
```

File đó phải dùng cùng model pair và đúng một epoch. Cấu hình CMT
lịch sử 8B/Base/3 epoch sẽ bị từ chối. Không truyền checkpoint CMT đã train
thành student initialization.

Xem cấu hình mà không tải model hoặc bắt đầu training:

```bash
bash TOPD/scripts/train_compe_math_4b_17b_1epoch.sh --dry-run
```

Kết quả preview: `TOPD/results/launch_configs/<RUN_NAME>.yaml`; đây chỉ là kiểm cấu hình,
không phải preflight xác nhận GPU, dataset, tokenizer hay weights.

## Output và resume

Mặc định: `TOPD/outputs/topd_compe_math_4b_17b_seed42/topd/`.

- `final/`: HF model, tokenizer, generation/config, optimizer và RNG. Checkpoint giữa rollout còn lưu frozen
  main rollout cần cho resume. `checkpoint-000150/` tương tự cho checkpoint giữa chừng.
- `resolved_config.yaml`, `run_metadata.json`, `preflight.json`: cấu hình,
  source hashes, local weight hashes và kiểu verification, số mẫu sau filter, sampler-order hash.
- `comparison_manifest.json`, `cmt_matched_config.yaml`: đối chiếu từng setting
  và config cho run CMT mới nếu chưa cung cấp baseline thật.
- `topd_metrics.jsonl`: OPD/traj/total loss, candidate/OT/acceptance, token counts,
  effective loss scale, gradient norm, thời gian và memory theo từng update.
- `topd_summary.json`: kiểm tra mỗi `(example,response)` đúng một lần, coverage 1.0.
- `metrics.jsonl`, `training_cost*.json*`, `training_sessions.jsonl`, TensorBoard:
  diagnostics chung và chi phí thực đo; không giả định overhead 1.41×.
- `training_eval/`, `eval_history*.jsonl` có thể tồn tại từ run cũ; các lần train
  mới không chạy evaluation callback.

Resume cùng output, seed, GPU layout và config:

```bash
CUDA_VISIBLE_DEVICES=0,1 TRAIN_SEED=42 RESUME_FROM_CHECKPOINT=auto \
bash TOPD/scripts/train_compe_math_4b_17b_1epoch.sh
```

Nếu dùng output khác, thêm cùng `OUTPUT_DIR` đã dùng. Nếu truyền đường dẫn
checkpoint cụ thể, script suy ra output từ thư mục cha. Giữ frozen original
rollout cho phần PPO chưa dùng; future windows được sinh mới ở mỗi update.
Không thay world size khi resume giữa rollout. Run đã hoàn thành một epoch
không được resume để chạy thêm epoch. Rollback checkpoint cũ xóa log/update
chưa commit sau checkpoint, dùng cơ chế rewind có sẵn của pipeline.

## Eval và so sánh

Luồng mặc định là **train xong → eval multiseed checkpoint final**. Ví dụ tại
thư mục `TOPD/`, sau khi chuẩn bị dependencies:

```bash
export CUDA_VISIBLE_DEVICES=0,1 TRAIN_SEED=42
export RUN_NAME=topd_compe_math_4b_17b_seed42_finalonly
bash scripts/train_compe_math_4b_17b_1epoch.sh && \
  bash scripts/eval_multiseed_b200.sh
```

Eval chỉ bắt đầu khi train kết thúc thành công; mặc định eval `final` trên sáu
dataset với ba evaluation seed `42 43 44`. Checkpoint định kỳ vẫn được lưu để
resume; chúng không kích hoạt eval.

Nếu cần eval một seed hoặc so sánh với CMT:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash TOPD/scripts/eval_compe_math_4b_17b_1epoch.sh

bash TOPD/scripts/compare_with_baseline.sh /absolute/path/to/matched_cmt_run
```

Eval dùng `resolved_config.yaml` của run, `final/`, cùng renderer/parser math-verify,
temperature `.7`, top-p `.95`, 7168 token, 8 responses và seed 1234 theo mặc định
launcher/config gốc. Nếu có `BASELINE_CONFIG`, dùng các giá trị thực của file đó.
Suite giữ Competition-MATH, MATH-500, AIME24, AIME25, AMC23 và GPQA-Diamond như gốc.
`TOPD/results/<RUN_NAME>/eval/topd/summary.json` và các file prediction ghi cả Avg@8/Pass@8 từ cùng tám
completion. Có thể đặt `TOPD_RUN_DIR`, `TOPD_CHECKPOINT`, `TOPD_EVAL_OUTPUT`, hoặc truyền `--model /path/checkpoint-...`,
`--output /new/eval_dir`, `--seed 43` cho re-eval riêng. Các option này thuộc
`python -m topd.cli evaluate --help`.

Đánh giá **cùng checkpoint** với ba evaluation seed, trên đủ sáu dataset:

```bash
CUDA_VISIBLE_DEVICES=0,1 TRAIN_SEED=42 \
bash TOPD/scripts/eval_multiseed_b200.sh
```

Mặc định: `final`, evaluation seeds `42 43 44`, mỗi seed sinh tám response/bài;
Avg@8 và Pass@8 dùng chung completions. Không cần eval đơn trước đó.
Kết quả nằm trong run train:
`eval_multiseed/seeds_42_43_44/final/seed_<seed>/`,
`eval_summary_seeds_42_43_44.{json,csv,md}`; thống kê dùng sample std (`ddof=1`).
Chạy lại sẽ dùng kết quả seed đã hoàn tất nếu request không đổi.

Nhận cùng CLI với `BellmanOPD_analysis/scripts/eval_multiseed_b200.sh`:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash TOPD/scripts/eval_multiseed_b200.sh \
  --run-output /absolute/path/to/topd_run/topd \
  --checkpoints final --seeds 42 43 44 --tag eval_3seeds
```

Thêm `--dry-run` để xem kế hoạch từ resolved config/checkpoint đã có mà không
tải model. Multiseed này đo biến thiên khi lấy mẫu eval, không đo biến thiên
của ba lần training. Script eval đơn phía trên vẫn chỉ chạy một eval seed.

Chạy nhiều training seed, chỉ khi muốn chi trả cho ba full run:

```bash
for seed in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0,1 TRAIN_SEED="$seed" \
    bash TOPD/scripts/train_compe_math_4b_17b_1epoch.sh
  CUDA_VISIBLE_DEVICES=0,1 TRAIN_SEED="$seed" \
    bash TOPD/scripts/eval_compe_math_4b_17b_1epoch.sh
done
```

Seed ảnh hưởng thứ tự data, original rollouts và future RNG; evaluation seed
mặc định giữ cố định như comparison. Không đồng nhất ba decoding seeds với ba
training seeds. Chưa có run accuracy thật hoặc thống kê nhiều seed trong folder này.

Sau khi cả ba run và eval hoàn tất, tổng hợp mean ± sample standard deviation:

```bash
cd TOPD
python3 -m topd.cli aggregate-seeds \
  --runs outputs/topd_compe_math_4b_17b_seed42/topd \
         outputs/topd_compe_math_4b_17b_seed43/topd \
         outputs/topd_compe_math_4b_17b_seed44/topd \
  --output outputs/topd_training_seeds_summary.json
```

Lệnh dừng nếu thiếu final checkpoint/eval, trùng training seed hoặc protocol
khác nhau. `compare_with_baseline.sh` tạo cả JSON và bảng Markdown, chỉ đưa
accuracy vào bảng khi tìm thấy evaluation đã chạy ở cả hai phương pháp.

## Các lựa chọn TOPD

| Biến môi trường | Mặc định / ý nghĩa |
|---|---|
| `TOPD_CANDIDATE_FRACTION` | `.02`, top-loss budget cho từng response |
| `TOPD_DIVERGENCE_MODE` | `percentile`; có thể `absolute` |
| `TOPD_OT_QUANTILE` | `.5`, percentile training-only trong global PPO group |
| `TOPD_OT_THRESHOLD` | `0.0`, cutoff nếu dùng `absolute` |
| `TOPD_LAMBDA` | `.01`, nhân loss trajectory đã reduce |
| `TOPD_NORMALIZATION` | `window_mean_sum`; thêm `token_mean`, `sum`, `original_token_mean` |
| `TOPD_OT_SOLVER` | `exact`; có thể `sinkhorn` (xấp xỉ entropy, được log) |
| `TOPD_GENERATION_BATCH_SIZE` | `4` prefixes/lần sinh future |
| `TOPD_LOSS_MICRO_BATCH_SIZE` | `1` future window/lần backward |
| `MICRO_BATCH_SIZE_PER_GPU` | `16`, giữ global PPO batch 64 |
| `SCORE_MICRO_BATCH_SIZE` | `8` |
| `FSDP_TEACHER_CPU_OFFLOAD` | `false`, tùy chọn offload teacher có sẵn trong pipeline |

Các tham số còn lại nằm trong `configs/topd_compe_math.yaml`, bao gồm candidate
budget tuyệt đối, min loss, Sinkhorn tolerance/epsilon và future sampling.
`future_window=50` được khóa cho production; vocabulary Top-K vẫn là 16.
Giảm microbatch/offload khi thiếu memory phải được chọn rõ ràng và áp dụng cho
CMT đối chứng; script không tự đổi precision, batch hiệu dụng hay response length.

## Kiểm thử

```bash
bash TOPD/scripts/smoke_test.sh
```

Script chạy unit/regression tests, hai CPU process với Gloo, rồi smoke offline
với ba synthetic examples, hai original rollouts/example, future K=50, ba updates,
OT, student gradients và checkpoint/resume thật. Kết quả JSON nằm ở
`TOPD/outputs/smoke/smoke_result.json`; chúng không phải kết quả Competition-MATH.

## Path contract khi push lên B200

Xem [B200_PATHS.md](docs/B200_PATHS.md). Chỉ source và config được push; runtime
`outputs/` và `results/` bị gitignore. Script tạo lại resolved config trên B200,
không đọc các path preview từ máy kiểm thử. Cấu trúc tương ứng với bản gốc:

```text
BellmanOPD_analysis/outputs/<RUN_NAME>/cmt_opd/
TOPD/outputs/<RUN_NAME>/topd/
BellmanOPD_analysis/results/<RUN_NAME>/eval/cmt_opd/
TOPD/results/<RUN_NAME>/eval/topd/
```

`training_eval/`, `tensorboard/`, vLLM logs và checkpoints đều nằm dưới run output
của phương pháp; shared model/data assets giữ cùng path trên storage B200.
