Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /Users/abdslm/Desktop/SOLIDER-REID
model: gpt-5.5
provider: openai_http
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019f064f-d4d5-7c21-b183-442f62828047
--------
user
exp363 AG build 第一步: 查 AG-VPReID.VIR 具体下载命令(调研交你, 给可执行命令)。我要在远程 Linux 机器(RTX 4090, /home/afr 338G free, conda env mmpose-abu)下载 AG-VPReID.VIR 数据集(aerial-ground RGB-IR video ReID, 1837 ID/4861 tracklets/124855 frames, 论文 arxiv 2507.17995)。问: (a)AG-VPReID.VIR 的 **GitHub repo 精确 URL** + README 给的 **Google Drive 下载链接/file ID** + **gdown 具体命令**(gdown --id XXX 或 gdown --folder)? 若 repo 还没公开数据, 说实话。 (b)数据集解压后**目录结构**(train/test/query/gallery, RGB vs IR, video frames/tracklet 怎么组织, label 格式)? (c)**DINOv3 ViT-B/16 权重下载**: HuggingFace repo id + load 方式(timm.create_model 还是 torch.hub 还是 transformers AutoModel)? DINOv3 若需申请, 给 **DINOv2-reg ViT-B/14 的 HF id + timm 名**作为立即可用退路。 (d)许可/注意。给一串可直接跑的命令(git clone + gdown/wget + pip install gdown + HF snapshot_download)。务实中文。
codex
我先查原始论文、代码仓库和权重发布页，优先用官方 GitHub / HuggingFace / arXiv。这个问题里很多细节可能最近才公开，不能凭记忆给命令。
web search: 
web search: AG-VPReID.VIR GitHub AG-VPReID VIR Google Drive
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: https://raw.githubusercontent.com/agvpreid25/AG-VPReID.VIR/main/README.md
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: "AG-VPReID.VIR" "train" "query" "gallery"
codex
官方 README 只给了一个 Drive 文件夹链接，没有在页面上写单文件 ID 或目录树。我继续直接查公开 Drive 页面和论文 PDF，看能不能拿到压缩包名、split 组织方式和 label 格式，避免把 AG-VPReID 的结构误套到 AG-VPReID.VIR。
exec
/bin/zsh -lc "curl -L 'https://drive.google.com/drive/folders/1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7?usp=share_link' | sed -n '1,120p'" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
curl: (7) Failed to connect to 127.0.0.1 port 7890 after 0 ms: Couldn't connect to server

web search: 
web search: https://drive.google.com/drive/folders/1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7?usp=share_link
web search: 
web search: 
web search: 
web search: 
web search: 
web search: site:github.com/agvpreid25/AG-VPReID.VIR "Dataset Download"
web search: 
web search: AG-VPReID.VIR github
web search: 
web search: https://arxiv.org/abs/2507.17995
web search: 
web search: https://arxiv.org/pdf/2507.17995
web search: 
web search: "AG-VPReID.VIR" pdf
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: https://drive.google.com/drive/mobile/folders/1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7?usp=share_link
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR/releases/tag/v1.0
web search: 
web search: github agvpreid25 AG-VPReID.VIR releases v1.0
web search: 
web search: https://api.github.com/repos/agvpreid25/AG-VPReID.VIR/contents
web search: 
web search: api.github.com/repos/agvpreid25/AG-VPReID.VIR/contents
web search: 
web search: raw.githubusercontent.com/agvpreid25/AG-VPReID.VIR/main/README.md
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR/blob/main/README.md
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: DINOv3 Hugging Face ViT-B/16 model id timm create_model dinov3 vitb16
web search: 
web search: facebookresearch dinov3 GitHub
web search: 
web search: DINOv3 Meta official GitHub Hugging Face
web search: 
web search: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
web search: 
web search: DINOv2 with registers ViT-B/14 Hugging Face id timm model name
web search: 
web search: https://huggingface.co/facebook/dinov2-with-registers-vitb14
web search: 
web search: site:huggingface.co/facebook/dinov2-with-registers-vitb14
web search: 
web search: huggingface dinov2 registers vitb14 facebook
web search: 
web search: https://huggingface.co/facebook/dinov2-with-registers-base
web search: 
web search: timm dinov2_vitb14_reg pretrained model name
web search: 
web search: "vit_base_patch14_reg4_dinov2.lvd142m"
web search: 
web search: github timm dinov2_reg model names
web search: 
web search: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
web search: 
web search: 'dinov2' in https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
web search: 
web search: https://github.com/facebookresearch/dinov3
web search: 
web search: 'timm' in https://github.com/facebookresearch/dinov3
web search: 
web search: https://drive.google.com/embeddedfolderview?id=1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7#list
web search: 
web search: https://drive.google.com/embeddedfolderview?id=1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7
codex
DINOv3 部分已经查到官方路线：HF Transformers 和本地 torch.hub 都支持，但 ViT-B/16 权重是 gated，需要先在 HF/Meta 页面同意条款。我还在尝试拿 Drive 文件夹内部列表；如果公开页面不给静态目录，我会把“不在 README 公布”的边界写清楚，并给远程可执行的下载后自检命令。
codex
我会用 Browser 插件的节点运行器打开公开 Drive 页面，看看渲染后的页面是否能列出文件名；这不改动仓库，只是补下载命令需要的证据。
exec
/bin/zsh -lc "sed -n '1,220p' /Users/abdslm/.codex/plugins/cache/openai-bundled/browser/26.623.30605/skills/control-in-app-browser/SKILL.md" in /Users/abdslm/Desktop/SOLIDER-REID
 succeeded in 0ms:
---
name: control-in-app-browser
description: "Control the in-app Browser. Use to open, navigate, inspect, test, click, type, screenshot, or verify local targets such as localhost, 127.0.0.1, ::1, file://, the current in-app browser tab, and websites shown side by side inside Codex."
---

# Browser

Use this skill for browser automation tasks such as inspecting pages, navigating, testing local apps, clicking, typing, taking screenshots, and reading visible page state.

If this plugin is listed as available in the session, treat that as mandatory reading before browser work. Open and follow this skill before saying that Browser is unavailable and before falling back to standalone Playwright or Computer Use.

Do not skip this skill just because Computer Use MCP tool calls are directly visible or appear easier to invoke. The presence of Computer Use tools is not evidence that Computer Use is the preferred browser surface.

## Setup Documentation

Use `await agent.documentation.get("bootstrap-troubleshooting")` when browser setup succeeds but discovery or selection fails.

## Bootstrap

These setup details are internal. User-facing progress updates should be less technical in nature. Never mention `Node REPL`, `node_repl`, `REPL`, JavaScript sessions, module exports, reading documentation, or loading instructions unless a user is asking for that exact information. If setup or recovery is needed, describe it naturally as connecting to the browser or retrying the browser connection.

The `browser-client` module is the core entry point for browser use, and is available under `scripts/browser-client.mjs` in this plugin's root directory. ALWAYS import it using an absolute path. IMPORTANT: If this path cannot be found, stop and report that this plugin is missing `scripts/browser-client.mjs`. NEVER use the built in `browser-client` library.

Run browser setup code through the Node REPL `js` tool. In this environment the callable tool id typically appears as `mcp__node_repl__js`. If it is not already available, use tool discovery for `node_repl js` without setting a result limit. You need the `js` execution tool: `js_reset` only clears state, and `js_add_node_module_dir` only changes package resolution. Do not call either helper while trying to expose `js`. If `js` is still not available, search again for `node_repl js` with `limit: 10`.

Initialize the runtime once per fresh Node session, select the in-app browser, and immediately read its complete documentation:

```js
if (globalThis.agent?.browsers == null) {
  const { setupBrowserRuntime } = await import("<plugin root>/scripts/browser-client.mjs");
  await setupBrowserRuntime({ globals: globalThis });
}
globalThis.browser = await agent.browsers.get("iab");
nodeRepl.write(await browser.documentation());
```

If setup succeeds but browser discovery or selection fails, read `await agent.documentation.get("bootstrap-troubleshooting")` before resetting the JavaScript session or trying another browser-control mechanism.

Use the browser bound to `browser` for tasks in this skill. Do not switch to another browser-control mechanism to work around authentication. If sign-in blocks requested navigation and the browser documentation does not provide a supported authentication flow, ask the user to sign in in the in-app browser and tell you when it is ready.

The ability to interact directly with the browser is exposed through the `browser-client` runtime via the `agent.browsers.*` API. Before trying to interact with it, you MUST emit and read the complete documentation returned by `await browser.documentation()` in one go. For the initial documentation read, run the exact direct call `nodeRepl.write(await browser.documentation());` shown above. Do not assign the documentation to a variable, inspect its length, slice it, truncate it, summarize it, or emit only an excerpt. Do not proactively split the documentation into pages or chunks. Only if the tool output itself explicitly reports that it was truncated may you emit and read smaller chunks until you have read the documentation in its entirety.

Only the Node REPL `js` tool (`mcp__node_repl__js`) can be used to control the in-app browser. Do not use external MCP browser-control tools, separate browser automation servers, or other browser skills for this surface. References to Playwright mean the in-skill `tab.playwright` API after browser-client setup.

mcp: node_repl/js started
mcp: node_repl/js (failed)
tool call error: tool call failed for `node_repl/js`

Caused by:
    Mcp error: -32602: js: codex/sandbox-state-meta: sandboxCwd must be an absolute file URI: relative URL without a base
web search: 
web search: "AG-VPReID" "AG-VPReID.VIR"
web search: 
web search: https://arxiv.org/abs/2507.17995
web search: 
web search: https://arxiv.org/html/2507.17995
web search: 
web search: https://arxiv.org/html/2507.17995
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR/releases
web search: 
web search: https://api.github.com/repos/agvpreid25/AG-VPReID.VIR/releases
web search: 
web search: https://api.github.com/repos/agvpreid25/AG-VPReID.VIR/releases
web search: 
web search: https://github.com/agvpreid25/AG-VPReID.VIR/issues
web search: 
web search: site:github.com/agvpreid25/AG-VPReID.VIR/issues AG-VPReID.VIR
web search: 
web search: https://github.com/agvpreid25/AG-VPReID-Net
web search: 
web search: "AG-VPReID.VIR" "Download Link" "train"
web search: 
web search: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
web search: 
web search: https://github.com/facebookresearch/dinov3
web search: 
web search: "vit_base_patch16_224.dinov3"
web search: 
web search: site:github.com/huggingface/pytorch-image-models "dinov3"
web search: 
web search: site:huggingface.co/timm dinov3 vit_base patch16
web search: 
web search: https://huggingface.co/facebook/dinov2-base
web search: 
web search: https://huggingface.co/facebook/dinov2-with-registers-base
web search: 
web search: https://github.com/facebookresearch/dinov2
web search: 
web search: https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m
web search: 
web search: https://github.com/facebookresearch/dinov3
web search: 
web search: https://github.com/huggingface/pytorch-image-models/
web search: 
web search: https://huggingface.co/collections/timm/timm-dinov3-68cb08bb0bee365973d52a4d
codex
查到的结论如下。最关键的一点：**AG-VPReID.VIR 的 README 公开了 Google Drive 文件夹，但没有公开 on-disk 目录结构 / dataloader / split 文件说明；GitHub 仓库当前只有 `assets/` 和 `README.md`，没有代码。**

**(a) AG-VPReID.VIR 下载**
- GitHub 精确 URL: https://github.com/agvpreid25/AG-VPReID.VIR
- README 数据集链接: https://drive.google.com/drive/folders/1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7?usp=share_link
- Google Drive folder id: `1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7`
- 可跑命令:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmpose-abu

mkdir -p /home/afr/code /home/afr/datasets/AG-VPReID.VIR /home/afr/weights
cd /home/afr/code
git clone https://github.com/agvpreid25/AG-VPReID.VIR.git

pip install -U gdown huggingface_hub transformers timm safetensors

cd /home/afr/datasets/AG-VPReID.VIR
gdown --folder 'https://drive.google.com/drive/folders/1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7?usp=share_link' -O . --remaining-ok
du -sh .
find . -maxdepth 3 -type f | sed -n '1,120p'
```

如果 `gdown --folder URL` 失败，再试旧式 ID 命令：

```bash
gdown --folder --id 1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7 -O /home/afr/datasets/AG-VPReID.VIR --remaining-ok
```

**(b) 解压后目录结构**
公开 README / 论文没有给出精确文件树、label 文件格式或 frame 命名规则。不能诚实地说它一定是 `train/query/gallery` 或某种 Market-1501 风格结构。

论文只明确了协议统计：
- train: `326 IDs / 978 tracklets / 24,793 frames`
- test 分为 `Ground→Ground / Aerial→Aerial / Ground→Aerial / Aerial→Ground`
- 每个协议有 `V2I` 和 `I2V`
- 采集源包括 `UAV RGB`, `UAV IR/NIR`, `CCTV RGB`, `CCTV IR/NIR`, `wearable RGB`

下载后先跑这个看真实结构：

```bash
cd /home/afr/datasets/AG-VPReID.VIR

find . -maxdepth 4 -type d | sort | sed -n '1,200p'
find . -maxdepth 5 -type f \( -name '*.txt' -o -name '*.json' -o -name '*.csv' -o -name '*.mat' -o -name '*.pkl' -o -name '*.npy' \) | sort
find . -maxdepth 2 -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.rar' -o -name '*.7z' \) | sort
```

解压模板：

```bash
cd /home/afr/datasets/AG-VPReID.VIR
shopt -s nullglob

for f in *.zip; do unzip -n "$f" -d "${f%.zip}"; done
for f in *.tar; do tar -xf "$f"; done
for f in *.tar.gz *.tgz; do tar -xzf "$f"; done
for f in *.7z; do 7z x -y "$f" -o"${f%.7z}"; done
for f in *.rar; do unrar x -o- "$f" "${f%.rar}/"; done
```

**(c) DINOv3 / DINOv2-reg**
DINOv3 ViT-B/16 官方 HF repo id:

```text
facebook/dinov3-vitb16-pretrain-lvd1689m
```

官方 HF 用法是 `transformers AutoImageProcessor + AutoModel`，但它是 gated，需要登录 HF 并同意共享联系信息后才能下载。

```bash
huggingface-cli login

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
    local_dir="/home/afr/weights/dinov3-vitb16-pretrain-lvd1689m",
)
PY
```

加载：

```python
from transformers import AutoImageProcessor, AutoModel

name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(name)
model = AutoModel.from_pretrained(name, device_map="auto")
```

`timm` 也有 DINOv3 ViT-B/16，可直接这样试：

```python
import timm
model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
```

HF Hub 形式：

```python
import timm
model = timm.create_model("hf_hub:timm/vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
```

如果 DINOv3 gated 卡住，立即可用退路：

```text
HF id: facebook/dinov2-with-registers-base
timm name: vit_base_patch14_reg4_dinov2.lvd142m
```

```python
import timm
model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0)
```

或：

```python
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
model = AutoModel.from_pretrained("facebook/dinov2-with-registers-base")
```

**(d) 许可 / 注意**
- AG-VPReID.VIR 仓库没有看到 LICENSE；只按研究用途处理，引用 IJCB 2025 论文。
- 论文写了 IRB 审批、参与者同意、face deface 匿名化，但仍是 surveillance / ReID 数据，别外传原始数据。
- DINOv3 是 `dinov3-license`，官方 HF 需要同意条款；DINOv2-reg HF/timm 是 Apache-2.0 页面标注，更适合作为马上能跑的退路。

来源：GitHub README、Drive 链接、arXiv HTML、DINOv3 HF/GitHub、timm HF 页面。
tokens used
202,611
查到的结论如下。最关键的一点：**AG-VPReID.VIR 的 README 公开了 Google Drive 文件夹，但没有公开 on-disk 目录结构 / dataloader / split 文件说明；GitHub 仓库当前只有 `assets/` 和 `README.md`，没有代码。**

**(a) AG-VPReID.VIR 下载**
- GitHub 精确 URL: https://github.com/agvpreid25/AG-VPReID.VIR
- README 数据集链接: https://drive.google.com/drive/folders/1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7?usp=share_link
- Google Drive folder id: `1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7`
- 可跑命令:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmpose-abu

mkdir -p /home/afr/code /home/afr/datasets/AG-VPReID.VIR /home/afr/weights
cd /home/afr/code
git clone https://github.com/agvpreid25/AG-VPReID.VIR.git

pip install -U gdown huggingface_hub transformers timm safetensors

cd /home/afr/datasets/AG-VPReID.VIR
gdown --folder 'https://drive.google.com/drive/folders/1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7?usp=share_link' -O . --remaining-ok
du -sh .
find . -maxdepth 3 -type f | sed -n '1,120p'
```

如果 `gdown --folder URL` 失败，再试旧式 ID 命令：

```bash
gdown --folder --id 1Iy814PqWjwIZcv6CZpieFju-Dop9Y2G7 -O /home/afr/datasets/AG-VPReID.VIR --remaining-ok
```

**(b) 解压后目录结构**
公开 README / 论文没有给出精确文件树、label 文件格式或 frame 命名规则。不能诚实地说它一定是 `train/query/gallery` 或某种 Market-1501 风格结构。

论文只明确了协议统计：
- train: `326 IDs / 978 tracklets / 24,793 frames`
- test 分为 `Ground→Ground / Aerial→Aerial / Ground→Aerial / Aerial→Ground`
- 每个协议有 `V2I` 和 `I2V`
- 采集源包括 `UAV RGB`, `UAV IR/NIR`, `CCTV RGB`, `CCTV IR/NIR`, `wearable RGB`

下载后先跑这个看真实结构：

```bash
cd /home/afr/datasets/AG-VPReID.VIR

find . -maxdepth 4 -type d | sort | sed -n '1,200p'
find . -maxdepth 5 -type f \( -name '*.txt' -o -name '*.json' -o -name '*.csv' -o -name '*.mat' -o -name '*.pkl' -o -name '*.npy' \) | sort
find . -maxdepth 2 -type f \( -name '*.zip' -o -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.rar' -o -name '*.7z' \) | sort
```

解压模板：

```bash
cd /home/afr/datasets/AG-VPReID.VIR
shopt -s nullglob

for f in *.zip; do unzip -n "$f" -d "${f%.zip}"; done
for f in *.tar; do tar -xf "$f"; done
for f in *.tar.gz *.tgz; do tar -xzf "$f"; done
for f in *.7z; do 7z x -y "$f" -o"${f%.7z}"; done
for f in *.rar; do unrar x -o- "$f" "${f%.rar}/"; done
```

**(c) DINOv3 / DINOv2-reg**
DINOv3 ViT-B/16 官方 HF repo id:

```text
facebook/dinov3-vitb16-pretrain-lvd1689m
```

官方 HF 用法是 `transformers AutoImageProcessor + AutoModel`，但它是 gated，需要登录 HF 并同意共享联系信息后才能下载。

```bash
huggingface-cli login

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
    local_dir="/home/afr/weights/dinov3-vitb16-pretrain-lvd1689m",
)
PY
```

加载：

```python
from transformers import AutoImageProcessor, AutoModel

name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(name)
model = AutoModel.from_pretrained(name, device_map="auto")
```

`timm` 也有 DINOv3 ViT-B/16，可直接这样试：

```python
import timm
model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
```

HF Hub 形式：

```python
import timm
model = timm.create_model("hf_hub:timm/vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
```

如果 DINOv3 gated 卡住，立即可用退路：

```text
HF id: facebook/dinov2-with-registers-base
timm name: vit_base_patch14_reg4_dinov2.lvd142m
```

```python
import timm
model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0)
```

或：

```python
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
model = AutoModel.from_pretrained("facebook/dinov2-with-registers-base")
```

**(d) 许可 / 注意**
- AG-VPReID.VIR 仓库没有看到 LICENSE；只按研究用途处理，引用 IJCB 2025 论文。
- 论文写了 IRB 审批、参与者同意、face deface 匿名化，但仍是 surveillance / ReID 数据，别外传原始数据。
- DINOv3 是 `dinov3-license`，官方 HF 需要同意条款；DINOv2-reg HF/timm 是 Apache-2.0 页面标注，更适合作为马上能跑的退路。

来源：GitHub README、Drive 链接、arXiv HTML、DINOv3 HF/GitHub、timm HF 页面。
