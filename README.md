# SAGO Web — Python 后端 + 浏览器 UI

> **v2.0 source release:** the UI defaults to English; use the top-right button to switch between English and Chinese. Model checkpoints, local environments, scenes, and generated outputs are not included. See [models/README.md](models/README.md) for checkpoint setup and [CHANGELOG.md](CHANGELOG.md) for release highlights.

用 Web 工作台替代 splatviz / ImGui 窗口。3DGS 场景仍由原来的 Python/CUDA 渲染器渲染，SAM2 Hiera Large、SAM3 / SAM3.1、memory、mask 视锥筛选及 Virtual Drone 都运行在 Python 后端。浏览器显示后端返回的图像并发送交互请求，无需 WebGPU，也不加载 ONNX 模型。

所有工程源码、前端文件和 CUDA 扩展源码都在本目录内；运行时不向上一级源码目录添加 `sys.path`，不通过符号链接引用外部源码，也不修改原 SAGO。模型权重和场景作为外部数据通过路径读取。

## 本机直接启动

已有的 `TrackAnyGaussian` 环境已包含 PyTorch、CUDA 扩展和 Flask，可直接运行：

```bash
cd /home/pc/Desktop/SAGO/SAGO_web
/home/pc/anaconda3/envs/TrackAnyGaussian/bin/python run.py \
  --checkpoint /home/pc/Desktop/SAGO/sago_core/sam2/checkpoints/sam2_hiera_large.pt
```

打开 **http://127.0.0.1:8080**。也可以用 `--port 8081` 更改端口。

左侧“打开场景…”调用本机系统文件选择框，直接读取选中的 PLY，不复制大文件（Linux 桌面需 `zenity` 或 `kdialog`）。无桌面环境时自动使用浏览器文件选择；系统框打开失败后也提供浏览器选择入口。

也可把一个 PLY 拖入中央画面。浏览器选择和拖入的文件通过本机 HTTP 上传到 `outputs/uploads/<内容SHA256>/`，显示传输进度，同内容同名文件复用缓存；上传限制约 2 GB，更大文件请使用系统选择框或路径。浏览器无法直接提供文件的完整原始路径，因此上传的独立 PLY 不附带其所在目录的 `cameras.json`；需要原始相机时请用系统文件选择或场景目录路径。

成功加载的最近 8 个场景保存在当前浏览器中，可一键重开、逐条移除或清空。清除记录不删除原文件或上传缓存。

也可展开“手动输入路径”，输入已有场景，例如：

```text
/home/pc/gaussian-splatting/output/for_seg1
```

支持直接指定标准 3DGS `.ply`，或含 `point_cloud/iteration_*/point_cloud.ply` 的训练输出目录，优先选择最新迭代。普通 XYZ 点云不含高斯参数，不能作为 3DGS 场景加载。

若在场景附近找到 `cameras.json`，自动使用第一个原始视角，并提供视角选择器。否则按高斯中心和分布估计初始相机。

## SAM2 / SAM3 / SAM3.1

右侧“分割模型”默认 SAM2，可切换 SAM3 或 SAM3.1。SAM3 系列使用本机已有权重，不自动下载权重；SAM2 仍使用原 CrossMatch 和记忆跟踪接口。

| 模型 | 输入 | 二维候选 | 三维跟踪 |
| --- | --- | --- | --- |
| SAM2 | 文本（GroundingDINO）、前景/背景点、框 | 每个检测目标四级 mask | 原 SAM2 memory |
| SAM3 | 文本、点或框 | 按置信度排序的实例 mask | 官方 SAM3 video predictor |
| SAM3.1 | 文本、点或框 | 按置信度排序的实例 mask | 官方 Object Multiplex predictor |

进入分割提示后，可以只输入 `yellow rubber duck`、`blue toy` 等简短描述，停止输入约 450ms 后自动分割。建议用英文短语。文本或框检出多个物体时，可用前景/背景点筛选目标实例；仅点提示时调用官方实例交互接口。选择目标候选后，可应用单视角或执行三维分割。切换模型、修改文本或几何提示后，结果自动更新。未检测到目标时会提示修改输入，不使用旧结果。

本机自动发现的权重（只读）：

- SAM3：`/home/pc/haptic/sam-3d-objects/sam3_ckpts/sam3/sam3.pt`
- SAM3.1：`/home/pc/haptic/sam-3d-objects/sam3_ckpts/sam3.1/sam3.1_multiplex.pt`

也可将权重放在本工程 `models/sam3.pt`、`models/sam3.1_multiplex.pt`（优先于上述路径），或用 `--sam3-checkpoint`、`--sam31-checkpoint` 指定。环境变量为 `SAGO_SAM3_CHECKPOINT`、`SAGO_SAM31_CHECKPOINT`。

SAM3 系列通过独立 Python 子进程推理，避免升级原渲染环境的 PyTorch/CUDA。当前 `.venv-sam3` 已基于本机 `sam3d-objects` 环境建立，仅在工程内写入文件，实测 PyTorch 2.5.1+cu121；启动方式不变。迁移到其他机器时，在工程内建立独立环境，安装匹配的 CUDA PyTorch/torchvision 和 `requirements-sam3.txt`，然后通过 `--sam3-python /path/to/python` 或 `SAGO_SAM3_PYTHON` 指定（默认 `.venv-sam3/bin/python`）。例如已有兼容 CUDA 环境时：

```bash
/path/to/compatible/python -m venv --system-site-packages .venv-sam3
.venv-sam3/bin/python -m pip install -r requirements-sam3.txt
```

官方源码完整保存在 `vendor/sam3`，版本与兼容性补丁记录于 `vendor/SAM3_PROVENANCE.md`。关闭/切换模型会释放对应子进程；错误日志位于 `outputs/sam3-worker.log`。SAM3.1 使用其专用 multiplex 架构和原始权重；未转换权重，也没有将其当作 SAM3/SAM2 权重加载。

三维阶段使用常驻增量跟踪：首视角只建立一次种子记忆，复制为左右两个独立分支，每步只编码和处理新渲染的视角，不重放历史帧。每个新视角先在状态副本上预测，三维筛选接受后提交该方向的记忆；拒绝或推理失败时丢弃副本。取消和异常退出都会清理本次航线状态。首视角选中实例的固定 ID 输出给视锥筛选；目标丢失时输出空 mask，不改选其他同类实例。

官方检测器的未来帧预取已关闭，预留的帧槽不会被提前编码。一次完整六视角三维分割实际编码 7 张图像（首视角 1 张＋新视角 6 张），`tracking_stats` 报告种子建立次数、新视角数、实际图像编码数及各方向接受/拒绝次数。SAM3 系列仍通过 PNG 文件传递图像，二维预览的会话和特征缓存策略保持原样；本次仅优化三维阶段的历史重放。分割用时包含预处理、推理、结果准备及跨进程传输，不含首次模型加载和预览图保存时间。

### 三点地面标定

左侧“地面标定”提供“三点确定地面”：固定当前画面后，依次点击同一地面/桌面上三个清晰、分散且不共线的位置。页面显示编号、三角形和地面向上箭头；可撤回一点、重新选点、反转方向或取消。确认后点击“应用地面”，上方向 X/Y/Z 自动更新。

三点由该画面的有效渲染深度反投影到世界坐标，平面满足 `n·x+d=0`，单位法线为 `normalize((P2-P1)×(P3-P1))`。默认法线指向相机所在的半空间；若方向不符合场景，可在应用前或应用后点击“反转向上”。“地面参数”显示三个三维坐标、单位向量和平面方程。无有效深度、重复或接近共线的点会被拒绝，已有有效选点保留。

标定不加载 SAM。应用后相机交互和 Virtual Drone 使用该地面向上方向，切换原始场景视角也会保留；重新加载场景会自动读取已保存的标定。应用会重新计算球面相机参数以保持相机位置和观察中心，极点附近按现有 ±85° 俯仰范围限制。手动修改为不同的上方向会清除当前会话标定，但保留磁盘知识，重新加载场景时恢复。地面参数自动写入场景对应的 `SAGO_knowledge.json`，不修改 PLY 或原始相机文件。

点击“应用地面”确认后立即保存；对已应用地面点击“反转向上”也会同步保存。选点、预览、应用前反转或取消不会覆盖旧知识。标准目录 `场景/point_cloud/iteration_*/point_cloud.ply` 或 `场景/point_cloud/point_cloud.ply` 对应 `场景/SAGO_knowledge.json`；独立 PLY 对应其所在目录下的同名 JSON。浏览器上传文件的知识保存于服务端上传目录。

JSON 使用 `scenes[PLY相对路径]` 区分同目录中的不同 PLY，每条记录保存 `ground_plane`（`points`、`normal`、`offset`、`center`）、`up_vector` 和 `updated_at`，保留文件中已有的其他字段。加载场景时按实际 PLY 路径恢复并显示读取状态，无需重新选点；重启服务后仍有效。保存使用临时文件加原子替换，失败时保留旧文件和待确认选点并提示原因。损坏或不一致的知识记录不会被自动应用，也不会静默覆盖。

### SAM2 文本定位

SAM2 的文本输入由 GroundingDINO Swin-T 定位，再将检测框交给原 SAM2 CrossMatch 生成 mask。检测框按置信度排序，经 0.35 分数阈值和 0.5 IoU 去重后最多保留 8 个，每个目标展示四级候选及检测置信度。前景点筛选包含该点的目标，手动画框按检测中心限定范围并裁剪检测框；前景和背景点同时作为 SAM2 提示细化轮廓。仅点或框提示仍走原流程，不加载 GroundingDINO。

首次使用 SAM2 文本提示时单独弹框加载 GroundingDINO，后续复用；分割计时包含文本定位和 SAM2 编码/解码，不含权重加载。选择任意目标/层级后，三维巡航使用该候选对应的 prediction、pointer 和 object score；GroundingDINO 只用于首视角定位，不在每个巡航视角重新检测。切换到 SAM3 系列会释放 SAM2 和检测模型。

源码已包含在本工程内，无需访问 SAGO 源目录。开发机已配置下列本地权重；GitHub 源码发布不包含权重，首次部署请按 [models/README.md](models/README.md) 下载并配置：

- `vendor/GroundingDINO/groundingdino`：参考原 SAGO 中的 GroundingDINO 实现。
- `models/groundingdino_swint_ogc.pth`：本地检测权重。
- `models/bert-base-uncased/`：本地 BERT 模型和 tokenizer。

可用 `SAGO_GROUNDING_CHECKPOINT` 和 `SAGO_GROUNDING_BERT` 覆盖权重位置。依赖包含在 `requirements.txt`；运行环境已验证 transformers 4.56.2。来源、许可和兼容性补丁见 `vendor/GROUNDINGDINO_PROVENANCE.md`。

## 使用流程

1. 加载场景。拖动旋转，Shift 拖动或右键拖动平移，滚轮缩放；松开拖动后由后端更新图像。可以修改上方向、观察中心、FOV 和渲染长边分辨率。中央画面按视口宽高比渲染并铺满，支持长方形窗口；调整浏览器大小后会自动刷新，固定提示期间保持捕获图像的比例与坐标。
2. 让目标完整可见，按 M 或勾选“进入分割提示”。
3. 左键添加前景点，右键添加背景点，或切换框提示并拖框。点与框可以组合。
4. 选择分割模型并添加文本、点或框提示。添加点或框后自动分割，文本停止输入约 450ms 后自动分割；选择完整覆盖目标的候选（SAM2 每目标四级 mask，SAM3 系列为不同实例）。切换模型时立即加载，独立提示框实时显示等待时间，完成后显示本次模型加载用时并在 1.5 秒后自动关闭，失败时保留错误提示；默认使用 SAM2，在首次勾选“进入分割提示”（或按 M）时加载。右侧“分割用时”显示所选模型、三维筛选/细化和合计的后端用时，CUDA 同步后计时，不含浏览器网络传输和首次模型加载；模型加载用时只在独立提示框显示。SAM3 系列的计时范围见上节。任务进行时还会显示页面等待用时。
5. 边缘扩展默认 2 px。点击“应用单视角选区”，或使用“三维分割”（快捷键 P，文本输入时不触发）。后者按 +45°、-45°、+90°、-90°、+135°、-135° 顺序扫描，仰角 30°，调用原 SAGO 视锥筛选和 80% 初始覆盖检查。s0→s1 仅使用首视角 mask 内有效深度反投影的表面点均值作为中心；完成 s1 后，每个后续视角均使用上一步保留的前景高斯位置均值作为中心，并渲染最新前景。若首视角没有有效表面深度则报错，不回退到高斯均值。支持替换、增加、减去及取交集。
6. 取消按 M 或勾选“进入分割提示”后旋转检查；“全部 / 此轮前景 / 此轮背景”切换三维显示。给选区命名并保存为物体，可隐藏已完成物体后继续分割。支持撤销和重做，历史保留最近 12 步。
7. 导出当前选区 PLY、保持原高斯顺序的布尔 `.npy` mask、当前视角 PNG 或预览中的二维 mask PNG。导出保存在本工程 `outputs/` 并下载到浏览器。

三维分割时显示当前 V-Drone 的分割叠加图，并在右侧按 D0/D1 与 ±45°、±90°、±135° 保留六个视角缩略图。点击可放大对照实际跟踪输入的原始渲染与分割叠加图，并下载原图、叠加图和二值 mask。图像保存于 `outputs/drone-previews/`，保存当前视角时不额外渲染或推理；PNG 保存时间单独统计，不计入分割计算时间。每次新巡航或成功加载新场景时清空界面中的上一组图；取消巡航后保留已完成视角，但不提交三维选区。

巡航可取消；已提交的 GPU 运算完成后响应取消，不提交中间选区。模型加载和单次模型推理不能立即中断。相机、物体或场景改变后，旧固定画面失效，需要重新捕获。

## 在另一台机器安装

推荐 Linux + NVIDIA GPU，Python 3.10，CUDA 版本与 PyTorch / CUDA toolkit 一致。先按显卡环境安装相容的 PyTorch 与 torchvision，再执行：

```bash
python -m pip install -r requirements.txt
bash scripts/install-cuda.sh
mkdir -p models
# 将原版 sam2_hiera_large.pt 放到 models/，或通过 --checkpoint 指定其路径
python run.py
```

也可以设置环境变量 `SAGO_CHECKPOINT`。权重未包含在代码工程里，不会自动下载。`vendor/` 包含四个原项目 CUDA 扩展的构建源码及 GLM，构建不需要上一级 SAGO 源码。已有环境中安装了兼容扩展时无需重新编译。

## 工程结构

```text
run.py                       启动入口，固定工作目录以收拢旧代码的 debug 输出
frontend/                    浏览器工作台，无 Node 构建步骤及 CDN 依赖
backend/server.py            Flask HTTP 服务、单 GPU 任务队列、进度、取消和下载
backend/engine.py            场景/相机/提示/选区/物体/历史/导出的 UI 适配
backend/camera_math.py       原相机数学函数，去掉 ImGui 导入链
sago_core/                   原分割、SAM2 跟踪和几何代码的本地副本
gs/                          原 3DGS 场景对象及 CUDA 渲染接口的本地副本
vendor/                      CUDA rasterizer 和 simple-knn 构建源码
scripts/gpu-smoke.py         可选真实 GPU 集成检查
tests/                       HTTP 与浏览器测试
```

复制的 SAGO 核心代码改动包括：将相机数学导入指向本地无 UI 模块、去掉 CrossMatch 中未使用的 ptflops 调试代码、将旧 2DGS 导入延后。渲染适配使用真实 inverse-view camera center，修复原 CustomCam 中由投影矩阵推导中心导致的球谐视角错误。多视角适配调用原 VirtualDroneWithMemory 和 MaskShapedFrustumFilter，每侧独立记忆，拒绝收紧时恢复该步记忆。

## 验证

不需要 GPU 的接口检查：

```bash
python -m unittest discover -s tests -v
node --check frontend/app.js
```

真实 GPU 集成检查（仅向本工程写输出）：

```bash
python scripts/gpu-smoke.py \
  --scene /path/to/scene \
  --checkpoint /path/to/sam2_hiera_large.pt --drone
```

浏览器测试需要 Node.js 和 Playwright；应用运行本身不需要 Node：

```bash
npm install --prefix /tmp/sago-web-test @playwright/test
PLAYWRIGHT_MODULE=/tmp/sago-web-test/node_modules/@playwright/test \
SAGO_TEST_SCENE=/path/to/scene node tests/browser-smoke.cjs
```

新增场景入口专项检查：`PLAYWRIGHT_MODULE=/tmp/sago-web-test/node_modules/@playwright/test node tests/scene-inputs.cjs`（同样先启动 8080 服务，使用小型合成高斯场景验证真实上传和渲染；系统选择框取消通过模拟返回验证）。

默认 Chrome 路径为 `/usr/bin/google-chrome`，可设置 `SAGO_CHROME`。先在 8080 端口启动服务。截图及测试下载保存在 `test-results/`。自动化验证的是功能行为，不是分割准确率评测。

## 当前范围

当前 Web UI 覆盖静态 3DGS 的交互分割主流程。原 splatviz 的 Python 表达式编辑、训练连接、视频导出、动态 TAG 和 RAG 问答面板未迁移为 Web 控件；这不是所有旧 widgets 的逐项复刻。核心时间跟踪及 RAG 工具源码保留在本地副本中，但没有对应 Web 接口；旧 2DGS 渲染需要另外安装其后端，当前界面不提供 2DGS。

这是本机单用户共享工作空间：只监听 `127.0.0.1`，GPU 操作串行执行，避免多请求同时修改模型。页面刷新后应重新加载场景。浏览器不会直接操纵 GPU 渲染状态；当前以 PNG 请求/响应更新图像，因此拖动流畅度与桌面 OpenGL 版本不同。部署到远程或多用户场景需要另外设计鉴权、会话隔离和图像流。

## 来源与许可

SAGO 方法与核心来源：<https://github.com/leviome/SAGO>；SAM2 跟踪实现来自原项目的 CrossViewTracker。原有源码版权头和许可证均保留。3DGS 的研究用途许可见 `gs/LICENSE.md`；SAM2/CrossViewTracker 许可证见 `sago_core/CrossViewTracker/LICENSE`（如文件名带扩展，以目录中的实际文件为准）。各 CUDA 扩展及相机函数保留各自许可，不将整个工程统一重新声明为 MIT。

右侧“三维可视化”提供默认关闭的独立开关：`3D bbox` 显示当前选区沿上方向直立的包围盒，水平两轴用最小面积外接矩形确定；地面标定或手动修改上方向后自动重算，`v-drone 轨迹与视锥` 同时显示本场景最近一次成功完成巡航的两条相机路线（金色 D0 / 青色 D1）。四棱锥显示首视角 s0（浅白色）与六个真实采样相机的位置、朝向与横纵视场角，大小按目标尺寸与观察距离调整，沿用两架 drone 的配色，与路线由同一个 checkbox 控制。线框、视锥与路线随视角投影，缩小画面可查看物体周围的路线。轨迹仅保存在当前会话，切换场景清空；取消或失败不替换上一条成功路线。该叠加显示不改变分割或导出数据。

SAM3.1 的组合模型加载已移除随后会丢弃的 tracker 视觉 backbone：仅构建 detector 所需的一份视觉网络，完整 checkpoint 仍严格加载，最终模型统一迁移到 GPU。独立 tracker 接口默认保留自身 backbone。

二维分割随提示自动更新：点/框完成后短暂合并约 160ms，文本停止输入约 450ms 后计算；中文输入法组词期间不提交。计算期间可继续编辑提示，同一时间只运行一个请求，中间变更合并为最新一份，旧请求或旧图像解码结果不会覆盖新提示。清除提示、退出模式及切换模型会使旧候选失效。二维计算不遮挡画面，状态在“清除提示”旁显示，最新结果就绪后才允许应用、三维分割或导出二维 mask。

连续分割：完成三维分割后点击“增加ID”，将当前选区保存为带编号的独立物体并隐藏，清空当前选区、文本和几何提示，自动进入下一轮分割提示，选区操作恢复为“替换选区”。后续分割会排除隐藏物体。物体列表可显示、选择并导出已保存的 ID；保存与隐藏作为一步撤销/重做。完成后点击左侧“保存分割结果”，把与原始 PLY 高斯顺序一一对应的 `int32` 向量写入当前场景的 `SAGO_knowledge.json`。`0` 表示背景或未分配，`1, 2, 3, …` 表示物体；已有归属不会被新 ID 覆盖。物体名称、显示状态、下一个编号一并保存，地面知识保留。重新加载同一场景会校验高斯数量及位置顺序的 SHA-256 后恢复 ID，并默认显示所有物体，不沿用保存时的隐藏状态；不匹配的记录会提示错误并跳过。新增、删除、显隐或撤销/重做后需再次保存。删除物体会将其位置归零，后续编号不复用已删除编号。保存前的当前选区须先通过“增加ID”或物体保存按钮分配 ID，尚未确认的选区不写入 ID 向量。

多 ID 数据位于 `scenes[PLY相对路径].segmentation`：`gaussian_ids` 为长度 N 的整数数组，`gaussian_count` 为 N，`gaussian_order_sha256` 校验位置与顺序，`objects` 保存编号、名称与显隐信息，`next_id` 保存后续编号。JSON 使用紧凑数组，原子替换写入；保存失败保留旧文件和当前会话结果。

ID 合并：在左侧物体区展开“合并两个 ID”，选择两个不同的物体，点击“合并ID”。保留较小的 ID 及其名称，把另一 ID 的全部高斯改为该 ID，删除另一条物体记录；背景和其他 ID 不变，后续编号不回收。合并后物体显示并设为当前选区，便于检查/导出；一次撤销恢复原有编号、显隐和选区。点击“保存分割结果”将合并结果写入场景知识文件。

连续编号：点击左侧“ID连续编号”，按原 ID 从小到大映射为 `1..K`，背景保持 `0`，下一次分配为 `K+1`。位置顺序、物体名称、显隐和当前选区保持不变；支持撤销/重做，完成后保存。该显式排序会回收编号空缺。

组合显示：顶部只保留“全部 / 此轮前景 / 此轮背景”。“全部”使用右侧“三维可视化 → 全部 · 组合显示”的三个开关：显示背景（ID 0）、显示物体前景、前景按 ID 着色。背景与勾选物体可独立显示，也可在同一渲染中混合，保留深度遮挡；着色时背景保留原始球谐颜色。默认显示背景和物体、不着色。物体列表的隐藏状态仍生效，“增加ID”后继续隐藏刚完成的物体；需要查看时在物体列表点击“显示”。颜色按物体唯一标识生成，连续编号后保持稳定。组合显示只影响观察和画面导出，不改变 ID 归属或当前选区；“此轮前景 / 此轮背景”独立显示当前选区及其补集，不受组合开关影响。

高斯 ID 修正：在“全部”中点击画面上方“修正”，选择“椭圆修正”（默认）或“点修正”，输入目标 ID（默认 `0`）后点击“开始选择”。`0` 表示背景，已有正整数 ID 表示归入该物体，不存在的正整数 ID 自动创建同编号物体并更新下一编号；范围为 `0..2147483647`。点修正按投影中心是否落入矩形框判定，中心坐标使用画布像素边缘坐标，Points 使用实际点像素中心；椭圆修正直接选择框内可见像素所属的高斯，与点击共用实际渲染归属图，不再用额外的 3σ 数学椭圆过滤实际渲染足迹。两者共用像素边缘取整规则，包含中心在框外但可见椭圆覆盖框内的高斯，遵循当前相机、透明度阈值和 Splats 的 0.35 缩放。两种方式均限制在当前视角可见表层：使用渲染器按深度排序、通过 alpha 阈值的首个贡献高斯作为每像素可见归属，Points 使用实际点渲染的最近深度归属；只有在选框内拥有可见像素的高斯才参与，因此不会穿透选中后方被遮挡的高斯。该规则偏保守，半透明层后方的高斯也不选中。仅修改当前已显示图层中的高斯：背景开启时可将背景高斯改为物体 ID，关闭背景可限定为物体前景；隐藏或未勾选物体不受影响。已经属于目标 ID 的高斯不重复修改。拖框后以亮黄色高亮选中的完整高斯，ID、当前选区、对象元数据和保存状态保持不变；点击“确定修正”才提交一次可撤销修改。连续左键点击可累加选中鼠标像素处最前方的可见高斯，重复点击不重复计数，点击空白或目标 ID 不清除原选择；点击拾取不受“点修正”的框选中心条件限制。“取消选中”清除预览，重新拖框替换候选；Esc 或“退出修正”放弃未确认预览并结束本轮。支持连续框选确认、撤销/重做；重新进入默认使用椭圆修正、目标为 0。空物体记录自动清理，其他 ID 不重排；已有目标物体的名称和显隐不变。修正会更新当前选区、物体计数和 ID 向量，点击“保存分割结果”写入场景知识文件。框选期间固定相机，请求绑定实际画面标识，拒绝过期画面或无效矩形；不调用 SAM。

物体改名：点击左侧物体列表中的名称，在弹窗中编辑（最多 80 个字符），按 Enter 或点击“确定”应用；取消或 Esc 放弃修改。改名同步更新多物体前景与合并下拉列表，支持撤销/重做，不改变 ID、选区或显隐状态。点击“保存分割结果”写入 `SAGO_knowledge.json`，重新打开场景后恢复名称。

控制模块折叠：左右侧栏的场景、相机、地面标定、物体、分割提示、生成三维选区、三维可视化、V-Drone 视角、分割用时和导出均可点击标题折叠/展开，支持键盘 Enter/空格。浏览器记住各模块及其子面板的展开状态，刷新后恢复；折叠不清空输入、提示或分割结果。首次使用沿用默认展开状态。

渲染模式：画面上方的 RGB / Splats / Points / Depth 与“全部 / 此轮前景 / 此轮背景”独立。RGB 为原有彩色高斯画面；Splats 将高斯按原有协方差形状显示为缩小的不透明椭圆；Points 使用 1px 高斯中心点及深度缓冲，近点遮挡远点；Depth 对有效深度的 2%–98% 分位区间做伪彩色归一化，近处暖色、远处冷色，无效深度为黑色。RGB、Splats、Points 沿用前景 ID 着色，Depth 用深度颜色。相机交互、图层勾选、修正和当前画面导出均使用当前渲染模式，不改变高斯模型或 ID 向量。加载场景默认 RGB；进入 SAM 分割提示或三点地面标定会自动切回 RGB，冻结期间不能切换模式。

界面语言：右上角按钮在 English / 中文之间即时切换。首次打开默认英语，语言偏好保存在当前浏览器的 `sago-language` 中，刷新后沿用；切换保留相机、提示、用户名称与路径以及未确认的修正高亮，不改后端 ID 或场景知识文件。覆盖界面、弹窗、计时和常见后端状态/错误提示。

## Citation and licenses

SAGO is based on **Online Segment 3D Gaussians via Launching Virtual Drones**, Liwei Liao, Rongjie Wang, and Ronggang Wang, ECCV 2026 ([paper](https://arxiv.org/abs/2607.01628)). See [CITATION.cff](CITATION.cff). Vendored components retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
