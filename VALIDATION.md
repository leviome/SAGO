# 验证记录

2026-10-17，本机 TrackAnyGaussian Python 环境、CUDA、Chrome。

- 6 项 unittest 通过：静态资源、会话/Host 校验、请求排队互斥、取消标志、错误恢复、下载目录限制、JSON 整数相机向量回归（后两项分别在相关测试方法中覆盖）。
- 真实场景：`for_seg1`，1,969,390 个高斯；从工程内的源码导入，无原项目源码路径注入。
- SAM2 Hiera Large 返回 4 个候选 mask。
- 六视角巡航、物体 PLY / 原顺序布尔 mask / PNG 导出、物体隐藏、撤销/重做、过期画面拒绝检查通过。
- 使用自动读取的第一个原始相机和通用测试框，GPU smoke 最终选择 14,348 个高斯。该数字仅用于确认流程执行，不代表目标标注质量。
- 真实浏览器端到端测试通过：场景加载、框提示、SAM2 候选、六视角细化、命名物体、撤销/重做、三维 mask 下载、二维 mask 下载、相机旋转、取消巡航不提交选区、窄屏布局；无 JavaScript 页面异常。
- `node --check frontend/app.js` 和 Python compileall 通过。

测试截图位于 `test-results/`，包括 `empty.png`、`segmentation.png`、`drone-result.png`、`mobile.png`。测试所用权重与场景为本机已有数据，未修改。所有新工程与测试产物均在 SAGO_web 内。

## 场景打开入口补充验证

- 9 项 unittest 通过，新增上传文件类型与路径限制、内容缓存复用、失败临时文件清理、系统选择任务及取消处理。
- `tests/scene-inputs.cjs` 通过：浏览器文件选择上传、600 个高斯的真实 CUDA 渲染、最近记录刷新持久化与一键重开、记录移除、拖拽加载、同内容文件复用。无页面异常。
- 系统选择框的取消返回在浏览器测试中模拟；该测试未自动操作真实桌面选择框。
- 截图：`test-results/scene-inputs.png`。

## V-Drone 逐视角预览

- 10 项 unittest 通过，包括预览图片路径与文件名限制。
- 真实 GPU 浏览器回归通过：六张视角缩略图、±45° 等角度标识、原始渲染/分割叠加放大对照、二值 mask 获取，以及原有选区、物体、撤销/重做、取消回滚流程。
- 图片来自 VirtualDroneWithMemory 本次真实跟踪输入与输出，无重复渲染或推理；PNG 写入耗时从分割计算时间中分离。
- 对照截图位于 `test-results/drone-view-detail.png`。

## 逐视角质心更新

- 13 项 unittest 通过，新增首步表面中心、后续使用最新前景、筛选拒绝后保留前景，以及无有效表面深度时禁止高斯均值回退的验证。
- 真实 CUDA 六视角回归通过：1,969,390 个高斯场景，s1 中心来源为 surface，s2 起为 foreground；各步输入候选数量等于上一步保留数量。最终选中 14,722 个高斯（仅验证执行流程，不代表分割质量）。导出、物体、撤销/重做、过期画面检查通过。
- 路径报告增加 `center`、`center_source`、`input_count`，可检查每个视角的导航中心和输入候选数量。

## SAM3 / SAM3.1 升级

- 本地原始权重：`sam3_ckpts/sam3/sam3.pt`、`sam3_ckpts/sam3.1/sam3.1_multiplex.pt`，未修改或下载权重。独立运行环境使用本机 PyTorch 2.5.1+cu121，官方源码与兼容补丁均在工程内。
- 17 项 unittest 通过。新增文本/空间提示校验、实例筛选、跟踪固定 ID、目标丢失不切换实例等回归。
- SAM3.1 真实 CUDA 六视角流程通过：文本 `yellow rubber duck` 检出 2 个候选，选中首个实例后完成六视角跟踪/筛选，最终保留 2,286 个高斯（仅确认流程，非质量指标）。逐步质心策略、导出、物体管理、撤销重做、过期画面拒绝通过。
- `tests/sam3-browser.cjs` 通过：默认 SAM3、纯文本预览与 mask 下载、SAM3 六视角与预览图、SAM3.1 文本预览及单视角应用、切回 SAM2 获得四级候选、修改文本/模型使候选失效、输入框内 M 不触发快捷键。无浏览器页面异常。
- 截图：`test-results/sam3-text.png`、`sam3-3d.png`、`sam31-text.png`。
- Python compileall 与 `node --check frontend/app.js` 通过。
- 原 `tests/browser-smoke.cjs` 的 SAM2 完整回归通过：六视角与逐视角预览/下载、物体保存、撤销重做、导出、相机交互、取消不提交选区、窄屏布局；无页面异常。

## SAM3 常驻增量跟踪（第一项性能优化）

- SAM3 / SAM3.1 三维阶段改为首视角种子初始化一次、两个方向独立持有记忆、每步仅处理当前新视角。候选状态先深拷贝，筛选接受才提交，拒绝/异常不污染已提交记忆；取消清理整条航线。
- 20 项 unittest 通过，覆盖独立分支、无历史重放、拒绝/推理失败回滚、状态内别名、SAM3 Tensor 与 SAM3.1 NestedTensor 输入、固定实例 ID 和会话清理。
- 实际视觉 backbone hook 验证：SAM3 及 SAM3.1 每次完整六视角航线均只编码 7 张图像（s0 一张＋六张新视角），种子初始化 1 次，新视角处理 6 次。预留的未来帧槽不会被检测器预取。
- SAM3 真实 CUDA 完整回归通过：六视角、质心更新、导出、物体、撤销重做及过期画面检查；本次三维阶段 2.892 秒。
- SAM3.1 同场景 `for_seg1`、同文本 `yellow rubber duck`、同 768 方形分辨率、同 5px padding：修改前三维阶段 7.519 秒，修改后 2.196 秒，约减少 70.8%（不含首次模型加载）。日志为 `outputs/sam31-before-incremental.log`、`sam31-after-incremental.log`。这是单次本机对照，不代表所有场景的加速比例。
- 常驻记忆采用完整航线的固定帧槽，旧路径按历史前缀重新建立视频；本次 SAM3.1 最终选中 3,750 个高斯，旧路径为 2,286。输出并非逐像素/逐高斯等价，测试未量化标注质量。
- 更新后的 `tests/sam3-browser.cjs` 通过：SAM3 与 SAM3.1 两条完整增量航线的 7 次真实编码断言、SAM3 取消不提交选区、取消后继续新任务、文本提示及切回 SAM2；无页面异常。
- Python compileall、前端与浏览器测试 JavaScript 语法检查通过。PNG 跨进程传输和二维预览特征缓存未在本次优化中调整。

## 模型切换加载提示框

- 23 项单元测试通过，新增无场景加载、模型复用、切换时清除旧特征及 API 加载任务检查。
- `tests/model-loading-browser.cjs` 实际切换 SAM2 → SAM3 → SAM3.1 → SAM2，通过加载中提示、禁止加载中 Escape 关闭、完成时间显示和分割计时分离检查。本机此次加载分别为 2.647、15.105、20.018、2.166 秒，仅为单次记录。
- `tests/browser-smoke.cjs` 通过：SAM2 二维预览、六视角分割及图集、保存/撤销/导出、取消回滚和响应式布局，无页面脚本错误。
- 切换时加载不要求先打开场景；默认使用 SAM2，在首次进入分割提示时加载，使用同一提示框。模型加载失败时提示框显示原因，可关闭后重试。

- 默认模型已调整为 SAM2，首次进入分割提示（勾选或 M）时触发加载提示框；切换模型仍立即加载。浏览器完整回归测试通过。

## SAM2 + GroundingDINO 文本提示

- 检测源码、Swin-T checkpoint 和 BERT 模型/tokenizer 已复制到 SAGO_web；原 SAGO 源目录和外部 Python 环境未修改。GPU 运行使用 vendored GroundingDINO 的 PyTorch deformable-attention fallback，无需编译新扩展。
- 28 项单元测试通过，覆盖检测框归一化、去重、分数/几何过滤、空检测失效、多实例候选以及第二个目标的 prediction/pointer/object score 正确传入三维记忆。
- `scripts/gpu-smoke.py --model sam2 --text 'yellow rubber duck' --drone` 在本机场景 `for_seg1` 通过，产生 12 个候选，完成六视角筛选并保留 3750 个高斯，PLY/mask/图像导出和对象历史检查通过。日志：`outputs/sam2-grounding-smoke.log`。这是一例功能验证，不是质量基准。
- `tests/sam2-text-browser.cjs` 通过：默认 SAM2、首次进入分割提示的模型加载、首次文本定位单独弹框、12 个目标/层级候选、检测器复用、六视角三维分割。截图：`test-results/sam2-text.png`、`test-results/sam2-text-3d.png`。
- `tests/browser-smoke.cjs` 原点/框提示、三维分割、对象管理、撤销/重做、导出、相机和取消回滚回归通过，无页面脚本错误。
- `tests/sam3-browser.cjs` 回归通过：SAM3 / SAM3.1 文本分割、各六视角增量跟踪（实际 7 次图像编码）、取消回滚，以及切回 SAM2，均无脚本错误。

## 三点地面与向上向量

- 33 项单元测试通过。新增三维平面方程、单位法线、点击顺序不影响朝向、反转、重复/共线/无效坐标和深度、会话失效、撤回重选、取消保留旧标定和不加载 SAM 的检查。
- `scripts/ground-smoke.py` 真实 GPU 场景通过：深度反投影三点满足平面方程，法线朝向相机半空间，应用后相机位置保持，切换原始视角保留标定，反转和取消正常。日志：`outputs/ground-smoke.log`。
- `tests/ground-browser.cjs` 通过：真实页面三点选取、重复点错误提示、撤回、地面三角形与向上箭头、应用、切换相机、反转、取消。标定期间未提交 SAM 加载任务，无页面脚本错误。截图：`test-results/ground-three-points.png`、`test-results/ground-applied.png`。
- 平面精度取决于渲染深度及用户所选表面是否共面；使用同一平面上清晰且分散的三点。标定确认后保存在场景对应的 SAGO_knowledge.json 中，重新加载自动恢复；手动改变上方向只清除当前会话标定，不删除磁盘记录。

- 模型加载成功后显示实际加载用时，1.5 秒后自动关闭，无需手动点击；失败时保留原因和关闭按钮。`tests/model-loading-browser.cjs` 实际切换 SAM3 → SAM3.1 → SAM2，加载中保持弹框、用时独立显示和成功后自动关闭检查通过。

## 地面知识持久化

- 39 项单元测试通过：标准 3DGS/独立 PLY 文件位置、JSON 字段合并、多 PLY 隔离、接受与反转自动保存、新引擎恢复、写入失败保留旧文件和选点、损坏/不一致记录拒绝应用。
- `scripts/ground-smoke.py` 在工程内复制的标准场景目录上通过真实 GPU 验证，确认文件落到场景根目录；新 Engine 从 PLY 路径加载后自动恢复已保存法线和三点。日志：`outputs/ground-persistence-smoke.log`。
- `tests/ground-browser.cjs` 在工程内独立 PLY 副本上通过确认/反转保存、取消保留旧标定、刷新页面后重新加载恢复，以及读取状态提示检查；未加载 SAM，无页面脚本错误。测试副本位于 `outputs/ground-browser-*`，没有覆盖原场景知识。
- JSON 采用 `scenes[PLY相对路径]` 保存每条记录的 `ground_plane`、`up_vector` 和 `updated_at`；使用临时文件加原子替换保留已有其他知识字段。

## 3D bbox 与 v-drone 轨迹

- 原有 39 项 Python 单元测试通过；新增 `test_visualization.py` 验证选区包围盒、更换选区、空选区及清空；`node tests/visualization.cjs` 验证投影、近平面/视口裁剪、12 条 bbox 边、独立路线与开关。
- `tests/visualization-browser.cjs` 在本地 8081 测试服务使用真实 SAM2 完成六视角分割，验证真实相机位置、bbox、开关实际绘出像素、缩放、旋转后更新投影及移动端无横向溢出。截图：`test-results/visualization-bbox-route.png`、`test-results/visualization-orbit.png`。
- 显示为透视线框叠加，不执行场景深度遮挡；bbox 包含当前选区全部高斯中心，离群点会扩大范围。路线是实际采样相机位置间的连线，不插值推断飞行路径。

## 沿上方向的包围盒与相机视锥

- 43 项 Python 单元测试通过：新增倾斜上方向下旋转物体的紧致包围盒、所有点包含性、正交轴、退化/空几何、反向法线、改变上方向使缓存失效，以及真实行向量外参对应的相机朝向和视场角检查。
- `node tests/visualization.cjs` 通过，包含四棱锥底面几何和视锥独立开关绘制检查。
- `tests/visualization-browser.cjs` 使用真实 SAM2 完成六视角巡航，验证 8 个 bbox 角点与当前上方向对齐、六个相机视锥、视锥单独显示确实改变像素、关闭后恢复原图、旋转更新投影、修改上方向重算 bbox 及移动端无横向溢出。截图：`test-results/visualization-frusta-only.png`、`test-results/visualization-bbox-route.png`、`test-results/visualization-orbit.png`。
- 视锥表示真实横纵视场比例与朝向，其显示长度按目标范围与相机到目标距离缩放，用作朝向示意，不代表实际远裁剪面。

## 合并巡航可视化开关与首视角

- 界面仅保留 `v-drone 轨迹与视锥` 一个巡航可视化开关，同时控制路线、六个巡航视锥和浅白色首视角 s0 视锥。首视角外参与视场角来自实际提示相机，按同一目标范围规则设置显示长度。
- 4 项巡航中心测试、JS 投影/线框测试通过，包含首视角数据及关闭巡航显示后无几何绘制。真实 SAM2 浏览器检查通过：无单独视锥 checkbox、首视角视锥存在、合并开关实际绘出并还原像素、相机交互和移动端布局正常。截图：`test-results/visualization-route-frusta.png`。

## SAM3.1 去除冗余视觉 backbone（第一项加载优化）

- `build_sam3_multiplex_video_model` 增加默认开启的 `with_backbone` 参数，独立 tracker 保持原行为。完整 multiplex predictor 用 `with_backbone=False, device="cpu"` 构建裸 tracker，随后沿用完整 checkpoint 严格加载及最终统一 CUDA 迁移。
- 只修改 SAM3.1 的组合构造路径；没有跳过参数随机初始化，没有更改 worker 生命周期、权重文件、SAM2 或普通 SAM3 加载路径。
- 43 项 Python 单元测试通过。真实 SAM3.1 在同一 `for_seg1` 场景、`yellow rubber duck`、768 分辨率、padding=5 下，优化前后的两张二维候选 mask（2×768×768）及最终全部 1,969,390 个高斯的选中标记逐元素完全一致。六步数量均为 `[3852, 3785, 3770, 3758, 3753, 3753]`，每条路线仍只编码 7 张图像。
- 通过实际 Engine/worker 加载计时，优化前 19.6639 秒，优化后 14.3412 秒，单次减少 5.3226 秒（27.07%）。该数值受文件缓存与系统负载影响，不代表所有机器/场景的固定收益。
- 另一次 cProfile 前后对比确认视觉 tri-backbone 构造从 2 次变为 1 次，checkpoint 都只读取 1 次，最终模型参数均为 873,185,516；严格 state_dict 加载成功。该次独立进程总耗时为 19.3780 → 16.6141 秒，性能分析开销及缓存会影响绝对值。
- 对比文件：`outputs/sam31-builder-before.json` / `sam31-builder-after.json`，对应 `.npz` 保存二维候选和最终选区；`outputs/load-profile-sam31-10-before.*` / `load-profile-sam31-10-after.*` 保存函数级加载统计。
- 下次新建 SAM3.1 worker 即读取新代码；若当前已加载 SAM3.1，切到其他模型再切回即可生效。

## 提示驱动的自动二维分割

- 移除“预览二维分割”按钮。点/框完成后约 160ms 合并请求，文本停止输入约 450ms 后触发，中文输入法 composition 期间不提交。模型切换或重新进入模式时，已有有效文本提示会自动重新分割。
- `LatestPreview` 串行处理不可中断的 GPU 请求，只保留最新提示快照。任何提示更改、清除、切换或退出都会使旧候选失效；请求及图像解码返回时校验版本。失败不自动循环重试，用户下一次修改提示重新触发。
- 二维计算不显示遮挡视口的全局忙碌层；允许继续输入点、框、文本、清除或退出。三维应用与二维导出等待最新候选就绪。候选点击立即更新提交层级，延迟解码不能覆盖后来选择。
- `node tests/auto-preview.cjs` 通过：防抖、最新快照、单请求串行、过期成功/错误、清除/退出、外部任务等待及延迟图像解码。`node tests/visualization.cjs` 与 43 项 Python 测试通过。
- `tests/auto-preview-browser.cjs` 在真实 SAM2/SAM3/SAM3.1 上通过点/背景点/框/文本持续更新、GroundingDINO 首次自动加载、文本防抖、IME、推理期间继续输入、清除与退出后丢弃延迟结果、重新进入及无页面错误。测试人为延迟任务响应，确认旧请求不会显示为新提示结果。截图：`test-results/auto-preview.png`。
- `tests/browser-smoke.cjs` 通过自动二维结果、非默认候选提交、六视角三维细化、视角预览图、物体、撤销重做、三维与二维导出、相机、取消回滚和移动布局。其余已有浏览器脚本已调整为等待自动结果，未在本次全部重跑。

连续分割（增加ID）：`python -m unittest discover -s tests -v` 共 45 项通过，新增 CPU mask 测试验证保存并隐藏为一次历史操作、撤销/重做、两轮物体排除、重新选择/显示、空选区拒绝及原保存行为。`tests/continuous-browser.cjs` 使用模拟 API 验证真实浏览器中的 P 快捷键、连续两轮增加 ID、显示编号与隐藏状态、自动进入新提示画面、清空名称/候选及恢复替换操作；无页面异常。本次未重复 GPU 模型推理测试。

连续分割运行服务排查：实际任务 `31fd6d5a7bc94343aa5fa00557bbcf1b` 返回“未知物体操作”，定位为旧 Python 服务与新前端不匹配。已补充重启提示。使用当前用户场景与选区执行真实 GPU 渲染验证：隐藏 7,225 个高斯，改变 4,651 个像素；下一轮 capture 图像与隐藏后的 frame 完全一致；撤销后原选区和原渲染完全恢复。证据见 `outputs/continuous-debug/render-check.json`、`before.png` 和 `hidden.png`。已备份选区与相机，用新后端启动并恢复；服务切换未保留旧撤销栈。

多 ID 向量与知识保存：48 项 Python 测试通过，新增长度/整数范围/唯一归属、地面合并、原子失败保护、空向量、删除归零、编号不复用、撤销/重做、数量/顺序校验与自动恢复测试。浏览器模拟流程覆盖两轮增加 ID、保存状态、刷新恢复服务当前场景及重新加载持久化记录。真实 GPU 场景验证了 1,969,390 维 int32 向量（含测试物体共 16 个 ID）写入约 4.02 MB JSON 后重新加载，向量、显隐 mask、最终渲染和地面完全一致；仅在 `outputs/instance-ids-debug/SAGO_knowledge.json` 写入验证数据，用户场景知识文件未被测试修改。实际服务已更新，恢复原有 15 个 ID、当前 3,365 个高斯的选区与相机；旧撤销历史未保留。`outputs/instance-ids-debug/persistence-check.json` 保存验证指标。

ID 合并：50 项 Python 测试通过，覆盖任意输入顺序保留较小 ID、归属向量并集、背景/其他物体保持、名称与唯一标识保留、合并显示与当前选区、编号不复用、单步撤销/重做、保存后恢复及非法 ID 无副作用。浏览器模拟验证两个下拉框、相同 ID 禁用、合并后计数与显隐、未保存状态和保存按钮。真实 1,969,390 个高斯场景验证 ID 1 与 ID 28 合并为 ID 1，共 12,210 个高斯，撤销/重做及保存后向量完全一致；原有 24 个 ID 和选区在测试后恢复，用户知识文件未被测试改写。指标位于 `outputs/instance-ids-debug/merge-check.json`。运行服务已更新，更新未保留原撤销历史。

连续编号与多物体前景：54 项 Python 测试通过，覆盖编号空缺映射、背景保留、元数据/选区/颜色稳定、后续编号更新、空列表及无变化重排、撤销/重做和保存恢复。多物体渲染测试验证任意 ID 子集、背景排除、忽略隐藏但不修改显隐、原色/ID 色、空选不调用栅格器、非法参数拒绝。浏览器模拟覆盖多选、全选/清空、原色切换与连续编号按钮。真实 1,969,390 个高斯、23 个物体场景验证重排向量与保存一致、重排前后彩色前景渲染逐像素一致、原色与 ID 色不同、空选为黑色；测试后恢复原始编号和当前选区，原知识文件未被测试改写。`outputs/instance-ids-debug/multi-check.json` 保存指标，`multi-colored.png`、`multi-original.png` 保存渲染。服务已更新，更新未保留原撤销历史。

背景高斯框选修正：57 项 Python 测试通过，覆盖行向量投影、双向拖框、深度贯穿、镜头后方/无效坐标排除、矩形校验、只修改当前勾选 ID、选区同步、空物体清理、编号保持、逐步撤销/重做、持久化及过期画面拒绝。浏览器模拟验证进入工具、锁定控件与滚轮、拖框提交实际画面标识、计数更新、Esc 退出和保存流程。真实 1,969,390 个高斯场景将测试 ID 22 的 62,111 个高斯归零，其他 ID 不变，渲染发生变化，旧画面被拒绝，撤销/重做及保存向量一致。测试后恢复原有 24 个物体与选区，未改写用户知识文件；指标位于 `outputs/instance-ids-debug/background-check.json`，渲染位于 `background-before.png` / `background-after.png`。运行服务已更新；更新未保留原撤销历史。
