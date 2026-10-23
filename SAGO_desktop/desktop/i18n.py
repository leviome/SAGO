import json
import re
from pathlib import Path
from PySide6.QtCore import QObject, Signal

ZH = {
 'Parallel exports':'并行任务数',
 'Each parallel job uses its own GPU memory':'每个并行任务独立占用显存',
 'Export object meshes':'导出物体 Mesh',
 'Choose objects; export one OBJ per ID. Selected objects are reconstructed in parallel; this takes longer than segmentation.':'选择物体，每个 ID 单独导出 OBJ；将并行重建所选物体，耗时长于分割。',
 'Select none':'全不选',
 'Mesh exporting directory':'Mesh exporting 目录',
 'Mesh exporting Python executable':'Mesh exporting Python 路径',
 'Mesh exporting network configuration':'Mesh exporting 网络配置',
 'Training steps':'训练步数',
 'Mesh resolution':'网格分辨率',
 'View resolution':'视图分辨率',
 'Views per orbit':'每圈视图数',
 'Four orbits at 0°, 20°, 30°, 60°. Leave paths blank to use environment variables or automatic discovery.':'四圈俯仰角：0°、20°、30°、60°。留空路径可使用环境变量或自动发现。',
 'Select at least one foreground object':'请选择至少一个前景物体',
 'Mesh export complete':'Mesh 导出完成',
 'Exported {0} meshes · {1} s':'已导出 {0} 个 Mesh · 用时 {1} 秒',
 '{0} vertices · {1} faces':'{0} 顶点 · {1} 面',

 'API settings':'API 设置',
 'API model':'API 模型',
 'API key (this session only)':'API 密钥（仅本次会话）',
 'Requires Chat Completions and tool calls. Leave blank to use backend environment variables. Keys are not saved.':'支持 Chat Completions 与工具调用；留空可使用后端环境变量。密钥不保存。',
 'Test API connection':'测试 API 连接',
 'Generate object descriptions with a vision API':'通过视觉 API 生成物体描述',
 'Enabling this sends four-view images to the configured API; questions send object descriptions and spatial facts.':'启用后将发送四视图到所填 API；提问时发送物体描述和空间事实。',
 'Build / update spatial knowledge':'建立 / 更新空间知识',
 'Uses all object IDs and also saves IDs. Distances are in scene units, not necessarily metres.':'使用全部物体 ID；同时保存 ID。距离单位为场景单位，不一定是米。',
 'Spatial question':'空间问题',
 'For example: which object is closest to the table?':'例如：离桌子最近的物体是什么？',
 'Ask about the scene':'询问场景',
 'API connected: {0}':'API 连接成功：{0}',
 'Spatial knowledge saved for {0} objects: {1}':'已保存 {0} 个物体的空间知识：{1}',
 'Model: {0} · Cited IDs: {1} · {2} s':'模型：{0} · 引用 ID：{1} · 用时 {2} 秒',

 'Clear search highlight':'取消搜索高亮',
 'Semantic knowledge':'语义知识库',
 'Build / update semantic knowledge':'建立 / 更新语义知识库',
 'All object IDs · Four views · GIF elevation and auto-fit distance · Also saves IDs':'所有物体 ID · 四个视图 · 沿用 GIF 俯仰角与自动取景距离 · 同时保存 ID',
 'Describe an object (English recommended)':'描述要查找的物体（建议英文）',
 'Find object IDs':'文本检索 ID',
 'Indexed {0} objects; updated {1}. Saved to {2}':'已索引 {0} 个物体，更新 {1} 个；已保存到 {2}',
 'Rebuild the index for IDs: {0}':'以下 ID 需要更新索引：{0}',
 'Ranked by maximum four-view cosine similarity; scores are not probabilities':'按四视图最高余弦相似度排序，分数不是概率',

 'Cube completion':'Cube 补全','Cylinder completion':'圆柱体补全',
 'Completion added {0} Gaussians':'补全新增 {0} 个高斯',
 'Exit application (Esc)':'退出程序 (Esc)',
 'Black / White':'黑 / 白',
 'Drag to rotate · Right/Shift drag to pan · WASD to move · Scroll to zoom':'拖动旋转 · 右键/Shift 拖动平移 · WASD 漫游 · 滚轮缩放',
 'Loading SAM2…':'正在加载 SAM2…',
    'Retry download':'重试下载',
 'Downloading SAM2 checkpoint…':'正在下载 SAM2 权重…',
 'Verifying SAM2 checkpoint…':'正在校验 SAM2 权重…',
 'Waiting for another SAM2 download…':'正在等待另一个 SAM2 下载任务…',
 'Download interrupted; retrying…':'下载中断，正在重试…',
 'SAM2 checkpoint ready':'SAM2 权重已就绪',
 'Delete object and return its Gaussians to background (ID 0)':'删除物体，并将对应高斯归为背景（ID 0）',
 'Adversarial background':'对抗性背景',
 'Choose black or white from the unpadded mask boundary; disable to use the manual background color.':'按膨胀前 mask 的边缘明暗选择黑底或白底；关闭后使用手动背景色。',
 'Background color R / G / B (0–1)':'背景颜色 R / G / B（0–1）','Background color':'背景颜色','Choose color…':'选择颜色…',
 'Auto':'自动',
 'Cancel current round (C)':'取消本轮 (C)',
 'Cancelling round after the current GPU step…':'等待当前 GPU 步骤结束后取消本轮…',
 'Round cancelled. Prompts cleared; saved IDs kept.':'本轮已取消，提示已清除，已保存 ID 保留。',
 'Clear recent':'清空最近场景',
 'Three points':'三点确定地面', 'Hidden':'隐藏',
 'Open PLY…':'打开 PLY…', 'Open scene folder…':'打开场景目录…', 'Recent scenes':'最近场景',
 'Scene':'场景','Camera':'相机','Ground calibration':'地面标定','Objects':'物体','Segmentation prompts':'分割提示',
 '3D segmentation':'三维分割','3D visualization':'三维可视化','Export':'导出','Timing':'用时',
 'Enter segmentation prompts (M)':'进入分割提示 (M)','Point':'点','Box':'框',
 'Clear prompts':'清除提示','Add ID':'增加 ID','Save selection':'保存当前选区','Save IDs':'保存分割结果',
 'Merge IDs':'合并 ID','Renumber IDs':'ID 连续编号','All':'全部','Current foreground':'此轮前景','Current background':'此轮背景',
 'Correction':'修正','Confirm correction':'确定修正','Discard selection':'取消选中','Ellipse correction':'椭圆修正',
 'Point correction':'点修正','Target ID (0 = background)':'目标 ID（0 为背景）','Show background (ID 0)':'显示背景（ID 0）',
 'Show object foreground':'显示物体前景','Color foreground by ID':'前景按 ID 着色','Select all':'全选','Select none':'全不选',
 'V-drone route and frustums':'V-drone 轨迹与视锥','3D bbox':'3D bbox','Cancel':'取消','Undo':'撤销','Redo':'重做',
 'Clear selection':'清空选区','Three ground points':'三点确定地面','Flip up':'反转向上','Undo point':'撤回一点',
 'Reset points':'重新选点','Apply ground':'应用地面','Cancel ground':'取消地面标定','FOV':'视场角','Render size':'渲染长边',
 'Up X / Y / Z':'向上 X / Y / Z','Target X / Y / Z':'观察中心 X / Y / Z','Update camera':'更新相机',
 'Original camera':'原始场景相机','Padding (px)':'扩展边缘（px）','Selection operation':'选区操作',
 'Replace':'替换','Add':'加入','Subtract':'减去','Intersect':'取交集','Model':'模型','Name':'名称',
 'Export object PLY':'导出物体 PLY','Export 3D mask':'导出三维 mask','Export image':'导出当前画面',
 'Export 2D mask':'导出二维 mask','Export all object GIFs':'导出所有物体 GIF','Export current GIF':'导出当前 GIF',
 'View GIFs':'查看 GIF','GIF settings':'GIF 设置','Resolution':'分辨率','Frames per orbit':'每圈帧数',
 'Orbit elevation (degrees)':'轨道俯仰角（度）','One orbit · 20 FPS · White background':'单圈环绕 · 20 FPS · 白色背景',
 'Output folder':'输出目录','Preferences…':'设置…','Help':'帮助','Ready':'就绪','Loading model…':'正在加载模型…',
 'Working…':'处理中…','Delete':'删除','Rename':'重命名','Select':'选择','Hide / Show':'隐藏 / 显示',
 'V-drone views':'V-drone 视角','Render':'原始渲染','Overlay':'分割叠加','Binary mask':'二值 mask',
 'Save as…':'另存为…','Close':'关闭','Files saved to:':'文件保存到：','Open output folder':'打开输出目录',
 'Save':'保存','Settings take effect after restarting the application.':'设置在重启应用后生效。',
 'Unsaved segmentation':'未保存的分割结果','Save segmentation before closing?':'关闭前保存分割结果？',
 'Discard':'放弃','Wait for current task':'等待当前任务完成','Cancel task and close after it finishes?':'取消任务并在任务结束后关闭？',
 'No matching mask. Adjust the prompts.':'未找到匹配物体，请调整提示。',
 'Exit prompts or calibration first.':'请先退出分割提示或地面标定。',
 'Open a Gaussian PLY to begin':'打开一个高斯 PLY 开始',
 'Choose two different object IDs.':'请选择两个不同的物体 ID。',
 'Left: foreground · Right: background · Drag: box':'左键：前景 · 右键：背景 · 拖动：框选',
 'Drag to rotate · Shift/right drag to pan · Scroll to zoom':'拖动旋转 · Shift/右键拖动平移 · 滚轮缩放',
 'Native Linux Gaussian segmentation':'Linux 原生三维高斯分割',
}


class Translator(QObject):
    changed = Signal()
    def __init__(self, language='en'):
        super().__init__();self.language=language
        self.backend=json.loads((Path(__file__).parent/'backend_messages.json').read_text())
    def text(self,key):return ZH.get(key,key) if self.language=='zh' else key
    def backend_text(self,message):
        mappings={value:key for key,value in self.backend.items()} if self.language=='zh' else self.backend
        if message in mappings:return mappings[message]
        for source,target in mappings.items():
            if '{' not in source:continue
            pattern=re.escape(source)
            keys=re.findall(r'\{(\d+)\}',source)
            for key in keys:pattern=pattern.replace(re.escape('{'+key+'}'), '(.*?)', 1)
            match=re.fullmatch(pattern,message)
            if match:
                for key,value in zip(keys,match.groups()):target=target.replace('{'+key+'}',value)
                return target
        return message
