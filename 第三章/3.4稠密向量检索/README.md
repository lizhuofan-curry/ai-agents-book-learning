# 实验3-4：稠密向量检索

日期：2026-10-17。3-3两条合成日志学习范围已收尾。3-4先做向量距离与精确检索基础，再接真实文本嵌入和ANNOY/HNSW对比，不运行3-2剩余190组。

## 第一步要看懂什么

稠密检索包含两件事：嵌入模型将文本变成向量；搜索算法从已有向量中找近邻。第一步只解释后半部分的距离和精确排序，使用6个手工指定的二维坐标，让你能直接看图。

查询q=(1,0)，向右。A与q同方向，B略偏，C呈45度，D垂直，F反向。将向量长度归一化后计算点积，就是余弦相似度；相似度越大，方向越近。这里的“余弦距离”定义为1−相似度。

这是人工坐标练习，不给向量贴上猫、法律等文本标签，也不声称证明模型懂语义。精确检索会遍历所有6个向量；后续近似索引才需要检查是否漏掉精确近邻。

## 第一步已完成：二维向量精确检索

```powershell
& 'E:\AI_Agents_Book\.venv\Scripts\python.exe' 'E:\AI_Agents_Book\第三章\3.4稠密向量检索\我的代码\01_向量距离与精确检索.py'
```

模型请求0、网络请求0，不需要下载新模型或安装依赖；使用共用环境已有numpy。运行会打印六个候选的相似度、距离及Top-3，并保存record.json和可打开的向量方向.svg。输出发来后核对，再讲解每一行如何计算。

用户已运行并核对[正式记录](运行结果/exact_vectors_20261017_194826_596079/record.json)与独立[核对记录](运行结果/exact_vectors_20261017_194826_596079/核对记录.json)。顺序为A、B、C、D、E、F，Top-3为A、B、C。A同向，余弦1、距离0；D垂直，余弦0、距离1；F反向，余弦−1、距离2。B相似度0.993884、C为0.707107。公式重算、源文件及SVG哈希均通过。原脚本与原始结果保持不变。

加--preview仅核验材料、不创建运行目录。之前[10项离线检查](离线检查_20261017.json)属于准备验证，与上述用户实测分开。

## 下一步：合成向量ANNOY/HNSW对照

准备了[02入口](我的代码/02_合成向量ANNOY与HNSW对照.py)。沿用官方CLI的3000条底库、128维、100次查询、数据种子42及Top-10设定，归一化float32后，用官方dense_rank建立精确基准，调用官方索引类检索。比较ANNOY的10棵树、HNSW的M=16和ef_construction=100；同一张HNSW图依次使用ef_search=20、200，观察扩大搜索范围能否减少遗漏。

Recall@10表示每次找回的精确Top-10中有几条命中。例如命中8条就是80%；这是索引召回，不是文本相关性。保留逐查询ID、距离、耗时、精确答案、数组及SHA256，可逐项复算。ANNOY angular距离为sqrt(2−2cos)，HNSW cosine距离为1−cos，两种距离数值不能直接混比。

这是教学适配：HNSW容量设为实际底库大小，原生查询线程设1；ANNOY沿用官方重建方法，使用库默认建树线程；原生索引随机种子未覆盖。数据种子固定不意味着所有库构建过程位级可复现。精确查询时间包含逐条点积和全排序，ANN时间包含官方Python封装。只测一次本机耗时，不据此声称通用速度排名。

### 先补齐C++编译环境

2026-10-17检查：共用Python为3.11、64位；annoy、hnswlib、colorlog未安装。三个依赖已作为可选chapter3-index组写入根pyproject.toml和uv.lock，既有锁定包未改变。uv安装预览只新增这三个包；实际安装在构建annoy时失败，错误为“Microsoft Visual C++ 14.0 or greater is required.”，未检测到已注册的Visual Studio产品。详见[环境检查](索引环境检查_20261017.json)。

1. 从[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)安装编译工具。按微软[安装说明](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)，选择“使用C++的桌面开发”，包含MSVC x64/x86工具和Windows SDK。这里只需要Build Tools，无需完整Visual Studio编辑器。安装结束后重新打开PowerShell。
2. 在项目目录同步可选依赖：

```powershell
& 'E:\Hermes\bin\uv.exe' sync --project 'E:\AI_Agents_Book' --locked --inexact --group chapter3-index --cache-dir 'E:\AI_Agents_Book\.uv-cache'
```

3. 检查实际原生库导入：

```powershell
& 'E:\AI_Agents_Book\.venv\Scripts\python.exe' 'E:\AI_Agents_Book\第三章\3.4稠密向量检索\我的代码\02_合成向量ANNOY与HNSW对照.py' --check-only
```

看到ANN_IMPORT_CHECK_ONLY后再去掉--check-only运行同一入口，保存正式比较记录。脚本无网络、无自动安装或模型请求。依赖没准备好时退出，不创建正式运行目录。

[8项离线检查](索引对照离线检查_20261017.json)通过，覆盖源文件、已知方向的解析答案、小型数据复现、召回计算、重复ID拒绝及部分记录保留等；当前原生库尚未安装，尚无真实ANNOY/HNSW结果。先反馈安装和导入检查输出，再核对并运行评估。

## 固定材料与边界

官方资料固定到dbc046eb896ac4e39aa19c7774c8bf49583b89a6，共27份公开文件，逐文件Git blob和SHA256核验，见[清单](官方资料/manifest.json)。原作者validation记录不是本人结果。

入口只从已校验的官方cli.py抽取dense_rank纯函数，用于归一化后精确排序；不导入该模块顶层的索引依赖或启动模型下载。人工二维坐标属于本地教学材料，不替代官方合成向量ANN对比或真实嵌入评测。

环境检查发现numpy、torch、transformers已有，sentence-transformers、annoy、hnswlib尚未安装。第一步无需索引包；第二步只补齐上述三个依赖，不安装整个HTTP服务或下载BGE-M3。真实文本嵌入的模型选择及下载留到下一阶段。

## 后续顺序

1. 给定向量的精确近邻：用户实测和核对已完成。
2. 固定合成向量比较ANNOY/HNSW与精确基准：代码及离线检查已准备，先补齐C++工具和索引库，尚未实测。
3. 用同一嵌入模型将文本变成向量，再检验文本相关性；模型效果与索引近似误差分开。

完成第一步只能记为向量基础完成，不能标整个3-4完成、文本语义检索已验证或索引性能已验证。见[实验记录](实验记录.md)。
