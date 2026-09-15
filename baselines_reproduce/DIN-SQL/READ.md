### 本方法一共三个有效阶段，每个阶段一个py，通过入参区别加/不加RC

1， shcema_linking.py

正常版：全量scheme + 各种其他prompt  -> baselines_reproduce/DIN-SQL/bird_dev/schema_linking/qwen38_result.json

RC版： **filtered shcema** + 各种其他prompt -》 baselines_reproduce/DIN-SQL/bird_dev/schema_linking/qwen38_result_filtered_meta.json


2. difficulty_decomposition

这是本方法的一个中间环节，不区分RC，直接执行difficulty decomposition.py即可 （但我没处理并发逻辑，需要的话加一下）


3. sql_generation.py

正常版：全量schcema + 正常版linking results + 正常difficulty decomposition+ 其他prompt -> baselines_reproduce/DIN-SQL/bird_dev/sql_generation/qwen38_result.json

RC版：全量schcema + 正常版linking results + 正常difficulty decomposition + **RC** + 其他prompt -> baselines_reproduce/DIN-SQL/bird_dev/sql_generation/qwen38_result_rc.json


4. sql_correction.py

正常版：全量schcema + 正常版sql results + 其他prompt -> code/baselines_reproduce/DIN-SQL/bird_dev/self_correction/qwen38_result.json

RC版： 全量schcema + 正常版sql results + **RC** + 其他prompt -> code/baselines_reproduce/DIN-SQL/bird_dev/self_correction/qwen38_result_rc.json


### 测评脚本：

evaluation_linking : 第一阶段
evaluation_sql : 第2+3阶段

均输出csv。


### 数据集

bird-dev已全部跑完，spider dev 和 test 文件夹即进度，接着跑即可。