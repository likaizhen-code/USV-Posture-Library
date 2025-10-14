# USV-Posture-Library
It contians code used for USV posture-predicting

#使用步骤

1.下载整个项目文件并安装依赖:
  pip install -r requirments.txt	
	
2.制作分类标签数据集
  运行目录Classification------data-pre-label.py文件（需先更改原始csv数据集的路径）	
	
3.训练并预测
  运行目录Classification------roll predict classify(25.10.2).py文件（需先更改csv数据集的路径为制作好的分类标签数据集，291、300、355行替换为本地保存权重Pt的路径）
