# manga-remake
## 描述
本项目旨在实现：高清原图 + 已翻译但画质差的图 => 高清已翻译图片。<br>
原理：训练模型生成对话框的蒙版（MASK），调用PS自动批量执行。<br><br>
实现效果：<br>
<img src="README_img/view1.jpg" alt="图片匹配" width="300px"><br>
<img src="README_img/view2.jpg" alt="PSD生成" width="300px">

## 使用
### 前提
需要提前安装PS，本人使用的是PS2022，其他较新的版本理论上也可以（未测试）
### 方法一
下载打包好的文件（仅可调用），解压直接运行`start.bat`

### 方法二
下载本项目，安装环境（可重新训练模型），然后运行：
```bash
pip install -r requirements.txt
```
如果生成的requirements有缺漏，那就自行安装吧。<br>
注意：如果`pillow-avif-plugin`安装时编译失败，可用去找镜像源里找现成的`whl`包。


## 叠甲
1. 代码苦手，感谢豆包和Gork~
2. 没有设计美感，UI凑合着用吧