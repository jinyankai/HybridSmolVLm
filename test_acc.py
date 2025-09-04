import torch
from accelerate import Accelerator, __version__ as accelerate_version

# 1. 打印版本号
print(f"当前 Accelerate 库版本: {accelerate_version}")

# 2. 创建 Accelerator 实例
accelerator = Accelerator()

# 3. 使用 hasattr() 函数进行最终检查
if hasattr(accelerator, 'load_model'):
    print("✅ 成功！'accelerator' 对象中找到了 'load_model' 方法。")
    print("   您遇到的问题很可能源于IDE编辑器缓存，请尝试重启IDE。")
else:
    print("❌ 失败！'accelerator' 对象中确实没有 'load_model' 方法。")
    print("   这可能表示您的环境或安装过程存在一些特殊问题。")