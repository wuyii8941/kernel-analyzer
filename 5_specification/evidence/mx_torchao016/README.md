# torchao 0.16.0 MXFP8 FLOOR 对照

`observed_20261019.json` 比较独立精确参考与已安装 torchao 0.16.0 `to_mx`。范围固定为有限 f32/bf16、
最后一维每32项一块、E4M3/E5M2、`ScaleCalculationMode.FLOOR` 和非 swizzled scales。

案例包含普通值、两个不同尺度块、全零块、E8M0 code 0 的非零 f32 次正规数，以及固定随机 f32/bf16
张量。输出逐项核对全部 scale byte 和数值元素。该结果只支持这一版本接口，不把 FLOOR 解释成所有 MX
实现的唯一正确转换，也不覆盖 RCEIL、FP4、scale swizzle 或特殊值。

复跑必须写入新文件：

```bash
/data1/tzh/envs/liger/bin/python run.py --out NEW.json
```

`triton_a6000_20261019.json` 另行保存 RTX A6000 上保留 Triton `_downcast_to_mxfp` 的完整原始 E5M2
byte、E8M0 code 和解码数值。安全范围与参考逐项一致；有限饱和边界会写出 infinity，code 0 的非零
输入会写成0。每项从两种不同输出初值运行，结果相同。它是该具体 kernel 和设备路径的裁决，不外推到
其他 MX 实现。GPU 复跑入口为 `device_triton_compare.py --out NEW.json`。
