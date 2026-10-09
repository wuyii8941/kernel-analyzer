# 工具的真实范围

这两个脚本不导入kernel-analyzer，不执行用户kernel，也不访问网络。

## 规则声明

安装`jsonschema`后：

```bash
python tools/validate_contracts.py examples/*_rule_contract.json
```

只检查JSON形状、条件化保证字段与证据引用是否存在。路径指向一个字符串不等于该证明或测试真实有效。

## 官方清单核账

```bash
python tools/reconcile_official_inventory.py \
  --source source.json --registered registered.json --observed observed.json \
  --routes routes.json --explanations explanations.json
```

三份输入由独立提取步骤产生，不能用本工具规则表反向生成官方源全集。每份需含：

```json
{"kind":"source","profile_id":"classic-nvidia-<version>","source_commit":"<full-sha>",
 "producer":"TableGen-expanded exporter","receipt":"<command log and source hashes>",
 "complete_for_profile":true,
 "records":[{"id":"tt.reduce::<normalized-signature>","contract_hash":"<digest of normalized contract>"}]}
```

`registered`来自该构建真正加载的注册信息。`observed`来自编译/运行日志，不声称穷举。
`routes.json`为`{"records":[{"id":"...","route":"...","implementation_status":"planned"}]}`。
`explanations.json`逐项说明source_only、registered_not_source或observed_not_registered差异，包含`difference/id/reason/evidence`。

READY_FOR_MANUAL_REVIEW只代表输入清单内部一致且目标项都有路线，不等于语言全覆盖、实现全完成或数值证明通过。
脚本拒绝缺输入、空清单、版本混用及缺失来源证明字段；不会覆写已有输出。
公共API到IR映射、TableGen的提取正确性以及该profile是否完整仍需人工/独立核查。本包没有运行这些真实全量提取。
