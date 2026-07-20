# exp407 formal聚焦盲审

## 结论

`0 BLOCKER / 0 HIGH`，授权启动唯一fresh `exp407-p0b-iso-teacher-v1`。

## 核验

- manifest SHA=`3932125980989a634df87cb71904e8d2a4772e9bae98ea1dcfb8def35ca70571`；
- schema、execution、source commit、五个源码SHA、runtime、inputs、arguments和thresholds与runner精确闭合；
- preflight result/COMPLETE/cache/started/seal SHA匹配，8/8 validity PASS，两个failure路径不存在；
- formal重新编码15,618张train图，独立选择target与2,000对diagnostic，重算full-train slot MAD和wrong-mask匹配；
- formal不加载preflight cache、scale、pair或donor plan，只验证preflight cache SHA；
- formal output/started/failure路径fresh，固定MMPOSE-ABU runtime闭合；
- scientific controls、阈值和validity/scientific gates未相对exp406删改。

审查过程只读、未启动GPU。
