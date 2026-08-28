
tc_args = {"queries": ["原始查询"]}
rewritten = "改写后的查询"

new_args = {**tc_args, "queries": [rewritten]}

assert new_args["queries"] == ["改写后的查询"], f"实际得到: {new_args}"
print("ok: 改写查询已正确覆盖")