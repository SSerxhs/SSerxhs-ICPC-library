vector<pair<int, int>> get_tree(vector<int> a)
{
	vector<pair<int, int>> edges;
	sort(all(a), [&](int u, int v) { return dfn[u] < dfn[v]; });
	vector<int> st(a.size() + 2);
	int tp = 0;
	auto ins = [&](int u) {
		if (tp == 0)
		{
			st[tp = 1] = u;
			return;
		}
		int v = lca(st[tp], u);
		while (tp > 1 && dep[v] < dep[st[tp - 1]])
		{
			edges.emplace_back(st[tp - 1], st[tp]);
			--tp;
		}
		if (dep[v] < dep[st[tp]]) edges.emplace_back(v, st[tp--]);
		if (!tp || st[tp] != v) st[++tp] = v;
		st[++tp] = u;
	};
	if (a[0] != 1) st[tp = 1] = 1;//先行添加根节点
	for (int u : a) ins(u);
	if (tp) while (--tp) edges.emplace_back(st[tp], st[tp + 1]);//回溯
	return edges;
}

