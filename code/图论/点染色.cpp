vector<int> chromatic_number(int n, const vector<pair<int, int>> &edges)//[0,n)
{
	vector r(n, -1), cur(n, -1);
	vector<vector<int>> e(n);
	int ans = 0, i;
	for (auto [u, v] : edges) e[u].push_back(v), e[v].push_back(u);
	for (i = 0; i < n; i++) ans = max(ans, (int)e[i].size());
	ans += 2;
	vector p(n, vector(ans, 0));
	function<void(int)> dfs = [&](int u) {
		int col = u ? *max_element(cur.begin(), cur.begin() + u) + 1 : 0;
		if (col >= ans) return;
		if (u == n)
		{
			r = cur;
			ans = col;
			return;
		}
		int i;
		for (int i = 0; i <= col; i++) if (!p[u][i])
		{
			cur[u] = i;
			for (int v : e[u]) ++p[v][i];
			dfs(u + 1);
			for (int v : e[u]) --p[v][i];
		}
	};
	dfs(0);
	return r;
}
