optional<vector<int>> undirected_euler_cycle(int n, const vector<pair<int, int>> &edges)//[1,n]/[1,m], 正数表示正向，负数表示反向
{
	int i = 0;
	vector<int> rd(n + 1), ed(edges.size() + 1), r;
	vector<vector<pair<int, int>>> e(n + 1);
	for (auto [u, v] : edges)
	{
		++rd[u], ++rd[v];
		e[u].push_back({v, ++i});
		e[v].push_back({u, -i});
	}
	for (i = 1; i <= n; i++) if (rd[i] & 1) return { };
	function<void(int)> dfs = [&](int u) {
		while (e[u].size())
		{
			auto [v, w] = e[u].back();
			e[u].pop_back();
			if (ed[abs(w)]) continue;
			ed[abs(w)] = 1;
			dfs(v);
			r.push_back(w);
		}
	};
	for (i = 1; i <= n; i++) if (rd[i]) { dfs(i); break; }
	reverse(all(r));
	if (r.size() != edges.size()) return { };
	return {r};
}
optional<vector<int>> directed_euler_cycle(int n, const vector<pair<int, int>> &edges)//[1,n]/[1,m]
{
	int i = 0;
	vector<int> rd(n + 1), cd(n + 1), r;
	vector<vector<pair<int, int>>> e(n + 1);
	for (auto [u, v] : edges)
	{
		++cd[u], ++rd[v];
		e[u].push_back({v, ++i});
	}
	for (i = 1; i <= n; i++) if (rd[i] != cd[i]) return { };
	function<void(int)> dfs = [&](int u) {
		while (e[u].size())
		{
			auto [v, w] = e[u].back();
			e[u].pop_back();
			dfs(v);
			r.push_back(w);
		}
	};
	for (i = 1; i <= n; i++) if (cd[i]) { dfs(i); break; }
	reverse(all(r));
	if (r.size() != edges.size()) return { };
	return {r};
}
optional<vector<int>> undirected_euler_trail(int n, const vector<pair<int, int>> &edges)//[1,n]/[1,m], 正数表示正向，负数表示反向
{
	int i = 0;
	vector<int> rd(n + 1), ed(edges.size() + 1), r;
	vector<vector<pair<int, int>>> e(n + 1);
	for (auto [u, v] : edges)
	{
		++rd[u], ++rd[v];
		e[u].push_back({v, ++i});
		e[v].push_back({u, -i});
	}
	int odd = 0;
	for (i = 1; i <= n; i++) odd += rd[i] & 1;
	if (odd > 2) return { };
	function<void(int)> dfs = [&](int u) {
		while (e[u].size())
		{
			auto [v, w] = e[u].back();
			e[u].pop_back();
			if (ed[abs(w)]) continue;
			ed[abs(w)] = 1;
			dfs(v);
			r.push_back(w);
		}
	};
	for (i = 1; i <= n; i++) if (rd[i] & 1) { dfs(i); break; }
	if (i > n)
	{
		for (i = 1; i <= n; i++) if (rd[i]) { dfs(i); break; }
	}
	reverse(all(r));
	if (r.size() != edges.size()) return { };
	return {r};
}
optional<vector<int>> directed_euler_trail(int n, const vector<pair<int, int>> &edges)//[1,n]/[1,m]
{
	int i = 0;
	vector<int> rd(n + 1), cd(n + 1), r;
	vector<vector<pair<int, int>>> e(n + 1);
	for (auto [u, v] : edges)
	{
		++cd[u], ++rd[v];
		e[u].push_back({v, ++i});
	}
	int diff = 0;
	for (i = 1; i <= n; i++)
	{
		if (abs(rd[i] - cd[i]) > 1) return { };
		if (rd[i] != cd[i]) ++diff;
	}
	if (diff > 2) return { };
	function<void(int)> dfs = [&](int u) {
		while (e[u].size())
		{
			auto [v, w] = e[u].back();
			e[u].pop_back();
			dfs(v);
			r.push_back(w);
		}
	};
	for (i = 1; i <= n; i++) if (cd[i] > rd[i]) { dfs(i); break; }
	if (i > n)
	{
		for (i = 1; i <= n; i++) if (cd[i]) { dfs(i); break; }
	}
	reverse(all(r));
	if (r.size() != edges.size()) return { };
	return {r};
}
optional<vector<int>> mixed_euler_cycle(int n, const vector<tuple<int, int, bool>> &edges)//true: 单向，false：双向
{
	int i = 0, j, m = edges.size();
	vector<int> rd(n + 1), cd(n + 1), r, rev(m + 1);
	vector<vector<pair<int, int>>> e(n + 1);
	for (auto [u, v, d] : edges) ++cd[u], ++rd[v];
	for (i = 1; i <= n; i++) if (cd[i] + rd[i] & 1) return { };
	vector<tuple<int, int, ll>> eg;
	for (auto [u, v, d] : edges) if (!d) eg.push_back({u, v, 1});
	ll sum = 0;
	for (i = 1; i <= n; i++)
	{
		int d = (cd[i] - rd[i]) / 2;
		if (d > 0) eg.push_back({0, i, d}), sum += d;
		else if (d < 0) eg.push_back({i, n + 1, -d});
	}
	auto [w, res] = net::max_flow(n + 1, eg, 0, n + 1);
	if (w != sum) return { };
	vector<pair<int, int>> G(m);
	for (i = j = 0; i < m; i++)
	{
		auto [u, v, d] = edges[i];
		if (d || !res[j++]) G[i] = {u, v};
		else G[i] = {v, u}, rev[i + 1] = 1;
	}
	auto ans = directed_euler_cycle(n, G);
	if (!ans) return ans;
	for (int &x : *ans) if (rev[x]) x = -x;
	return ans;
}
