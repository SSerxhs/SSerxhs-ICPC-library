template<class T> vector<vector<T>> min_cut(int n, const vector<tuple<int, int, T>> &edges)//[0,n)
{
	int m = edges.size(), i, s, t, cnt = 0;
	vector<int> fir(n, -1), nxt(m * 2, -1), fc(n), q(n);
	vector<pair<int, T>> e(m * 2);
	vector<tuple<T, int, int>> eg;
	auto add = [&](int u, int v, T w) {
		e[cnt] = {v, w};
		nxt[cnt] = fir[u];
		fir[u] = cnt++;
	};
	for (auto [u, v, w] : edges) add(u, v, w), add(v, u, w);
	auto E = e;
	auto bfs = [&]() {
		fill(all(fc), 0);
		int ql = 0, qr = 0, u, i;
		fc[q[0] = s] = 1;
		while (ql <= qr)
		{
			u = q[ql++];
			for (int i = fir[u]; i != -1; i = nxt[i])
				if (auto &[v, w] = e[i]; w && !fc[v]) fc[q[++qr] = v] = fc[u] + 1;
		}
		return fc[t];
	};
	function<T(int, T)> dfs = [&](int u, T maxf) {
		if (u == t) return maxf;
		T j = 0, k;
		for (int i = fir[u]; i != -1; i = nxt[i])
			if (auto &[v, w] = e[i]; w && fc[v] == fc[u] + 1 && (k = dfs(v, min(maxf - j, w))))
			{
				j += k;
				w -= k;
				e[i ^ 1].second += k;
				if (j == maxf) return j;
			}
		fc[u] = 0;
		return j;
	};
	function<void(vector<int>)> solve = [&](vector<int> id) {
		static mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
		if (id.size() <= 1) return;
		vector<int> u(2);
		sample(all(id), u.begin(), 2, rnd);
		s = u[0], t = u[1], e = E;
		T ans = 0;
		while (bfs()) ans += dfs(s, numeric_limits<T>::max());
		auto it = partition(all(id), [&](int u) { return fc[u]; });
		eg.emplace_back(ans, s, t);
		solve(vector(id.begin(), it));
		solve(vector(it, id.end()));
	};
	solve(range(0, n));
	sort(all(eg), greater<>());
	vector<basic_string<int>> ver(n);
	vector ans(n, vector<T>(n));
	vector<int> f(n);
	for (i = 0; i < n; i++) ver[i] = {f[i] = i};
	function<int(int)> getf = [&](int u) { return f[u] == u ? u : f[u] = getf(f[u]); };
	for (auto [w, u, v] : eg)
	{
		u = getf(u);
		v = getf(v);
		for (int w1 : ver[u]) for (int w2 : ver[v]) ans[w1][w2] = ans[w2][w1] = w;
		ver[u] += ver[v];
		f[v] = u;
	}
	return ans;
}

