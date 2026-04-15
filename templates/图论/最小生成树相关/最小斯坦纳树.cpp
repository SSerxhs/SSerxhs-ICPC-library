ll steiner(int n, const vector<tuple<int, int, ll>> &eg, vector<int> id)//[0,n)
{
	using pa = pair<ll, int>;
	int m = id.size(), i;
	vector f(1 << m, vector<ll>(n, inf));
	for (i = 0; i < m; i++) f[1 << i][id[i]] = 0;
	vector<vector<pair<int, ll>>> e(n);
	for (auto [u, v, w] : eg) e[u].push_back({v, w}), e[v].push_back({u, w});
	for (int S = 1; S < 1 << m; S++)
	{
		auto &d = f[S];
		priority_queue<pa, vector<pa>, greater<pa>> q;
		for (i = 0; i < n; i++)
		{
			for (int T = S - 1 & S; T; T = T - 1 & S) cmin(d[i], f[T][i] + f[S ^ T][i]);
			if (d[i] != inf) q.push({d[i], i});
		}
		while (q.size())
		{
			auto [_, u] = q.top(); q.pop();
			if (_ != d[u]) continue;
			for (auto [v, w] : e[u]) if (cmin(d[v], d[u] + w)) q.push({d[v], v});
		}
	}
	return *min_element(all(f.back()));
}
pair<ll, vector<int>> steiner_construct(int n, const vector<tuple<int, int, ll>> &eg, vector<int> id)//[0,n)
{
	using pa = pair<ll, int>;
	int m = id.size(), i;
	vector f(1 << m, vector<ll>(n, inf));
	vector pre(1 << m, vector(n, pair{-1, -1}));
	for (i = 0; i < m; i++) f[1 << i][id[i]] = 0;
	vector<vector<tuple<int, ll, int>>> e(n);
	i = 0;
	for (auto [u, v, w] : eg) e[u].push_back({v, w, i}), e[v].push_back({u, w, i++});
	for (int S = 1; S < 1 << m; S++)
	{
		auto &d = f[S];
		priority_queue<pa, vector<pa>, greater<pa>> q;
		for (i = 0; i < n; i++)
		{
			for (int T = S - 1 & S; T; T = T - 1 & S)
				if (cmin(d[i], f[T][i] + f[S ^ T][i]))
					pre[S][i] = {-2, T};

			if (d[i] != inf) q.push({d[i], i});
		}
		while (q.size())
		{
			auto [_, u] = q.top(); q.pop();
			if (_ != d[u]) continue;
			for (auto [v, w, id] : e[u]) if (cmin(d[v], d[u] + w))
			{
				q.push({d[v], v});
				pre[S][v] = {u, id};
			}
		}
	}
	vector<int> chosen;
	int S = (1 << m) - 1, u = min_element(all(f[S])) - f[S].begin();
	auto dfs = [&](auto &&dfs, int S, int u) {
		auto [x, y] = pre[S][u];
		while (x >= 0)
		{
			u = x;
			chosen.push_back(y);
			tie(x, y) = pre[S][u];
		}
		if (x == -1) return;
		dfs(dfs, y, u); dfs(dfs, S ^ y, u);
	};
	dfs(dfs, S, u);
	sort(all(chosen));
	return {f[S][u], chosen};
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(15);
	int n, m, q, i;
	cin >> n >> m;
	vector<tuple<int, int, ll>> eg(m);
	cin >> eg >> q;
	vector<int> id(q);
	cin >> id;
	auto [ans, eid] = steiner_construct(n, eg, id);
	cout << ans << ' ' << eid.size() << '\n' << eid << endl;
}

