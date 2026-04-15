ll triple(const vector<pair<int, int>> &edges)//start from 0
{
	int n = 0, i;
	for (auto [u, v] : edges) n = max({n, u, v});
	++n;
	vector<int> d(n), id(n), rk(n), cnt(n);
	vector<vector<int>> e(n);
	for (auto [u, v] : edges) ++d[u], ++d[v];
	iota(all(id), 0); sort(all(id), [&](int x, int y) { return d[x] < d[y]; });
	for (i = 0; i < n; i++) rk[id[i]] = i;
	for (auto [u, v] : edges)
	{
		if (rk[u] > rk[v]) swap(u, v);
		e[u].push_back(v);
	}
	ll ans = 0;
	for (i = 0; i < n; i++)
	{
		for (int u : e[i]) cnt[u] = 1;
		for (int u : e[i]) for (int v : e[u]) ans += cnt[v];
		for (int u : e[i]) cnt[u] = 0;
	}
	return ans;
}
ll quadruple(const vector<pair<int, int>> &edges)
{
	int n = 0, i;
	for (auto [u, v] : edges) n = max({n, u, v});
	++n;
	vector<int> d(n), id(n), rk(n), cnt(n);
	vector<vector<int>> e(n), lk(n);
	for (auto [u, v] : edges) ++d[u], ++d[v];
	iota(all(id), 0); sort(all(id), [&](int x, int y) { return d[x] < d[y]; });
	for (i = 0; i < n; i++) rk[id[i]] = i;
	for (auto [u, v] : edges)
	{
		if (rk[u] > rk[v]) swap(u, v);
		e[u].push_back(v);
		lk[u].push_back(v);
		lk[v].push_back(u);
	}
	ll ans = 0;
	for (i = 0; i < n; i++)
	{
		for (int u : lk[i]) for (int v : e[u]) if (rk[v] > rk[i]) ans += cnt[v]++;
		for (int u : lk[i]) for (int v : e[u]) cnt[v] = 0;
	}
	return ans;
}

map<pair<int, int>, ll> quadruple(vector<pair<int, int>> edges)
{
	int n = 0, i;
	for (auto [u, v] : edges) n = max({n, u, v});
	++n;
	map<pair<int, int>, int> ec;
	for (auto [u, v] : edges)
	{
		if (u > v) swap(u, v);
		++ec[{u, v}];
	}
	vector<ll> c;
	edges.clear();
	for (auto [_, cc] : ec) edges.push_back(_), c.push_back(cc);
	vector d(n, 0), id(d), rk(d);
	vector<ll> cnt(n);
	vector<vector<pair<int, int>>> e(n), lk(n);
	for (auto [u, v] : edges) ++d[u], ++d[v];
	iota(all(id), 0); sort(all(id), [&](int x, int y) { return d[x] < d[y]; });
	for (i = 0; i < n; i++) rk[id[i]] = i;
	i = 0;
	for (auto [u, v] : edges)
	{
		if (rk[u] > rk[v]) swap(u, v);
		e[u].push_back({v, i});
		lk[u].push_back({v, i});
		lk[v].push_back({u, i});
		++i;
	}
	int m = edges.size();
	vector<ll> ans(m);
	for (i = 0; i < n; i++)
	{
		for (auto [u, w1] : lk[i]) for (auto [v, w2] : e[u]) if (rk[v] > rk[i])
		{
			cnt[v] += c[w1] * c[w2];
		}
		for (auto [u, w1] : lk[i]) for (auto [v, w2] : e[u]) if (rk[v] > rk[i])
		{
			ans[w1] += (cnt[v] - c[w1] * c[w2]) * c[w2];
			ans[w2] += (cnt[v] - c[w1] * c[w2]) * c[w1];
		}
		for (auto [u, w1] : lk[i]) for (auto [v, w2] : e[u]) if (rk[v] > rk[i]) cnt[v] = 0;
	}
	map<pair<int, int>, ll> mp;
	for (i = 0; i < m; i++) mp[edges[i]] = ans[i];
	return mp;
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(15);
	int n, m, i;
	cin >> n >> m;
	vector<pair<int, int>> eg(m);
	cin >> eg;
	auto mp = quadruple(eg);
	for (i = 0; i < m; i++)
	{
		auto [u, v] = eg[i];
		if (u > v) swap(u, v);
		cout << mp[{u, v}] << " \n"[i + 1 == m];
	}
}

